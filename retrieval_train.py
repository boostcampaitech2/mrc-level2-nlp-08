import argparse

from numpy.lib.function_base import gradient
from retrieval_model import BertEncoder, ElectraEncoder, RobertaEncoder
from utils_mrc import (
    InBatchNegativeRandomDataset,
    seed_everything,
    get_tensor_for_dense,
    get_tensor_for_dense_temp,
    get_tensor_for_dense_negative,
)
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AdamW,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tqdm import tqdm
import random
import wandb


def to_cuda(batch):
    return tuple(t.cuda() for t in batch)


def train(args, p_encoder, q_encoder, train_dataset, valid_dataset):
    wandb.login()
    wandb.init(
        project="retrieval_aug",
        entity="chungye-mountain-sherpa",
        name="retrieval_train",
        group="klue-bert",
    )

    # Dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.per_device_train_batch_size,
        drop_last=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=args.per_device_eval_batch_size
    )

    # Optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in p_encoder.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in p_encoder.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
        {
            "params": [
                p
                for n, p in q_encoder.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in q_encoder.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )

    t_total = (
        len(train_dataloader)
        // args.gradient_accumulation_steps
        * args.num_train_epochs
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Start training!
    global_step = 0
    if torch.cuda.is_available():
        p_encoder.cuda()
        q_encoder.cuda()

    p_encoder.zero_grad()
    q_encoder.zero_grad()
    torch.cuda.empty_cache()

    best_loss = 9999  # valid_loss
    num_epoch = 0

    for _ in range(int(args.num_train_epochs)):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        num_epoch += 1

        train_loss = 0
        train_acc = 0
        train_step = 0
        for batch in epoch_iterator:
            train_step += 1
            q_encoder.train()
            p_encoder.train()

            if torch.cuda.is_available():
                batch = tuple(t.cuda() for t in batch)

            p_inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            q_inputs = {
                "input_ids": batch[3],
                "attention_mask": batch[4],
                "token_type_ids": batch[5],
            }

            p_outputs = p_encoder(**p_inputs)  # (batch_size, emb_dim)
            q_outputs = q_encoder(**q_inputs)  # (batch_size, emb_dim)

            # Calculate similarity score & loss
            sim_scores = torch.matmul(
                q_outputs, torch.transpose(p_outputs, 0, 1)
            )  # (batch_size, emb_dim) x (emb_dim, batch_size) = (batch_size, batch_size)

            # target: position of positive samples = diagonal element
            targets = torch.arange(0, args.per_device_train_batch_size).long()
            if torch.cuda.is_available():
                targets = targets.to("cuda")

            # print(sim_scores)
            # print()
            # print(targets)
            sim_scores = F.log_softmax(sim_scores, dim=1)
            loss = F.nll_loss(sim_scores, targets)
            train_loss += loss.item()

            _, preds = torch.max(sim_scores, 1)  #
            train_acc += (
                torch.sum(preds.cpu() == targets.cpu())
                / args.per_device_train_batch_size
            )

            loss.backward()
            optimizer.step()
            scheduler.step()
            q_encoder.zero_grad()
            p_encoder.zero_grad()
            global_step += 1

            # validation
            if train_step % 100 == 0:
                valid_loss = 0
                valid_acc = 0
                v_epoch_iterator = tqdm(valid_dataloader, desc="Iteration")
                for step, batch in enumerate(v_epoch_iterator):
                    with torch.no_grad():
                        q_encoder.eval()
                        p_encoder.eval()

                        cur_batch_size = batch[0].size()[0]  # 마지막 배치 때문에
                        if torch.cuda.is_available():
                            batch = tuple(t.cuda() for t in batch)
                        p_inputs = {
                            "input_ids": batch[0],
                            "attention_mask": batch[1],
                            "token_type_ids": batch[2],
                        }
                        q_inputs = {
                            "input_ids": batch[3],
                            "attention_mask": batch[4],
                            "token_type_ids": batch[5],
                        }
                        p_outputs = p_encoder(**p_inputs)
                        q_outputs = q_encoder(**q_inputs)

                        sim_scores = torch.matmul(
                            q_outputs, torch.transpose(p_outputs, 0, 1)
                        )
                        targets = torch.arange(0, cur_batch_size).long()
                        if torch.cuda.is_available():
                            targets = targets.to("cuda")

                        sim_scores = F.log_softmax(sim_scores, dim=1)
                        loss = F.nll_loss(sim_scores, targets)

                        _, preds = torch.max(sim_scores, 1)  #
                        valid_acc += (
                            torch.sum(preds.cpu() == targets.cpu()) / cur_batch_size
                        )

                        valid_loss += loss
                valid_loss = valid_loss / len(valid_dataloader)
                valid_acc = valid_acc / len(valid_dataloader)
                print()
                print(f"valid loss: {valid_loss}")
                print(f"valid acc: {valid_acc}")
                wandb.log({"valid loss": valid_loss, "valid acc": valid_acc})
                if best_loss > valid_loss:
                    print("best model save")
                    p_encoder.save_pretrained(args.output_dir + "/p_encoder")
                    q_encoder.save_pretrained(args.output_dir + "/q_encoder")
                    best_loss = valid_loss

        train_loss = train_loss / len(train_dataloader)
        train_acc = train_acc / len(train_dataloader)

        print(f"train loss: {train_loss}")
        print(f"train acc: {train_acc}")

        # valid_loss가 작아질 때만 저장
        # 두 모델을 합쳐서 trainer에 넘겨줄 수 있게 만들면 좀더 간단해질듯
    wandb.finish()

    return p_encoder, q_encoder


def train_with_negative(
    args, p_encoder, q_encoder, train_dataset, valid_dataset, num_neg
):
    # Dataloader
    wandb.login()
    wandb.init(
        project="retrieval_aug",
        entity="chungye-mountain-sherpa",
        name="retrieval_add_negative",
        group="klue-bert",
    )

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.per_device_train_batch_size,
        drop_last=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=args.per_device_eval_batch_size
    )

    # Optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in p_encoder.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in p_encoder.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
        {
            "params": [
                p
                for n, p in q_encoder.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in q_encoder.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )

    t_total = (
        len(train_dataloader)
        // args.gradient_accumulation_steps
        * args.num_train_epochs
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Start training!
    global_step = 0
    if torch.cuda.is_available():
        p_encoder.cuda()
        q_encoder.cuda()

    p_encoder.zero_grad()
    q_encoder.zero_grad()
    torch.cuda.empty_cache()

    best_loss = 9999  # valid_loss
    best_acc = -1
    num_epoch = 0

    for _ in range(int(args.num_train_epochs)):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        num_epoch += 1

        train_loss = 0
        train_acc = 0
        train_step = 0
        for batch in epoch_iterator:
            train_step += 1
            q_encoder.train()
            p_encoder.train()

            # if torch.cuda.is_available():
            #     batch = tuple(t.cuda() for t in batch)

            neg_batch_ids = []
            neg_batch_att = []
            neg_batch_tti = []
            random_sampling_idx = random.randrange(0, num_neg)
            for batch_in_sample_idx in range(args.per_device_train_batch_size):
                neg_batch_ids.append(
                    batch[3][:][batch_in_sample_idx][random_sampling_idx].unsqueeze(0)
                )
                neg_batch_att.append(
                    batch[4][:][batch_in_sample_idx][random_sampling_idx].unsqueeze(0)
                )
                neg_batch_tti.append(
                    batch[5][:][batch_in_sample_idx][random_sampling_idx].unsqueeze(0)
                )
            neg_batch_ids = torch.cat(neg_batch_ids)
            neg_batch_att = torch.cat(neg_batch_att)
            neg_batch_tti = torch.cat(neg_batch_tti)
            p_inputs = {
                "input_ids": torch.cat((batch[0], neg_batch_ids), 0).cuda(),
                "attention_mask": torch.cat((batch[1], neg_batch_att), 0).cuda(),
                "token_type_ids": torch.cat((batch[2], neg_batch_tti), 0).cuda(),
            }

            q_inputs = {
                "input_ids": batch[6].cuda(),
                "attention_mask": batch[7].cuda(),
                "token_type_ids": batch[8].cuda(),
            }

            p_outputs = p_encoder(**p_inputs)  # (batch_size * 2, emb_dim)
            q_outputs = q_encoder(**q_inputs)  # (batch_size, emb_dim)

            # Calculate similarity score & loss
            sim_scores = torch.matmul(
                q_outputs, torch.transpose(p_outputs, 0, 1)
            )  # (batch_size, emb_dim) x (emb_dim, batch_size) = (batch_size, batch_size * 2)

            # target: position of positive samples = diagonal element
            targets = torch.arange(0, args.per_device_train_batch_size).long()
            if torch.cuda.is_available():
                targets = targets.to("cuda")

            # print(sim_scores)
            # print()
            # print(targets)
            sim_scores = F.log_softmax(sim_scores, dim=1)
            loss = F.nll_loss(sim_scores, targets)
            train_loss += loss.item()

            _, preds = torch.max(sim_scores, 1)  #
            train_acc += (
                torch.sum(preds.cpu() == targets.cpu())
                / args.per_device_train_batch_size
            )

            loss.backward()
            optimizer.step()
            scheduler.step()
            q_encoder.zero_grad()
            p_encoder.zero_grad()
            global_step += 1

            # validation
            if train_step % 40 == 0:
                valid_loss = 0
                valid_acc = 0
                v_epoch_iterator = tqdm(valid_dataloader, desc="Iteration")
                for step, batch in enumerate(v_epoch_iterator):
                    with torch.no_grad():
                        q_encoder.eval()
                        p_encoder.eval()

                        cur_batch_size = batch[0].size()[0]  # 마지막 배치 때문에
                        if torch.cuda.is_available():
                            batch = tuple(t.cuda() for t in batch)
                        p_inputs = {
                            "input_ids": batch[0],
                            "attention_mask": batch[1],
                            "token_type_ids": batch[2],
                        }

                        q_inputs = {
                            "input_ids": batch[3],
                            "attention_mask": batch[4],
                            "token_type_ids": batch[5],
                        }
                        p_outputs = p_encoder(**p_inputs)
                        q_outputs = q_encoder(**q_inputs)

                        sim_scores = torch.matmul(
                            q_outputs, torch.transpose(p_outputs, 0, 1)
                        )
                        targets = torch.arange(0, cur_batch_size).long()
                        if torch.cuda.is_available():
                            targets = targets.to("cuda")

                        sim_scores = F.log_softmax(sim_scores, dim=1)
                        loss = F.nll_loss(sim_scores, targets)

                        _, preds = torch.max(sim_scores, 1)  #
                        valid_acc += (
                            torch.sum(preds.cpu() == targets.cpu()) / cur_batch_size
                        )

                        valid_loss += loss
                valid_loss = valid_loss / len(valid_dataloader)
                valid_acc = valid_acc / len(valid_dataloader)
                print()
                print(f"valid loss: {valid_loss}")
                print(f"valid acc: {valid_acc}")
                wandb.log({"valid loss": valid_loss, "valid acc": valid_acc})
                if best_loss > valid_loss:
                    print("best model save")
                    p_encoder.save_pretrained(args.output_dir + "/p_encoder")
                    q_encoder.save_pretrained(args.output_dir + "/q_encoder")
                    best_acc = valid_acc
                    best_loss = valid_loss

        train_loss = train_loss / len(train_dataloader)
        train_acc = train_acc / len(train_dataloader)

        print(f"train loss: {train_loss}")
        print(f"train acc: {train_acc}")

        # valid_loss가 작아질 때만 저장
        # 두 모델을 합쳐서 trainer에 넘겨줄 수 있게 만들면 좀더 간단해질듯
    wandb.finish()
    return p_encoder, q_encoder


def train_wight_grad_acum(args, p_encoder, q_encoder, train_dataset, valid_dataset):
    wandb.login()
    wandb.init(
        project="retrieval_aug",
        entity="chungye-mountain-sherpa",
        name="retrieval_train",
        group="klue-bert",
    )

    # Dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.per_device_train_batch_size,
        drop_last=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=args.per_device_eval_batch_size
    )

    # Optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in p_encoder.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in p_encoder.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
        {
            "params": [
                p
                for n, p in q_encoder.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in q_encoder.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )

    t_total = (
        len(train_dataloader)
        // args.gradient_accumulation_steps
        * args.num_train_epochs
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Start training!
    global_step = 0
    if torch.cuda.is_available():
        p_encoder.cuda()
        q_encoder.cuda()

    p_encoder.zero_grad()
    q_encoder.zero_grad()
    torch.cuda.empty_cache()

    best_loss = 9999  # valid_loss
    best_acc = 0
    num_epoch = 0

    for _ in range(int(args.num_train_epochs)):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        num_epoch += 1

        train_loss = 0
        train_acc = 0
        train_step = 0

        grad_acum = args.gradient_accumulation_steps

        p_output_total = []
        q_output_total = []

        for batch in epoch_iterator:
            train_step += 1
            q_encoder.train()
            p_encoder.train()

            if torch.cuda.is_available():
                batch = tuple(t.cuda() for t in batch)

            p_inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            q_inputs = {
                "input_ids": batch[3],
                "attention_mask": batch[4],
                "token_type_ids": batch[5],
            }

            p_outputs = p_encoder(**p_inputs)  # (batch_size, emb_dim)
            q_outputs = q_encoder(**q_inputs)  # (batch_size, emb_dim)
            if grad_acum != 1:
                p_output_total.append(p_outputs)
                q_output_total.append(q_outputs)
                grad_acum -= 1
                torch.cuda.empty_cache()
                continue
            else:
                grad_acum = args.gradient_accumulation_steps
                p_output_total.append(p_outputs)
                q_output_total.append(q_outputs)
                torch.cuda.empty_cache()
                p_output = torch.cat(p_output_total).to("cuda")
                q_output = torch.cat(q_output_total).to("cuda")
                p_output_total = []
                q_output_total = []
                # cur_batch_size = p_output.size()[0]
                # print(p_output.size())
                # print(q_output.size())
                # Calculate similarity score & loss
                sim_scores = torch.matmul(
                    q_output, torch.transpose(p_output, 0, 1)
                )  # (batch_size, emb_dim) x (emb_dim, batch_size) = (batch_size, batch_size)
                targets = torch.arange(
                    0,
                    args.per_device_train_batch_size * args.gradient_accumulation_steps,
                ).long()

                if torch.cuda.is_available():
                    targets = targets.to("cuda")

                sim_scores = F.log_softmax(sim_scores, dim=1)
                loss = F.nll_loss(sim_scores, targets)
                train_loss += loss.item()

                _, preds = torch.max(sim_scores, 1)  #
                train_acc += (
                    torch.sum(preds.cpu() == targets.cpu())
                    / args.per_device_train_batch_size
                )

                loss.backward()
                optimizer.step()
                scheduler.step()
                q_encoder.zero_grad()
                p_encoder.zero_grad()
                global_step += 1

            # validation
            if train_step % 100 == 0:
                valid_loss = 0
                valid_acc = 0
                v_epoch_iterator = tqdm(valid_dataloader, desc="Iteration")
                for step, batch in enumerate(v_epoch_iterator):
                    with torch.no_grad():
                        q_encoder.eval()
                        p_encoder.eval()

                        cur_batch_size = batch[0].size()[0]  # 마지막 배치 때문에
                        if torch.cuda.is_available():
                            batch = tuple(t.cuda() for t in batch)
                        p_inputs = {
                            "input_ids": batch[0],
                            "attention_mask": batch[1],
                            "token_type_ids": batch[2],
                        }
                        q_inputs = {
                            "input_ids": batch[3],
                            "attention_mask": batch[4],
                            "token_type_ids": batch[5],
                        }
                        p_outputs = p_encoder(**p_inputs)
                        q_outputs = q_encoder(**q_inputs)

                        sim_scores = torch.matmul(
                            q_outputs, torch.transpose(p_outputs, 0, 1)
                        )
                        targets = torch.arange(0, cur_batch_size).long()
                        if torch.cuda.is_available():
                            targets = targets.to("cuda")

                        sim_scores = F.log_softmax(sim_scores, dim=1)
                        loss = F.nll_loss(sim_scores, targets)

                        _, preds = torch.max(sim_scores, 1)  #
                        valid_acc += (
                            torch.sum(preds.cpu() == targets.cpu()) / cur_batch_size
                        )

                        valid_loss += loss
                valid_loss = valid_loss / len(valid_dataloader)
                valid_acc = valid_acc / len(valid_dataloader)
                print()
                print(f"valid loss: {valid_loss}")
                print(f"valid acc: {valid_acc}")
                wandb.log({"valid loss": valid_loss, "valid acc": valid_acc})
                if best_loss > valid_loss:
                    print("best model save")
                    p_encoder.save_pretrained(args.output_dir + "/p_encoder")
                    q_encoder.save_pretrained(args.output_dir + "/q_encoder")
                    # best_loss = valid_loss
                    best_acc = valid_acc

        train_loss = train_loss / len(train_dataloader)
        train_acc = train_acc / len(train_dataloader)

        print(f"train loss: {train_loss}")
        print(f"train acc: {train_acc}")

        # valid_loss가 작아질 때만 저장
        # 두 모델을 합쳐서 trainer에 넘겨줄 수 있게 만들면 좀더 간단해질듯
    wandb.finish()

    return p_encoder, q_encoder


def main(args):
    print(args)
    seed_everything(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    train_dataset = InBatchNegativeRandomDataset(
        data_path=args.train_data_path,
        bm25_path=args.train_bm25_path,
        max_context_seq_length=args.max_context_seq_length,
        max_question_seq_length=args.max_question_seq_length,
        neg_num=args.num_neg,
        tokenizer=tokenizer,
    )
    valid_dataset = get_tensor_for_dense(
        data_path=args.valid_data_path,
        # bm25_path=args.valid_bm25_path,
        max_context_seq_length=args.max_context_seq_length,
        max_question_seq_length=args.max_question_seq_length,
        tokenizer=tokenizer,
    )

    p_encoder = BertEncoder.from_pretrained(args.model_checkpoint)
    q_encoder = BertEncoder.from_pretrained(args.model_checkpoint)
    # p_encoder = ElectraEncoder.from_pretrained(args.model_checkpoint)
    # q_encoder = ElectraEncoder.from_pretrained(args.model_checkpoint)
    # p_encoder = RobertaEncoder.from_pretrained(args.model_checkpoint)
    # q_encoder = RobertaEncoder.from_pretrained(args.model_checkpoint)
    """
    https://github.com/bcaitech1/p3-mrc-team-ikyo/blob/main/code/retrieval_model.py를 참고하면 별개의 Encoder를 하나로 통합 할 수 있을 듯
    변경 한다면 train 함수의 save_pretreained를 save_state_dict로 수정하여야함
    """

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy=args.evaluation_strategy,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    p_encoder, q_encoder = train_with_negative(
        training_args, p_encoder, q_encoder, train_dataset, valid_dataset, args.num_neg
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="klue/bert-base",
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        default="/opt/ml/data/new_train_dataset/train",
    )
    parser.add_argument(
        "--valid_data_path",
        type=str,
        default="/opt/ml/data/new_train_dataset/validation",
    )
    parser.add_argument(
        "--train_bm25_path",
        type=str,
        default="/opt/ml/data/elastic_train_200.bin",
    )
    parser.add_argument(
        "--valid_bm25_path",
        type=str,
        default="/opt/ml/data/elastic_valid_100.bin",
    )
    parser.add_argument("--max_context_seq_length", type=int, default=512)
    parser.add_argument("--max_question_seq_length", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="retrieval")
    parser.add_argument("--evaluation_strategy", type=str, default="epoch")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_neg", type=int, default=5)

    args = parser.parse_args()
    # "kykim/bert-kor-base"
    # monologg/koelectra-base-v3-finetuned-korquad
    # /opt/ml/data/train_dataset/train
    # /opt/ml/data/train_dataset/retrieval_train_add_wiki_qa 위키 추가
    # "monologg/kobigbird-bert-base"
    main(args=args)

"""
    parser.add_argument("--max_context_seq_length", type=int, default=512)
    parser.add_argument("--max_question_seq_length", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="retrieval")
    parser.add_argument("--evaluation_strategy", type=str, default="epoch")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_neg", type=int, default=50)
"""
