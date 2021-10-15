import argparse
from .retrieval_model import BertEncoder, ElectraEncoder
from utils import seed_everything
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


def to_cuda(batch):
    return tuple(t.cuda() for t in batch)


def get_tensor_dataset(data_path: str, max_seq_length: int, tokenizer) -> TensorDataset:
    """
    question과 context를 tokenize 한 수 TensorDataset으로 concat하여 return합니다.
    """
    dataset = load_from_disk(data_path)
    q_seqs = tokenizer(
        dataset["question"],
        max_length=max_seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    p_seqs = tokenizer(
        dataset["context"],
        max_length=max_seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    tensor_dataset = TensorDataset(
        p_seqs["input_ids"],
        p_seqs["attention_mask"],
        p_seqs["token_type_ids"],
        q_seqs["input_ids"],
        q_seqs["attention_mask"],
        q_seqs["token_type_ids"],
    )

    return tensor_dataset


def train(args, p_encoder, q_encoder, train_dataset, valid_dataset):
    # Dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.per_device_train_batch_size,
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=args.per_device_train_batch_size, drop_last=False
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

        for batch in epoch_iterator:
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

            sim_scores = F.log_softmax(sim_scores, dim=1)
            loss = F.nll_loss(sim_scores, targets)

            loss.backward()
            optimizer.step()
            scheduler.step()
            q_encoder.zero_grad()
            p_encoder.zero_grad()
            global_step += 1

            torch.cuda.empty_cache()
        print(loss)

        # validation
        v_epoch_iterator = tqdm(valid_dataloader, desc="Iteration")
        valid_loss = 0
        for step, batch in enumerate(v_epoch_iterator):
            with torch.no_grad():
                q_encoder.eval()
                p_encoder.eval()

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

                sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))
                targets = torch.arange(0, args.per_device_train_batch_size).long()
                if torch.cuda.is_available():
                    targets = targets.to("cuda")

                sim_scores = F.log_softmax(sim_scores, dim=1)
                loss = F.nll_loss(sim_scores, targets)

                valid_loss += loss
        valid_loss = valid_loss / len(valid_dataloader)
        print(f"valid loss: {valid_loss}")

        if best_loss > valid_loss:
            p_encoder.save_pretrained(args.output_dir + "/p_encoder")
            q_encoder.save_pretrained(args.output_dir + "/q_encoder")
            best_loss = valid_loss

    return p_encoder, q_encoder


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    train_dataset = get_tensor_dataset(
        args.train_data_path, args.max_seq_length, tokenizer
    )
    valid_dataset = get_tensor_dataset(
        args.valid_data_path, args.max_seq_length, tokenizer
    )

    p_encoder = ElectraEncoder.from_pretrained(args.model_checkpoint)
    q_encoder = ElectraEncoder.from_pretrained(args.model_checkpoint)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy=args.evaluation_strategy,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
    )

    p_encoder, q_encoder = train(
        training_args,
        p_encoder,
        q_encoder,
        train_dataset,
        valid_dataset,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="monologg/koelectra-base-v3-discriminator",
    )
    parser.add_argument(
        "--train_data_path", type=str, default="../data/train_dataset/train/"
    )
    parser.add_argument(
        "--valid_data_path", type=str, default="../data/train_dataset/validation/"
    )
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="retrieval")
    parser.add_argument("--evaluation_strategy", type=str, default="epoch")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    args = parser.parse_args()

    main(args=args)
