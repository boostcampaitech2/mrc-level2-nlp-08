import torch
from torch.utils.data import (DataLoader, RandomSampler)
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn.functional as F

from tqdm import trange, tqdm



def train(args, train_dataset,  val_q_seqs, val_p_seqs, validation_document_id, p_model, q_model):
    
    # logging
    best_acc = 0
    stop_counter = 0 

    # Dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_bs)

    # Optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
          {'params': [p for n, p in p_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
          {'params': [p for n, p in p_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
          {'params': [p for n, p in q_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
          {'params': [p for n, p in q_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
          ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # Start training!
    global_step = 0
    
    p_model.zero_grad()
    q_model.zero_grad()
    torch.cuda.empty_cache()
    
    train_iterator = trange(int(args.num_epochs), desc="Epoch")

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")

        # to compute average loss in an epoch
        train_loss_list = []

        print(f"**********Train: epoch {epoch}**********")
        for step, batch in enumerate(epoch_iterator):
            q_model.train()
            p_model.train()
            
            targets = torch.zeros(args.train_bs).long()
            if torch.cuda.is_available():
                batch = tuple(t.cuda() for t in batch)
                targets = targets.cuda()

            p_inputs = {'input_ids': batch[0].view(
                                          args.train_bs*(args.num_neg+1), -1),
                        'attention_mask': batch[1].view(
                                          args.train_bs*(args.num_neg+1), -1),
                        'token_type_ids': batch[2].view(
                                          args.train_bs*(args.num_neg+1), -1)
                        }
            
            q_inputs = {'input_ids': batch[3],
                        'attention_mask': batch[4],
                        'token_type_ids': batch[5]}

            p_outputs = p_model(**p_inputs)  #(batch_size*(num_neg+1), emb_dim) -> 16, 756
            q_outputs = q_model(**q_inputs)  #(batch_size*, emb_dim) -> 4, 756

            # Calculate similarity score & loss
            # p_outputs = p_outputs.view(args.per_device_train_batch_size, -1, num_neg+1) -> 4, 756, 4
            p_outputs = torch.transpose(p_outputs.view(args.train_bs, args.num_neg+1,-1),1,2) # (batch_size, emb_dim, num_neg+1)
            q_outputs = q_outputs.view(args.train_bs, 1, -1) # (batch_size,1,emb_dim) -> 4, 1, 756

            sim_scores = torch.bmm(q_outputs, p_outputs).squeeze()  #(batch_size, num_neg+1)
            sim_scores = sim_scores.view(args.train_bs, -1)
            sim_scores = F.log_softmax(sim_scores, dim=1)

            train_loss = F.nll_loss(sim_scores, targets)
            train_loss_list.append(train_loss.detach().cpu().numpy())

            # print loss every 1000 steps
            if step % 100 == 0 and step > 99:
                epoch_average_loss = sum(train_loss_list[step-100:step]) / 99
                print(f'step: {step} with loss: {epoch_average_loss}')

            train_loss = train_loss / args.gradient_accumulation_steps
            train_loss.backward()

            if ((step + 1) % args.gradient_accumulation_steps == 0) or (step + 1 == len(epoch_iterator)):
                optimizer.step()
                scheduler.step()
                q_model.zero_grad()
                p_model.zero_grad()

            global_step += 1
            
            torch.cuda.empty_cache()
        
        print("**********EVALUATION**********")
        with torch.no_grad():
            p_model.eval()
            q_model.eval()
            
            q_emb = q_model(**val_q_seqs).to('cpu')  #(num_query, emb_dim)
                
            p_embs = []
            for i in tqdm(range(len(val_p_seqs))):
                p = val_p_seqs[i].to('cuda')
                p_emb = p_model(**p).to('cpu').numpy()
                p_embs.append(p_emb)

        p_embs = torch.Tensor(p_embs).squeeze()
        dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()


        counter = 0
        for i in range(len(rank)):
            if validation_document_id[i] == rank[i][0]:
                counter += 1
        
        acc  = counter / len(rank)

        if acc > best_acc:
            stop_counter = 0
            best_acc = acc
            torch.save(p_model, f'checkpoints/val_acc{acc:4.2%}_p_encoder.pt')
            torch.save(q_model, f'checkpoints/val_acc{acc:4.2%}_q_encoder.pt')
        else:
            stop_counter += 1
            print(f"early stop count {stop_counter} out of {args.early_stop}")
            if args.early_stop == stop_counter:
                break

        print("epoch acc:", acc)
        print("best acc from all epochs", best_acc)