import torch
from torch.utils.data import (DataLoader, RandomSampler)
from torch.nn import MSELoss

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer

from tqdm import trange, tqdm

import pickle

from scipy.stats import pearsonr

def train(args, train_dataset, validation_data, validation_labels, sen_encoder, model_checkpoint):

    # logging
    best_cor = 0
    stop_counter = 0 

    # Dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_bs)

    # Optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
          {'params': [p for n, p in sen_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
          {'params': [p for n, p in sen_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
          ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # Start training!
    global_step = 0
    
    sen_encoder.zero_grad()
    torch.cuda.empty_cache()
    
    train_iterator = trange(int(args.num_epochs), desc="Epoch")

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")

        # to compute average loss in an epoch
        train_loss_list = []

        print(f"**********Train: epoch {epoch}**********")
        for step, batch in enumerate(epoch_iterator):
            
            sen_encoder.train()
            
            
            if torch.cuda.is_available():
                batch = tuple(t.cuda() for t in batch)

            inputs = {'input_ids_1': batch[0],
                        'input_ids_2': batch[1],
                        'attention_mask_1': batch[2],
                        'attention_mask_2': batch[3]
                       }

            print(inputs['input_ids_1'])
            break
            targets = batch[4]
            cos_sim_outputs = (sen_encoder(**inputs) * (5/2)) + 2.5

            # loss
            criterion = MSELoss()
            train_loss = criterion(cos_sim_outputs, targets)
            train_loss_list.append(train_loss.detach().cpu().numpy())

            # print loss every 1000 steps
            if step % 500 == 0 and step > 99:
                epoch_average_loss = sum(train_loss_list[step-100:step]) / 99
                print(f'step: {step} with loss: {epoch_average_loss}')

            train_loss = train_loss / args.gradient_accumulation_steps
            train_loss.backward()

            if ((step + 1) % args.gradient_accumulation_steps == 0) or (step + 1 == len(epoch_iterator)):
                optimizer.step()
                scheduler.step()
                sen_encoder.zero_grad()

            global_step += 1
            
            torch.cuda.empty_cache()
        
        print("**********EVALUATION**********")
        with torch.no_grad():
            sen_encoder.eval()
            sen_encoder.eval()

            print(val_dataset[0])
            ## validation_data
            val_u_seqs = tokenizer(validation_data['sentence1'], padding="max_length", truncation=True, return_tensors='pt').to('cuda')
            val_v_seqs = tokenizer(validation_data['sentence2'], padding="max_length", truncation=True, return_tensors='pt').to('cuda')

            val_outputs = val_dataset[4]

            val_cos_sim = (sen_encoder(**val_inputs)*5/2-2.5).to('cpu')  #(num_query, emb_dim)
                
        pearson_cor = pearsonr(val_outputs, val_cos_sim)

        if pearson_cor > best_cor:
            stop_counter = 0
            best_cor = pearson_cor
        
            torch.save(sen_encoder, f'checkpoints/val_acc{pearson_cor:4.2%}_p_encoder.pt')
                
        else:
            stop_counter += 1
            print(f"early stop count {stop_counter} out of {args.early_stop}")
            if args.early_stop == stop_counter:
                break

        print("epoch pearson correaltion:", pearsonr)
        print("best cor from all epochs", best_cor)
