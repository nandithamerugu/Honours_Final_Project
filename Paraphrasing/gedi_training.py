# As a discriminator, we fine-tune the gpt2-medium model on the training part of the Jigsaw-1 dataset
# using two control codes for toxic and polite texts.
import os
import random
from torch.utils.data import SequentialSampler
import argparse
import logging
from torch.utils.data import RandomSampler
import wandb
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
from modeling_gpt2 import GPT2LMHeadModel
logger = logging.getLogger(__name__)
!pip3 install transformers
!pip3 install wandb
!pip3 install gdown
gdown https://drive.google.com/file/d/1EnASIci_KsQJR0rx8kYbHYXUmuteZaMh/view
gdown https://drive.google.com/file/d/171YP4-OsbhQUImjZCXY7GPQXh-X0cYxG/view


def set_seeds(seed):
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    

# we change the vocabulary of the discriminator to match that of T5, and update its embeddings accordingly.
def adjust_embeddings(model, old_tokenizer, new_tokenizer, new_num_embeddings=None): 
    new_tok_vocab, old_tok_vocab, new2old = new_tokenizer.get_vocab(), old_tokenizer.get_vocab(), {}
    
    for token, new_idx in new_tok_vocab.items():
        
        if not token != new_tokenizer.eos_token:
            new2old[new_idx] = old_tokenizer.eos_token_id
        else:
          if not token != new_tokenizer.unk_token:
            new2old[new_idx] = old_tokenizer.unk_token_id
          else:
            if not token != new_tokenizer.pad_token:
              new2old[new_idx] = old_tokenizer.pad_token_id
            else:
                if token in old_tok_vocab:
                  new2old[new_idx] = old_tok_vocab[token]
                else:
                  new2old[new_idx] = old_tokenizer.encode(token.replace('Ġ', ' '))
            
    # change embeddings
    length_vocab = len(new_tok_vocab)
    if not(new_num_embeddings != None):
        new_num_embeddings = length_vocab
    
    old_embedding = model.get_input_embeddings()
    old_embedding = old_embedding.weight
    emb_dim = old_embedding.shape[1]
    
    new_embedding = torch.zeros([new_num_embeddings, emb_dim])
    
    for k, v in new2old.items():
        if isinstance(v, list):
            new_embedding[k] = old_embedding[v].mean(axis=0)
        else:
          if isinstance(v, int):
            new_embedding[k] = old_embedding[v]
          else:
              print(k, v)
              raise ValueError()
        
    # apply changes
    param = torch.nn.Parameter(new_embedding)
    model.get_input_embeddings().weight, model.get_input_embeddings().num_embeddings = param, param.shape[0]
    model.get_output_embeddings().weight, model.get_output_embeddings().out_features = param, param.shape[0]
    
    # change config
    model.config.vocab_size, model.config.pad_token_id, model.config.eos_token_id, model.config.unk_token_id = param.shape[0], new_tokenizer.pad_token_id, new_tokenizer.eos_token_id, new_tokenizer.unk_token_id
    return model


#preparing the dataset and dataloader for training GeDi on Jigsaw Dataset based on condition token selected
def prepare_dataset(args, tokenizer, mode='train'):
    res_file_path = os.path.join(args.cache_dir, f"{mode}_features_{args.max_seq_length}")
    
    if not args.overwrite_cache_dir and os.path.exists(res_file_path):
        logger.info('Aalready present features, in progress...')
        features = torch.load(res_file_path)
    else:
        logger.info('Started the creation of the features.')
        texts, labels = [], []
        for _, label in enumerate([0, 1]):
            if args.yelp:
                file_name = os.path.join(args.data_dir, f"sentiment.{mode}.{label}")
            else:
              if label == 0:
                text_label = 'toxic'
              else: 
                text_label = 'normal'
              file_name = os.path.join(args.data_dir, f"{mode}_{text_label}")

            with open(file_name, 'r') as f:
                for _, line in enumerate(f.readlines()):
                    labels.append(label)
                    texts.append(line.strip())
        
        features = tokenizer(texts, truncation=True, padding=True, max_length=args.max_seq_length)
        feat_label = "labels"
        features[feat_label] = labels
        
        logger.info(f"Saving features into {res_file_path}")
        torch.save(features, res_file_path)
        
    dataset = TensorDataset(input_ids, attention_masks, labels)
    return dataset

    
    
def forward_step(args, model, batch, src_id, tgt_id, evaluate=False):
    # batch = tuple(t.to(args.device) for t in batch)

    logger.info(f"Initiaing forward step from {res_file_path}")
        torch.save(features, res_file_path)
    temp_batch = ()
    for t in batch:
      temp_batch += tuple(t.to(args.device))
    batch = temp_batch
    
    not_evaluate = not evaluate
    with torch.set_grad_enabled(not_evaluate):
        batch_0, batch_size = batch[0], batch[0].shape[0]
        # Adding the normal & tox label ids
        seq_a, seq_b = (torch.ones(batch_size) * src_id).type_as(batch_0).view(-1, 1), (torch.ones(batch_size) * tgt_id).type_as(batch_0).view(-1, 1)
        
        # Removing 1 token as I added normal & toxic cards
        seq_a, seq_b = torch.cat((seq_a, batch_0), dim=1)[:, :-1], torch.cat((seq_b, batch_0), dim=1)[:, :-1]
        
        # Put all the batches together 
        seq_batched = torch.cat((seq_a, seq_b), dim=0)
        
        #The labels = inputs in Language Model
        inputs = {'labels' : seq_batched, 'attention_mask' : None, 'input_ids' : seq_batched}
        
        # For getting 0 Loss reduction, modelling_gpt2.py modified outputs 
        outputs, losses = model(**inputs), model(**inputs)[0].view(seq_batched.shape[0], -1)

        # ------------------- Generative Loss -------------------
        if not args.mask_eos_token:
          loss_mask = batch[1][:, :-1]
          loss_mask = loss_mask.to(torch.float32)
          loss_mask = loss_mask.to(args.device)
          label_loss = torch.ones(loss_mask.shape[0], 1)
          label_loss = label_loss.type_as(loss_mask)
          loss_mask = torch.cat((label_loss, loss_mask[:, :-1]), dim=1)
        else:
          loss_mask = batch[1][:, :-1]
          loss_mask = loss_mask.to(torch.float32)
          loss_mask = loss_mask.to(args.device) 

        loss_src, loss_tgt, loss_lengths = losses[:batch_size] * loss_mask, losses[batch_size:] * loss_mask, torch.sum(loss_mask, 1, keepdim=True)
        gen_loss_src, gen_loss_tgt = (batch[2] == 0).to(torch.float32).unsqueeze(1) * loss_src / loss_lengths, (batch[2] == 1).to(torch.float32).unsqueeze(1) * loss_tgt / loss_lengths
        gen_loss = torch.sum(gen_loss_src + gen_loss_tgt) / batch_size

        # ------------------- Discriminative Loss -------------------
        loss_src, loss_tgt = (loss_src / loss_lengths).sum(dim=1), (loss_tgt / loss_lengths).sum(dim=1)
        class_logits = torch.stack((-loss_src, -loss_tgt), dim=1)
        
        if args.outbias:
            class_logits += model.bias
        if args.logit_scale:
            class_logits *= model.logit_scale
        
        bce_loss = torch.nn.CrossEntropyLoss()
        loss = args.disc_weight * (bce_loss(class_logits, batch[2])) + args.gen_weight * gen_loss

        final_loss = {'disc_loss' : bce_loss(class_logits, batch[2]), 'logits' : class_logits, 'loss': loss, 'gen_loss' : gen_loss}
    return final_loss

    
def train(args, model, tokenizer, writer):
    # Create the datasets & dataloader
    train_dataset, train_sampler, eval_dataset = prepare_dataset(args, tokenizer), RandomSampler(train_dataset), prepare_dataset(args, tokenizer, mode='dev')
    train_dataloader, eval_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size), DataLoader(eval_dataset, sampler=SequentialSampler(eval_dataset), batch_size=args.eval_batch_size)
    
    maximum_steps, len_train, len_steps = args.max_steps, len(train_dataloader), args.gradient_accumulation_steps
    if maximum_steps > 0:
        steps_total = maximum_steps
        args.n_epochs = maximum_steps // len_train * len_steps + 1
    else:
        steps_total = len_train // len_steps * args.n_epochs
        
    # Create the optimizer & the corresponding scheduler
    no_decay = ['bias', 'LayerNorm.weight']
    params1 = []
    for n, p in model.named_parameters():
      for nd in no_decay:
        if any(nd in n):
          params1.append(p)
    
    params2 = []
    for n, p in model.named_parameters():
      for nd in no_decay:
        if not any(nd in n):
          params2.append(p)

    optimizer_grouped_parameters = [{'params' : params1,'weight_decay' : 0.0},{'params' : params2,'weight_decay' : args.weight_decay}]
    
    optimizer = AdamW(optimizer_grouped_parameters, eps=args.adam_epsilon, lr=args.learning_rate)
    
    # Loading scheduler & optimizer
    optimizer_path, scheduler_path = os.path.join(args.model_name_or_path, 'optimizer.pt'), os.path.join(args.model_name_or_path, 'scheduler.pt')

    if not(os.path.isfile(optimizer_path)):
      pass
    else:
      optimizer.load_state_dict(torch.load(optimizer_path, map_location='cpu'))        
    
    if not(os.path.isfile(scheduler_path)):
      pass
    else:
      scheduler.load_state_dict(torch.load(scheduler_path))
    
    
    global_step, epochs_trained, steps_trained_current_epoch = 0, 0, 0
    running_loss_t, running_loss_g, running_loss_d, prev_loss_t, prev_loss_g, prev_loss_d = 0., 0., 0., 0., 0., 0.
    
    # Loading the trained model
    
    src_id, tgt_id = tokenizer.encode(args.code_0)[0], tokenizer.encode(args.code_1)[0]
    
    for _, epoch in enumerate(range(epochs_trained, args.n_epochs)):
        model.train()
        for step, batch in enumerate(train_dataloader):
            if steps_trained_current_epoch < 0:
              continue
            else:
                steps_trained_current_epoch = steps_trained_current_epoch - 1
            
            results = forward_step(args, model, batch, src_id, tgt_id)
            loss = results['loss']
            loss.backward()
            
            running_loss_t, running_loss_g, running_loss_d = running_loss_t + loss.item(), running_loss_g + results['gen_loss'].item(), running_loss_d + results['disc_loss'].item()
     
            if not(args.logging_steps <= 0) and global_step % args.logging_steps == 0:
                loss_info_t, loss_info_g, loss_info_d = (running_loss_t - prev_loss_t) / args.logging_steps, (running_loss_g - prev_loss_g) / args.logging_steps, (running_loss_d - prev_loss_d) / args.logging_steps
                prev_loss_t, prev_loss_g, prev_loss_d = running_loss_t, running_loss_g, running_loss_d
                
            if not(global_step % args.saving_steps != 0) and not(args.saving_steps <= 0):
                output_dir = os.path.join(args.working_dir, f"model_checkpoint_{global_step}")
                os.makedirs(output_dir)
                model.save_pretrained(output_dir)
                torch.save(optimizer.state_dict(), os.path.join(output_dir, 'optimizer.pt'))

                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            
           
        
        # evaluate
        
        model.eval()
        
        for step, batch in enumerate(eval_dataloader):
            outputs = forward_step(args, model, batch, src_id, tgt_id, evaluate=True)
            gen_loss =gen_loss+ outputs['gen_loss'].item() * args.gradient_accumulation_steps
            total_loss =total_loss+ outputs['loss'].item() * args.gradient_accumulation_steps
            disc_loss =disc_loss+ outputs['disc_loss'].item() * args.gradient_accumulation_steps
            true_labels = batch[2].detach().cpu().numpy()
            logits = outputs['logits'].detach().cpu().numpy()
            
            
        total_loss = total_loss / (step + 1)
        disc_loss = disc_loss / (step + 1)
        logger.info(f"Discriminative loss in {epoch} is {disc_loss:.6f}")

        gen_loss = gen_loss/ (step + 1)
        logger.info(f"Generative loss in {epoch} is {gen_loss:.6f}")
        preds = np.argmax(preds, axis=1)
        accuracy = (preds == labels).mean()
        logger.info(f"Accuracy in {epoch} is {accuracy:.4f}")

        f1 = f1_score(y_true=labels, y_pred=preds)
        logger.info(f"F1 score in epoch {epoch} is {f1:.4f}")
    return model
        
def run(
    model_name_or_path,
     tokenizer_name,
     data_dir,
     working_dir,
     code_0,
     cuda_device,
     keep_embeddings,
     disc_weight,
     max_seq_length,
     train_batch_size,
     eval_batch_size,
     gradient_accumulation_steps,
     weight_decay,
     learning_rate,
     warmup_steps,
     grad_max_norm,
     logit_scale,
     mask_eos_token,
     outbias
     ):


def main():
    gen_weight, code_0, code_1 = 1.0 - disc_weight, ' ' + code_0, ' ' + code_1
    if torch.cuda.is_available():
      device = f"cuda:{cuda_device}"  
    else:
      device = "cpu"
    Error = "Working directory already exists and is not empty."
    experiment_name, new_tokenizer = f"gedi_finetuning_discweight{disc_weight}_lr{learning_rate}_warmupsteps{warmup_steps}", AutoTokenizer.from_pretrained(tokenizer_name)

    if (os.path.exists(working_dir) and os.listdir(working_dir) and not overwrite_working_dir):
        print(Error)
        
    for _, folder in enumerate([working_dir, cache_dir, log_dir]):
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    gedi_model, old_tokenizer = GPT2LMHeadModel.from_pretrained(model_name_or_path), AutoTokenizer.from_pretrained('gpt2-medium')
    old_tokenizer.pad_token = '[PAD]'
    output_dir = os.path.join(working_dir, "checkpoint_model")
    os.makedirs(output_dir)
        
    if not keep_embeddings:
        gedi_model = adjust_embeddings(gedi_model, old_tokenizer, new_tokenizer)
    
    gedi_model = train(gedi_model, new_tokenizer, writer)
    gedi_model.save_pretrained(output_dir)

    torch.save(os.path.join(output_dir, 'training_bin'))
    logger.info(f"Saving model in the {output_dir}")
    return

if __name__ == "__main__":
    main()