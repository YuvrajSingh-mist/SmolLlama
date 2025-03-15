
import argparse
import inference
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch

import wandb

from inference import remove_prefix
import torch.optim as optim


import os
from config import ModelArgs
from model import Llama

from inference import greedy_decode
from data import prepare_dataset
from tokenizer import Tokenizer


torch.set_float32_matmul_precision('high')

scaler = torch.amp.GradScaler(enabled=(ModelArgs.dtype == 'float16'))


save_checkpoint_iter = ModelArgs.save_checkpoint_iter
total_iters = ModelArgs.total_iters
eval_iters = ModelArgs.eval_iters
eval_check = ModelArgs.eval_check
warmup_iters = ModelArgs.warmup_iters
min_lr = ModelArgs.min_lr
lr_decay_iters = ModelArgs.lr_decay_iters
total_batch_size = ModelArgs.total_batch_size
micro_batch_size = ModelArgs.micro_batch_size
gradient_accumulation_steps = ModelArgs.gradient_accumulation_steps


class Trainer:
    
    def __init__(self, model_args):


        def setup(rank=None, world_size=None):
            # os.environ['MASTER_ADDR'] = 'localhost'
            # os.environ['MASTER_PORT'] = '12355'
            init_process_group("nccl")
            # torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
            
        self.model_args = model_args  
        self.tokenizer = Tokenizer().ready_tokenizer()
        setup()
        
    def cleanup(self):
        destroy_process_group()

    def _save_snapshot(self, model, optimizer, epoch, step, save_dir):
        snapshot = {}
        snapshot["MODEL_STATE"] = model.module.state_dict()
        snapshot["OPTIMIZER_STATE"]= optimizer.state_dict()
        snapshot["EPOCHS_RUN"] = epoch
        snapshot["STEP_RUN"] = step
        torch.save(snapshot, os.path.join(save_dir, "snapshot.pt"))
        print(f"Epoch: {epoch} | step {step} | Training snapshot saved at snapshot.pt")

    # Warmup phase for 2000 steps
    def warmup_fn(step):
        if step < 2000:
            return step / 2000  # LR gradually increases
        return 1.0


    # learning rate decay scheduler (cosine with warmup) from https://github.com/karpathy/nanoGPT/blob/master/train.py

    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return ModelArgs.max_lr * (it + 1) / (warmup_iters + 1)
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
        return min_lr + coeff * (ModelArgs.max_lr - min_lr)


    def train_dpo():
        class DPO:
            def __init__(self, ref_model, sft_model, device, beta, tokenizer):
            
            
                self.ref_model = ref_model
                self.sft_model = sft_model
                self.device=device
                self.beta = beta
                self.tokenizer = tokenizer
                self.ref_model.eval()
            
            
            
            def DPOloss(self, datapoint, sft_model, ref_model):
            
            
            
                self.win_prompt = datapoint['chosen']
                self.lose_prompt = datapoint['rejected']
            
                with torch.no_grad():
                    self.win_log_ref = torch.nn.functional.log_softmax(ref_model(self.win_prompt['input_ids']), dim=-1)
                    self.win_log_ref = torch.gather(self.win_log_ref, -1, self.win_prompt['input_ids'].unsqueeze(-1)).squeeze(-1) #Why gather? Because its not token level stuff we care about but sequence level. Hence, we will sum up the probs of every token to get seq level but we don't want to do it for attention maksed tokens too. Hence we we will use gather() to get the ids and multiply the probs by the masked out tokens indexes.
                    # print("Gather: ", self.chosen_log_probs)
                    self.win_log_ref = self.win_log_ref * (self.win_prompt['attention_mask'])
                    self.win_log_ref = self.win_log_ref.sum(dim=-1) #Mean may dilute the encodings 
                    
                    self.lose_log_ref = torch.nn.functional.log_softmax(ref_model(self.lose_prompt['input_ids']), dim=-1)
                    self.lose_log_ref = torch.gather(self.lose_log_ref, -1, self.lose_prompt['input_ids'].unsqueeze(-1)).squeeze(-1) #Why gather? Because its not token level stuff we care about but sequence level. Hence, we will sum up the probs of every token to get seq level but we don't want to do it for attention maksed tokens too. Hence we we will use gather() to get the ids and multiply the probs by the masked out tokens indexes.
                    # print("Gather: ", self.chosen_log_probs)
                    self.lose_log_ref = self.lose_log_ref * (self.lose_prompt['attention_mask'])
                    self.lose_log_ref = self.lose_log_ref.sum(dim=-1) #Mean may dilute the encodings 
                
                    
                    self.win_log_sft = torch.nn.functional.log_softmax(sft_model(self.win_prompt['input_ids']), dim=-1)
                    self.win_log_sft = torch.gather(self.win_log_sft, -1, self.win_prompt['input_ids'].unsqueeze(-1)).squeeze(-1) #Why gather? Because its not token level stuff we care about but sequence level. Hence, we will sum up the probs of every token to get seq level but we don't want to do it for attention maksed tokens too. Hence we we will use gather() to get the ids and multiply the probs by the masked out tokens indexes.
                    # print("Gather: ", self.chosen_log_probs)
                    self.win_log_sft = self.win_log_sft * (self.win_prompt['attention_mask'])
                    self.win_log_sft = self.win_log_sft.sum(dim=-1) #Mean may dilute the encodings 
                    
                    self.lose_log_sft = torch.nn.functional.log_softmax(sft_model(self.lose_prompt['input_ids']), dim=-1)
                    self.lose_log_sft = torch.gather(self.lose_log_sft, -1, self.lose_prompt['input_ids'].unsqueeze(-1)).squeeze(-1) #Why gather? Because its not token level stuff we care about but sequence level. Hence, we will sum up the probs of every token to get seq level but we don't want to do it for attention maksed tokens too. Hence we we will use gather() to get the ids and multiply the probs by the masked out tokens indexes.
                    # print("Gather: ", self.chosen_log_probs)
                    self.lose_log_sft = self.lose_log_sft * (self.lose_prompt['attention_mask'])
                    self.lose_log_sft = self.lose_log_sft.sum(dim=-1) #Mean may dilute the encodings

                    self.diff1 = self.win_log_sft - self.win_log_ref
                    self.diff2 = self.lose_log_sft - self.lose_log_ref
                
                    self.final = -nn.functional.logsigmoid(self.beta *(self.diff1 - self.diff2)).mean() #Remember we have to maximize the rewards thus minimizing the negative sign! Also, since the var of rewards could be very much, we take mean so as to have a notion of normalizing it!
                
                    # sft_model.train()
                    return self.final
            
        setup()
        device = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(int(device))
        # torch.set_default_device('cuda')
        # train_dataloader = prepare_dataset(ModelArgs.batch_size)
        # rank = torch.distributed.get_rank()
        print(f"Start running DDP on rank {device}.")
        # # create model and move it to GPU with id rank
        # device_id = rank % torch.cuda.device_count()
        # CFG = ModelArgs()
        
        if(device == 0):

        
        
    #         # Initialise run
            wandb.init(
                # entity = 'rajceo2031',
                            project = 'Llama-DDP-DPO-10k-tokens',
                            # config = CFG,
                            # save_code = True,
                            #group = 'ANN',
                            #job_type = 'train'
    )
        print("wand initialized")
        ref_model = Llama(embeddings_dims=ModelArgs.embeddings_dims, block_size=ModelArgs.block_size, vocab_size=ModelArgs.vocab_size, dropout=ModelArgs.dropout, device=device)
        sft_model = Llama(embeddings_dims=ModelArgs.embeddings_dims, block_size=ModelArgs.block_size, vocab_size=ModelArgs.vocab_size, dropout=ModelArgs.dropout, device=device)
    
        dict_model = torch.load(ModelArgs.fine_tune_model_dir)
        dict_model['MODEL_STATE'] = remove_prefix(dict_model['MODEL_STATE'], '_orig_mod.')
        ref_model.load_state_dict(dict_model['MODEL_STATE'])

        dict_model = torch.load(ModelArgs.fine_tuned_model_dir)
        dict_model['MODEL_STATE'] = remove_prefix(dict_model['MODEL_STATE'], '_orig_mod.')
        sft_model.load_state_dict(dict_model['MODEL_STATE'])
        
        # print(f"Model on device {device} is ready")
        print(f"Model on device {device} is ready")
        for params in sft_model.parameters():
            params.requires_grad = True

        for params in ref_model.parameters():
            params.requires_grad = False
    
    
        sft_model.train()
        optimizer = optim.AdamW(sft_model.parameters(), lr=ModelArgs.max_lr, betas=(ModelArgs.beta_1, ModelArgs.beta_2), weight_decay=ModelArgs.weight_decay_optim, eps=ModelArgs.eps)
        
        # model = torch.compile(model)
        ref_model = ref_model.to(device)
        
        # ref_model = DDP(ref_model, device_ids=[device])

        sft_model = sft_model.to(device)
        
        sft_model = DDP(sft_model, device_ids=[device])

        
        dpo_loss = DPO(ref_model, sft_model, device, ModelArgs.beta, tokenizer)
    
        

        world_size = torch.cuda.device_count()
        @torch.inference_mode()
        def estimate_loss(val_loader, val_iterator, device):
            out = {}
            # train_loader = prepare_dataset('train', ModelArgs.batch_size)
            sft_model.eval()
            ref_model.eval()
            # val_loader_iterator = iter(val_loader)
            loader = None
            epoch_loss = None
            epoch_losses = []
            # print("Starting the eval...")
            for split in ['val']:
                print(f"Starting with {split} evaluation...")
                # losses = torch.zeros(ModelArgs.val_epochs)
                # if(split == 'train'):
                #         loader = train_loader
                # if(split == 'val'):
                #         loader = val_loader
                for step in range(eval_check):  
                    try:
                        batch = next(val_iterator)
                    except StopIteration:
                        val_loader_iterator = iter(val_loader)
                        batch = next(val_loader_iterator)
                    
                    total_loss = 0  
                    # loader.sampler.set_epoch(step)
                    total_batches = 0 
                    # batch = next(val_loader_iterator)
                    # for batch in loader:  # Loop through DataLoader batches
                    datapoint = batch
                    # idx = batch['input_ids']
                    # targets = batch['labels']
                    # idx = idx.to(device)
                    # targets = targets.to(device)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        
                        # logits = model(idx)
                        # batch_size, block_size, embeddings_dims = logits.shape
                        # logits = logits.view(batch_size * block_size, embeddings_dims)  # Flatten tokens
                        # targets = targets.view(batch_size * block_size)

                        # loss = F.cross_entropy(logits, targets, ignore_index=tokenizer.pad_token_id)
        
                        loss = dpo_loss.DPOloss(datapoint, sft_model, ref_model)
                        total_loss += loss.item()
                        total_batches += 1

                # Compute mean loss for this epoch
                epoch_loss = total_loss / total_batches if total_batches > 0 else 0.0
                epoch_losses.append(epoch_loss)

                    # print(f"Epoch {epoch + 1}/{ModelArgs.val_epochs}: Loss = {epoch_loss:.4f}")

                # Compute mean loss across all evaluation epochs
                out[split] = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
                epoch_loss = None
                epoch_losses = []

            sft_model.train()
            return out

        # model = model.to(rank)
        sft_model.train()
        count = 0
    
        train_dataloader = prepare_dataset('train', device, ModelArgs.batch_size)
        val_loader= prepare_dataset('val', device, ModelArgs.batch_size)
        # for step in tqdm(range(total_iters)):
        # for epoch in range(ModelArgs.epochs):
            # torch.cuda.synchronize() 
        
        # train_dataloader.sampler.set_epoch(epoch)
        
        # val_loader.sampler.set_epoch(epoch)
        print("Loaders ready both")
        epochs = ModelArgs.epochs

        # train_step_iterator = range(len(train_dataloader))
        # if device == 0:  # Only create progress bar on rank 0
        #   train_step_iterator = tqdm(train_step_iterator, desc="Training Progress", position=0, leave=True)

            # Print progress on rank 0
        train_loader_length = 0
        train_data_iterator = iter(train_dataloader)
        val_data_iterator = iter(val_loader)
        token_count = 0
        if(device == 0):
            train_loader_length = len(train_dataloader)
            # print("Total batches: ", train_loader_length)
        # print("Length of : ", len(train_dataloader))
        # print("Length of val: ", len(val_loader))
        # for  step, batch in enumerate(train_dataloader):
        for step in tqdm(range(total_iters)):
            # print("Dataloader things: ", batch)
            # print("Total batches: ", len(train_dataloader))
            
            
            if(device == 0):
                # if(step % 100 == 0):
            #     if(step == train_loader_length):
            #       break
                    print("Step : ", step, "/", total_iters)
                    print('Total batches: ', len(train_dataloader))
                    print("Total gradient accumulation steps: ", gradient_accumulation_steps)
                    print("Total tokens processed: ", token_count)
                    
            # all_gpus_avg_train_loss = None
            # all_gpus_avg_val_loss = None
            # every once in a while evaluate the loss on train and val sets
            if (step  % eval_iters == 0 and step != 0) or step == total_iters - 1:
                losses = estimate_loss( val_loader, val_data_iterator, 'cuda')
                # avg_train_loss = losses['train']
                avg_val_loss = losses['val']
                # print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                # if device == 0:  # Only print on main process
                print(f"[GPU {device}] | Step: {step} / {total_iters} | Val Loss: {losses['val']:.4f}")
            
                avg_val_loss = torch.Tensor([losses['val']]).to(device)
                # torch.distributed.reduce(avg_train_loss, dst=0, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.reduce(avg_val_loss, dst=0, op=torch.distributed.ReduceOp.SUM)
                
                if device == 0:
                    # all_gpus_avg_train_loss = avg_train_loss / world_size
                    # print(f"All_GPUs_Train_losses: {all_gpus_avg_train_loss.item():.4f}")
                    all_gpus_avg_val_loss = avg_val_loss / world_size
                    print(f"All_GPUs_Val_losses: {all_gpus_avg_val_loss.item():.4f}")
                    
            
                    
                    wandb.log({
                        # "Learning Rate": optimizer.param_groups[0]['lr'],
                        # "All_GPUs_Train_losses": all_gpus_avg_train_loss,
                        "All_GPUs_Val_losses": all_gpus_avg_val_loss,
                        # "training_step_loss": losses['train'],
                        "val_step_loss": losses['val'],
                        # "Step": step,
                        # "Epoch": epoch
                    })
                
                
            
        
            if step % save_chechpoint_iter == 0 and device == 0 and step != 0:
                print(f"Saving the model checkpoint for step: {step}")
                _save_snapshot(sft_model, optimizer, None, None, step)
            
            accumulated_loss = 0.0
            
            
            optimizer.zero_grad(set_to_none=True)
            for micro_step in range(gradient_accumulation_steps):
                try:
                    batch = next(train_data_iterator)
                except StopIteration:
                    train_data_iterator = iter(train_dataloader)
                    batch = next(train_data_iterator)
            
                with torch.autocast(device_type=ModelArgs.device, dtype=torch.bfloat16):
        
                    loss = dpo_loss.DPOloss(batch, sft_model, ref_model)
            
                    loss = loss / gradient_accumulation_steps #IDK why div is done here specifically? Maybe think of it in terms of a very big batch being processed and there is need for equal important of each mini batch for the overall big batch
                    accumulated_loss += loss.detach()
                
                sft_model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1) # so that we dont synchronize the gradient everytime across the GPU devices
                scaler.scale(loss).backward()
                    # Check for unused parameters
                # unused_params = find_unused_parameters(model)
                # if unused_params:
                #     print(f"Unused parameters: {unused_params}")
            # break
        
                if(device == 0):
                    if(micro_step % 10 == 0):
                #     if(step == train_loader_length):
                #       break
                        
                        print("Micro Batch : ", micro_step)
                        print("Step : ", step, "/", total_iters)
                        print('Total batches: ', len(train_dataloader))
                        print("Total gradient accumulation steps: ", gradient_accumulation_steps)
                        # print("Total tokens processed: ", token_count)
                # count += 1
        
            lr = get_lr(step)
            for params in optimizer.param_groups:
                params['lr'] = lr
                
            
            
            # Compute gradient norms before clipping
            if(ModelArgs.clip != 0.0):
                scaler.unscale_(optimizer) #To avoid underflow
                total_norm_before = torch.norm(
                    torch.stack([torch.norm(p.grad.detach(), 2) for p in sft_model.parameters()]), 2
                )
        
                torch.nn.utils.clip_grad_norm_(sft_model.parameters(), max_norm=ModelArgs.clip)
        
                # Compute gradient norms after clipping
                total_norm_after = torch.norm(
                    torch.stack([torch.norm(p.grad.detach(), 2) for p in sft_model.parameters()]), 2
                )
                
                if(device  == 0 and step !=0):
                    print(f"Gradient Norm Before Clipping: {total_norm_before.item():.4f}")
                    print(f"Gradient Norm After Clipping: {total_norm_after.item():.4f}")
        
            scaler.step(optimizer)
            scaler.update()
        
            # optimizer.step()
            # new_scheduler.step()
            
            torch.cuda.synchronize() 
            torch.distributed.reduce(loss, dst=0, op=torch.distributed.ReduceOp.SUM)
            if(device == 0):
                wandb.log({
                        "Learning Rate": lr,
                        "All_GPUs_Train_losses": accumulated_loss.item(),
                        # "All_GPUs_Val_losses": all_gpus_avg_val_loss,
                        # "training_step_loss": losses['train'],
                        # "val_step_loss": losses['val'],
                        "Step": step,
                        # "Epoch": epoch
                        
                    })
        
            if device == 0 and step % 50 == 0:
                count = 1
                while(count):  # Only generate text on the main process
                    # print("Generating text...")
                    
                    alpaca_prompt = '''
        
                        ### Instruction:
                        {}
        
                        ### Input: 
                        {}
                        
                        ### Response:
                    
                        '''
                    
                    prompt = alpaca_prompt.format("Follow the given instrcutions carefully. ", "You are a helpful assistant. Tell me about the beauty of the night sky",  "")
        #             print("Generating text")
                    # prompt = "Once upon a time"
                    generated_text = topk_sampling(sft_model, prompt, max_length=100, top_k=50, temperature=1.0, device=device)
        
        
                    print(f" Step: {step} | Generated Text: {generated_text}")
            
                    count -= 1
            
        
        # Cleanup
        if device == 0:
            # writer.close()
            wandb.finish()
        cleanup()






    def train():

        setup()
        device = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(int(device))

        print(f"Start running DDP on rank {device}.")

        if(device == 0):

        
        
    #         # Initialise run
            wandb.init(
                # entity = 'rajceo2031',
                            project = 'Llama-DDP-Pretrain-10-billion-tokens',
                            # config = CFG,
                            # save_code = True,
                            #group = 'ANN',
                            #job_type = 'train'
    )
        print("wand initialized")
        
        model = Llama(embeddings_dims=ModelArgs.embeddings_dims, block_size=ModelArgs.block_size, vocab_size=ModelArgs.vocab_size, dropout=ModelArgs.dropout, device=device)
        
        # print(f"Model on device {device} is ready")
        print(f"Model on device {device} is ready")
        
     
        optimizer = optim.AdamW(model.parameters(), lr=ModelArgs.max_lr, betas=(ModelArgs.beta_1, ModelArgs.beta_2), weight_decay=ModelArgs.weight_decay_optim, eps=ModelArgs.eps)
        
        # model = torch.compile(model)
        model = model.to(device)
        
        model = DDP(model, device_ids=[device])
        

        if(ModelArgs.fine_tuned_model_dir is not None):
            dict_model = torch.load(ModelArgs.fine_tuned_model_dir)
            dict_model['MODEL_STATE'] = remove_prefix(dict_model['MODEL_STATE'], '_orig_mod.')
            model.load_state_dict(dict_model['MODEL_STATE'])
            print("Model loaded from fine-tuned model directory.")
        
        
        model.eval()
        world_size = torch.cuda.device_count()
        @torch.inference_mode()
        def estimate_loss(val_loader, val_iterator, device):
            out = {}
  
            loader = None
            epoch_loss = None
            epoch_losses = []
         
            for split in ['val']:
                print(f"Starting with {split} evaluation...")
              
                for step in range(eval_check):  
                    try:
                        batch = next(val_iterator)
                    except StopIteration:
                        val_loader_iterator = iter(val_loader)
                        batch = next(val_loader_iterator)
                    
                    total_loss = 0  
                    
                    total_batches = 0 
          
                    idx = batch['input_ids']
                    targets = batch['labels']
                    idx = idx.to(device)
                    targets = targets.to(device)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        
                        logits = model(idx)
                        batch_size, block_size, embeddings_dims = logits.shape
                        logits = logits.view(batch_size * block_size, embeddings_dims) 
                        targets = targets.view(batch_size * block_size)

                        loss = F.cross_entropy(logits, targets, ignore_index=tokenizer.pad_token_id)

                        total_loss += loss.item()
                        total_batches += 1

             
                epoch_loss = total_loss / total_batches if total_batches > 0 else 0.0
                epoch_losses.append(epoch_loss)

                  
                out[split] = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
                epoch_loss = None
                epoch_losses = []

            model.train()
            return out

      
        model.train()
        count = 0
    
        train_dataloader = prepare_dataset('train', device, ModelArgs.batch_size)
        val_loader= prepare_dataset('val', device, ModelArgs.batch_size)
      
        print("Loaders ready both")
        epochs = ModelArgs.epochs

        train_loader_length = 0
        train_data_iterator = iter(train_dataloader)
        val_data_iterator = iter(val_loader)
        token_count = 0
        if(device == 0):
            train_loader_length = len(train_dataloader)
      
        for step in tqdm(range(total_iters)):
           
            
            if(device == 0):

                    print("Step : ", step, "/", total_iters)
                    print('Total batches: ', len(train_dataloader))
                    print("Total gradient accumulation steps: ", gradient_accumulation_steps)
                    print("Total tokens processed: ", token_count)
                    
          
            if (step  % eval_iters == 0 and step != 0) or step == total_iters - 1:
                losses = estimate_loss( val_loader, val_data_iterator, 'cuda')
                # avg_train_loss = losses['train']
                avg_val_loss = losses['val']
      
                print(f"[GPU {device}] | Step: {step} / {total_iters} | Val Loss: {losses['val']:.4f}")
              
                avg_val_loss = torch.Tensor([losses['val']]).to(device)
                # torch.distributed.reduce(avg_train_loss, dst=0, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.reduce(avg_val_loss, dst=0, op=torch.distributed.ReduceOp.SUM)
                
                if device == 0:
                  
                    all_gpus_avg_val_loss = avg_val_loss / world_size
                    print(f"All_GPUs_Val_losses: {all_gpus_avg_val_loss.item():.4f}")
          
                    wandb.log({
                        # "Learning Rate": optimizer.param_groups[0]['lr'],
                        # "All_GPUs_Train_losses": all_gpus_avg_train_loss,
                        "All_GPUs_Val_losses": all_gpus_avg_val_loss,
                        # "training_step_loss": losses['train'],
                        "val_step_loss": losses['val'],
                        # "Step": step,
                        # "Epoch": epoch
                    })
                
                
            
         

            if step % save_chechpoint_iter == 0 and device == 0 and step != 0:
                print(f"Saving the model checkpoint for step: {step}")
                _save_snapshot(model, optimizer, None, None, step)
            
            accumulated_loss = 0.0
            
            
            optimizer.zero_grad(set_to_none=True)
            for micro_step in range(gradient_accumulation_steps):
                try:
                    batch = next(train_data_iterator)
                except StopIteration:
                    train_data_iterator = iter(train_dataloader)
                    batch = next(train_data_iterator)
                # print(batch)
                # batch = next(train_data_iterator)
                # print(batch)
                # batch = {k: v.to(self.local_rank) for k, v in batch.items()}
                idx = batch['input_ids'].to(device)
                # idx, targets = get_batch(split='train')
                # print(f"Starting the train step: {step}...")
                # for idx, targets in train_loader:
                # idx, targets = next(iter(train_loader))
                
                # print("Idx: ", idx)
                # print("Targets: ", targets)
                
                # idx = idx.to(device)
                # print("Idx: ", idx)
                # print("Targets: ", targets)
                targets = batch['labels'].to(device)
                token_count += len(idx)
                with torch.autocast(device_type=ModelArgs.device, dtype=torch.bfloat16):
                    logits = model(idx)
                    batch_size, block_size, embeddings_dims = logits.shape
                    # print(logits.shape)
                    # print(targets)
                    logits = logits.view(batch_size*block_size, embeddings_dims)
                    # print("OK")
                    targets = targets.view(batch_size * block_size)
                    # print("OK2")
                    loss = nn.functional.cross_entropy(logits, targets, ignore_index=tokenizer.pad_token_id)
                    
                    loss = loss / gradient_accumulation_steps #IDK why div is done here specifically? Maybe think of it in terms of a very big batch being processed and there is need for equal important of each mini batch for the overall big batch
                    accumulated_loss += loss.detach()
                
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1) # so that we dont synchronize the gradient everytime across the GPU devices
                scaler.scale(loss).backward()
                    # Check for unused parameters
                unused_params = find_unused_parameters(model)
                if unused_params:
                    print(f"Unused parameters: {unused_params}")
            # break
        
                if(device == 0):
                    if(micro_step % 10 == 0):
                #     if(step == train_loader_length):
                #       break
                        
                        print("Micro Batch : ", micro_step)
                        print("Step : ", step, "/", total_iters)
                        print('Total batches: ', len(train_dataloader))
                        print("Total gradient accumulation steps: ", gradient_accumulation_steps)
                        print("Total tokens processed: ", token_count)
                # count += 1
        
            lr = get_lr(step)
            for params in optimizer.param_groups:
                params['lr'] = lr
                
            
            
            # Compute gradient norms before clipping
            if(ModelArgs.clip != 0.0):
                scaler.unscale_(optimizer) #To avoid underflow
                total_norm_before = torch.norm(
                    torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters()]), 2
                )

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=ModelArgs.clip)

                # Compute gradient norms after clipping
                total_norm_after = torch.norm(
                    torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters()]), 2
                )
                
                if(device  == 0 and step !=0):
                    print(f"Gradient Norm Before Clipping: {total_norm_before.item():.4f}")
                    print(f"Gradient Norm After Clipping: {total_norm_after.item():.4f}")

            scaler.step(optimizer)
            scaler.update()
        
            # optimizer.step()
            # new_scheduler.step()
            
            torch.cuda.synchronize() 
            torch.distributed.reduce(loss, dst=0, op=torch.distributed.ReduceOp.SUM)
            if(device == 0):
                wandb.log({
                        "Learning Rate": lr,
                        "All_GPUs_Train_losses": accumulated_loss.item(),
                        # "All_GPUs_Val_losses": all_gpus_avg_val_loss,
                        # "training_step_loss": losses['train'],
                        # "val_step_loss": losses['val'],
                        "Step": step,
                        # "Epoch": epoch
                        
                    })
            # print(loss.item())
       
            # break
            if device == 0 and step % 5 == 0:
                count = 3
                while(count):  # Only generate text on the main process
                  
                    prompt = "Once upon a time"
                    generated_text = topk_sampling(model, prompt, max_length=50, top_k=50, temperature=1.0, device=device)
        
       
                    print(f" Step: {step} | Generated Text: {generated_text}")
              
                    count -= 1
            
     
        if device == 0:
          
            wandb.finish()
        cleanup()


    world_size = torch.cuda.device_count()
    print(f"World size: {world_size}")
        
    


def parse_args():
    parser = argparse.ArgumentParser(description="Model Training Arguments")
    
    # Add arguments for each field in ModelArgs
    parser.add_argument("--epochs", type=int, default=ModelArgs.epochs, help="Number of training epochs.")
    parser.add_argument(
    "--train", 
    type=str, 
    choices=["P", "D"], 
    required=True, 
    help="Choose training mode: 'P' for pretraining/sft, 'D' for preference alignment."
)
    parser.add_argument("--block_size", type=int, default=ModelArgs.block_size, help="Block size for the model.")
    parser.add_argument("--batch_size", type=int, default=ModelArgs.batch_size, help="Batch size for training.")
    parser.add_argument("--fine_tuned_model_dir", type=str, default='/mnt/c/Users/yuvra/OneDrive/Desktop/Work/pytorch/ChatLlama/snapshot_fine_tuned_model_900.pt', help="Directory of fine-tuned model.")
    parser.add_argument("--embeddings_dims", type=int, default=ModelArgs.embeddings_dims, help="Embedding dimensions.")
    parser.add_argument("--attn_dropout", type=float, default=ModelArgs.attn_dropout, help="Attention dropout rate.")
    parser.add_argument("--no_of_heads", type=int, default=ModelArgs.no_of_heads, help="Number of attention heads.")
    parser.add_argument("--dropout", type=float, default=ModelArgs.dropout, help="Dropout rate.")
    parser.add_argument("--val_epochs", type=int, default=ModelArgs.val_epochs, help="Number of validation epochs.")
    parser.add_argument("--max_lr", type=float, default=ModelArgs.max_lr, help="Learning rate.")
    parser.add_argument("--no_of_decoder_layers", type=int, default=ModelArgs.no_of_decoder_layers, help="Number of decoder layers.")
    parser.add_argument("--weight_decay_optim", type=float, default=ModelArgs.weight_decay_optim, help="Weight decay for optimizer.")
    parser.add_argument("--beta_1", type=float, default=ModelArgs.beta_1, help="Beta1 for Adam optimizer.")
    parser.add_argument("--beta_2", type=float, default=ModelArgs.beta_2, help="Beta2 for Adam optimizer.")
    parser.add_argument("--clip", type=float, default=ModelArgs.clip, help="Gradient clipping value.")
    parser.add_argument("--device", type=str, default=ModelArgs.device, help="Device to run the model on (e.g., 'cuda' or 'cpu').")
    parser.add_argument("--no_kv_heads", type=int, default=ModelArgs.no_kv_heads, help="Number of key/value heads.")
    parser.add_argument("--vocab_size", type=int, default=ModelArgs.vocab_size, help="Vocabulary size.")
    parser.add_argument("--eps", type=float, default=ModelArgs.eps, help="Epsilon value for numerical stability.")
    parser.add_argument("--dtype", type=str, default=ModelArgs.dtype, help="Data type for tensors (e.g., 'float16' or 'bfloat16').")
    parser.add_argument("--save_checkpoint_dir", type=str, default=ModelArgs.save_checkpoint_dir, help="Directory to save model checkpoints.")
    parser.add_argument("--prompt", type=str, default=ModelArgs.prompt, help="Prompt for testing during training.")
    parser.add_argument("--pre_trained_model_dir", type=str, default=ModelArgs.pre_trained_model_dir, help="Directory of pre-trained model.")
    # Additional arguments
    parser.add_argument("--save_checkpoint_iter", type=int, default='/mnt/c/Users/yuvra/OneDrive/Desktop/Work/pytorch/ChatLlama/snapshot_6750.pt', help="Save checkpoint every N iterations.")
    parser.add_argument("--total_iters", type=int, default=ModelArgs.total_iters, help="Total number of training iterations.")
    parser.add_argument("--eval_iters", type=int, default=ModelArgs.eval_iters, help="Number of iterations for evaluation.")
    parser.add_argument("--eval_check", type=int, default=ModelArgs.eval_check, help="Evaluate model every N iterations.")
    parser.add_argument("--warmup_iters", type=int, default=ModelArgs.warmup_iters, help="Number of warmup iterations for learning rate scheduling.")
    parser.add_argument("--min_lr", type=float, default=ModelArgs.min_lr, help="Minimum learning rate.")
    parser.add_argument("--lr_decay_iters", type=int, default=ModelArgs.lr_decay_iters, help="Number of iterations for learning rate decay.")
    parser.add_argument("--total_batch_size", type=int, default=ModelArgs.total_batch_size, help="Total batch size across all devices.")
    parser.add_argument("--micro_batch_size", type=int, default=ModelArgs.micro_batch_size, help="Micro batch size per device.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=ModelArgs.gradient_accumulation_steps, help="Number of gradient accumulation steps.")
    
    args = parser.parse_args()
    return args


def initialize_model_args(args):
    # Create a ModelArgs instance from the parsed arguments
    model_args = ModelArgs(
        fine_tuned_model_dir=args.fine_tuned_model_dir,
        epochs=args.epochs,
        block_size=args.block_size,
        batch_size=args.batch_size,
        # inference=args.inference,
        embeddings_dims=args.embeddings_dims,
        attn_dropout=args.attn_dropout,
        no_of_heads=args.no_of_heads,
        dropout=args.dropout,
        val_epochs=args.val_epochs,
        max_lr=args.max_lr,
        no_of_decoder_layers=args.no_of_decoder_layers,
        weight_decay_optim=args.weight_decay_optim,
        beta_1=args.beta_1,
        beta_2=args.beta_2,
        clip=args.clip,
        device=args.device,
        no_kv_heads=args.no_kv_heads,
        vocab_size=args.vocab_size,
        eps=args.eps,
        dtype=args.dtype,
        save_checkpoint_dir=args.save_checkpoint_dir,
        prompt=args.prompt,
        save_checkpoint_iter=args.save_checkpoint_iter,
        total_iters=args.total_iters,
        eval_iters=args.eval_iters,
        eval_check=args.eval_check,
        warmup_iters=args.warmup_iters,
        min_lr=args.min_lr,
        lr_decay_iters=args.lr_decay_iters,
        total_batch_size=args.total_batch_size,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    return model_args


if __name__ == "__main__":
    args = parse_args()
    

    model_args = initialize_model_args(args)
    
    trainer = Trainer()
    if(args.train == "P"):
        trainer.train()
    elif(args.train == "D"):
        trainer.train_dpo()