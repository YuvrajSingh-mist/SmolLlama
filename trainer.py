
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers.models.prophetnet.modeling_prophetnet import ProphetNetDecoderModelOutput
import wandb


import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


import os

from config import ModelArgs
from model import Llama

from inference import greedy_decode



def setup(rank=None, world_size=None):
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    init_process_group("nccl")
    # torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    
def cleanup():
    destroy_process_group()



def train():
    setup()
    device = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(int(device))

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
                        project = 'Llama-DDP-Pretrain-10-billion-tokens',
                        # config = CFG,
                        # save_code = True,
                        #group = 'ANN',
                        #job_type = 'train'
)

    model = Llama(embeddings_dims=ModelArgs.embeddings_dims, block_size=ModelArgs.block_size, vocab_size=ModelArgs.vocab_size, dropout=ModelArgs.dropout, device=device)
    # Optimizer setup and scheduler steup

    model = model.to(device)
    print(f"Model on device {device} is ready")
    # Wrap model with DDP after moving to GPU
    model = DDP(model, device_ids=[device])
    optimizer = optim.AdamW(model.parameters(), lr=ModelArgs.max_lr, betas=(ModelArgs.beta_1, ModelArgs.beta_2), weight_decay=ModelArgs.weight_decay_optim)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=2000, T_mult=1, eta_min=0.001)
    print(f"Model on device {device} is ready")
    # optimizer = torch.optim.AdamW(params=model.parameters(), lr=ModelArgs.max_lr)
    # Create DataLoader with collate_fn
    # train_loader = DataLoader(train_dataset,  batch_size=ModelArgs.batch_size, shuffle=False, sampler=DistributedSampler(train_dataset, shuffle=True, num_replicas=int(os.environ["WORLD_SIZE"]), rank=device))
    # val_loader = DataLoader(val_dataset,   batch_size=ModelArgs.batch_size, shuffle=False, sampler=DistributedSampler(train_dataset, shuffle=True, num_replicas=int(os.environ["WORLD_SIZE"]), rank=device))
    # print("Loader is ready")
        # print(train_loader)
    # print(next(iter(train_loader)))

    save_chechpoint_iter = 1000
    total_iters = 20000
    eval_iters = 1000
    eval_check = 100
    # for X,y in train_loader:
    #     print(X.shape)
    #     print(y.shape)

     # Only create progress bar for rank 0
    # eval_epoch_iterator = range(eval_iters)
    # train_epoch_iterator = range(total_iters)
    # if device == 0:
    #     train_epoch_iterator = tqdm(train_epoch_iterator, desc="Training")

    # train_epoch_iterator = range(ModelArgs.epochs)
    # if device == 0:  # Ensure tqdm only runs on rank 0
    #     train_epoch_iterator = tqdm(train_epoch_iterator, desc="Training Progress", position=0, leave=True)

    # lr_scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max= total_steps - initial_iters)
    world_size = torch.cuda.device_count()
    @torch.inference_mode()
    def estimate_loss(val_loader, train_loader=None):
        out = {}
        # train_loader = prepare_dataset('train', ModelArgs.batch_size)
        model.eval()
        loader = None
        epoch_loss = None
        epoch_losses = []
        # print("Starting the eval...")
        for split in ['train', 'val']:
            print(f"Starting with {split} evaluation...")
            # losses = torch.zeros(ModelArgs.val_epochs)
            if(split == 'train'):
                    loader = train_loader
            if(split == 'val'):
                    loader = val_loader
            for step in range(eval_check):  
                total_loss = 0  
                # loader.sampler.set_epoch(step)
                total_batches = 0  
                batch = next(iter(loader))
                # for batch in loader:  # Loop through DataLoader batches
                idx = batch['input_ids']
                targets = batch['labels']
                idx = idx.to(device)
                targets = targets.to(device)

                logits = model(idx)
                batch_size, block_size, embeddings_dims = logits.shape
                logits = logits.view(batch_size * block_size, embeddings_dims)  # Flatten tokens
                targets = targets.view(batch_size * block_size)

                loss = F.cross_entropy(logits, targets)

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

        model.train()
        return out

    # model = model.to(rank)
    model.train()

    # for step in tqdm(range(total_iters)):
    for epoch in range(ModelArgs.epochs):
        # torch.cuda.synchronize() 
        train_dataloader = prepare_dataset('train', ModelArgs.batch_size)
        train_dataloader.sampler.set_epoch(epoch)
        val_loader= prepare_dataset('val', ModelArgs.batch_size)
        val_loader.sampler.set_epoch(epoch)
        print("Loaders ready both")
        epochs = ModelArgs.epochs

        # train_step_iterator = range(len(train_dataloader))
        # if device == 0:  # Only create progress bar on rank 0
        #   train_step_iterator = tqdm(train_step_iterator, desc="Training Progress", position=0, leave=True)

         # Print progress on rank 0
        train_loader_length = 0
        if(device == 0):
            train_loader_length = len(train_dataloader)
            print("Total batches: ", train_loader_length)
        # print("Length of : ", len(train_dataloader))
        # print("Length of val: ", len(val_loader))
        for  step, batch in enumerate(train_dataloader):
            # print("Dataloader things: ", batch)
            # print("Total batches: ", len(train_dataloader))
            if(device == 0):
              if(step % 100 == 0):
                if(step == train_loader_length):
                  break
                print("Batch : ", step, "/", len(train_dataloader))
            # all_gpus_avg_train_loss = None
            # all_gpus_avg_val_loss = None
            # every once in a while evaluate the loss on train and val sets
            if (step  % eval_iters == 0 and step != 0) or step == total_iters - 1:
                losses = estimate_loss( val_loader, train_dataloader)
                avg_train_loss = losses['train']
                avg_val_loss = losses['val']
                # print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                # if device == 0:  # Only print on main process
                print(f"[GPU {device}] | Epoch {epoch}/{ModelArgs.epochs}| |Step: {step} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")
                # print(f"[GPU {device}] | Epoch {epoch}/{ModelArgs.epochs}| |Step: {step} | Train Loss: {losses['train']:.4f}")
                    # print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                    # Log training loss more frequently
                 # Aggregate average loss across all GPUs
                avg_train_loss = torch.Tensor([losses['train']]).to(device)
                avg_val_loss = torch.Tensor([losses['val']]).to(device)
                torch.distributed.reduce(avg_train_loss, dst=0, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.reduce(avg_val_loss, dst=0, op=torch.distributed.ReduceOp.SUM)
                
                if device == 0:
                    all_gpus_avg_train_loss = avg_train_loss / world_size
                    print(f"All_GPUs_Train_losses: {all_gpus_avg_train_loss.item():.4f}")
                    all_gpus_avg_val_loss = avg_val_loss / world_size
                    print(f"All_GPUs_Val_losses: {all_gpus_avg_val_loss.item():.4f}")
                    
                # if device == 0:
         
                    # writer.add_scalar("All_GPUs_Train_losses", all_gpus_avg_train_loss.item(), global_step=step)
                    # writer.add_scalar("All_GPUs_Val_losses", all_gpus_avg_val_loss.item(), global_step=step)
                    # writer.add_scalar("training_step_loss", losses['train'], global_step=step)
                    # writer.add_scalar("val_step_loss", losses['val'], global_step=step)
                    # writer.add_scalar("GPU", device, global_step=step)
                    # writer.add_scalar("Epoch", epoch, global_step=step)
                    
                    wandb.log({
                        "GPU": device,
                        "All_GPUs_Train_losses": all_gpus_avg_train_loss,
                        "All_GPUs_Val_losses": all_gpus_avg_val_loss,
                        "training_step_loss": losses['train'],
                        "val_step_loss": losses['val'],
                        "Step": step,
                        "Epoch": epoch
                    })
                
              
           
           #Loading a checkpoint
            # if(os.path.exists('snapshot.pt')):
            #    model, optimizer =  _load_snapshot(model=model, optimizer=optimizer, epoch=epoch, step=step, snapshot_path='snapshot.pt')
            
            if(step % save_chechpoint_iter == 0 and device == 0 and step != 0):
                print(f"Saving the model checkpoint for step: {step}")
                _save_snapshot(epoch=epoch, model=model, optimizer=optimizer, step=step)

            
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
            logits = model(idx)
            batch_size, block_size, embeddings_dims = logits.shape
            logits = logits.view(batch_size*block_size, embeddings_dims)
            targets = targets.view(batch_size * block_size)
            loss = nn.functional.cross_entropy(logits, targets)
    
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            # torch.cuda.synchronize() 
            # print(loss.item())
            # if(step % 100 == 0):
            #     print(f'Step : {step} | GPU: {device} Loss: {loss.item()}')
            # if device == 0:
            #     print("loss: ", loss.item())
            # train_epoch_iterator.set_postfix({"loss": f"{loss.item():.4f}"})
            # print(loss.item())
            # break
    
            # if step != 0 and (step % eval_iters == 0 or step == total_steps -1) :
            #     loss_values = estimate_loss()
            #     print("Train Loss at {} steps : {}".format(step, loss.item()), "Val Loss at {} steps : {}".format(step, loss_values['val']))
    
            # Add after a training step:
            # unused_params = find_unused_parameters(model)
            # print("Unused parameters:", unused_params)
            # break
            if device == 0 and step % 1000 == 0 and step != 0:
            #   count = 5
              # while(count):  # Only generate text on the main process
              print("Generating text...")
              prompt = "Once upon a time"
              generated_text = greedy_decode(model, tokenizer, prompt, max_length=50)
              # generated_text = beam_search(model, tokenizer, prompt, beam_width=5, max_length=50, temperature=1.0)
              print(f" Step: {step} | Generated Text: {generated_text}")
              save_to_file(generated_text)
                    # count -= 1
            
            # if step != 0:
            #         train_step_iterator.set_postfix({"Train loss": f"{all_gpus_avg_train_loss.item():.4f} | Val Loss : {all_gpus_avg_val_loss.item():.4f}"})
            
        
        break
    # Cleanup
    if device == 0:
        # writer.close()
        wandb.finish()
    cleanup()


world_size = torch.cuda.device_count()
print(f"World size: {world_size}")