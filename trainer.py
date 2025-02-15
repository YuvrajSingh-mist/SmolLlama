
import argparse
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
from data import prepare_dataset
from tokenizer import Tokenizer

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

    def _save_snapshot(self, model, optimizer, epoch, step):
        snapshot = {}
        snapshot["MODEL_STATE"] = model.module.state_dict()
        snapshot["OPTIMIZER_STATE"]= optimizer.state_dict()
        snapshot["EPOCHS_RUN"] = epoch
        snapshot["STEP_RUN"] = step
        torch.save(snapshot, "snapshot.pt")
        print(f"Epoch: {epoch} | step {step} | Training snapshot saved at snapshot.pt")

    def train(self):
        # setup()
        device = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(int(device))

        # train_dataloader = prepare_dataset(self.model_args.batch_size)
        # rank = torch.distributed.get_rank()
        print(f"Start running DDP on rank {device}.")
        # # create model and move it to GPU with id rank
        # device_id = rank % torch.cuda.device_count()
        # CFG = self.model_args()

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

        model = Llama(embeddings_dims=self.model_args.embeddings_dims, block_size=self.model_args.block_size, vocab_size=self.model_args.vocab_size, dropout=self.model_args.dropout, device=device)
        # Optimizer setup and scheduler steup

        model = model.to(device)
        print(f"Model on device {device} is ready")
        # Wrap model with DDP after moving to GPU
        model = DDP(model, device_ids=[device])
        optimizer = optim.AdamW(model.parameters(), lr=self.model_args.max_lr, betas=(self.model_args.beta_1, self.model_args.beta_2), weight_decay=self.model_args.weight_decay_optim)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=2000, T_mult=1, eta_min=0.001)
        print(f"Model on device {device} is ready")


        save_chechpoint_iter = 1000
        total_iters = 20000
        eval_iters = 1000
        eval_check = 100

        world_size = torch.cuda.device_count()
        @torch.inference_mode()
        def estimate_loss(val_loader, train_loader=None):
            out = {}
            # train_loader = prepare_dataset('train', self.model_args.batch_size)
            model.eval()
            loader = None
            epoch_loss = None
            epoch_losses = []
            # print("Starting the eval...")
            for split in ['train', 'val']:
                print(f"Starting with {split} evaluation...")
                # losses = torch.zeros(self.model_args.val_epochs)
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

                    # print(f"Epoch {epoch + 1}/{self.model_args.val_epochs}: Loss = {epoch_loss:.4f}")

                # Compute mean loss across all evaluation epochs
                out[split] = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
                epoch_loss = None
                epoch_losses = []

            model.train()
            return out

        # model = model.to(rank)
        model.train()

        # for step in tqdm(range(total_iters)):
        for epoch in range(self.model_args.epochs):
            # torch.cuda.synchronize() 
            train_dataloader = prepare_dataset('train', self.model_args.batch_size)
            train_dataloader.sampler.set_epoch(epoch)
            val_loader= prepare_dataset('val', self.model_args.batch_size)
            val_loader.sampler.set_epoch(epoch)
            print("Loaders ready both")
            epochs = self.model_args.epochs

            train_loader_length = 0
            if(device == 0):
                train_loader_length = len(train_dataloader)
                print("Total batches: ", train_loader_length)

            for  step, batch in enumerate(train_dataloader):

                if(device == 0):
                    if(step % 100 == 0):
                        # if(step == train_loader_length):
                        #     break
                        print("Batch : ", step, "/", len(train_dataloader))

                if (step  % eval_iters == 0 and step != 0) or step == total_iters - 1:
                    losses = estimate_loss( val_loader, train_dataloader)
                    avg_train_loss = losses['train']
                    avg_val_loss = losses['val']

                    print(f"[GPU {device}] | Epoch {epoch}/{self.model_args.epochs}| |Step: {step} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")

                    avg_train_loss = torch.Tensor([losses['train']]).to(device)
                    avg_val_loss = torch.Tensor([losses['val']]).to(device)
                    torch.distributed.reduce(avg_train_loss, dst=0, op=torch.distributed.ReduceOp.SUM)
                    torch.distributed.reduce(avg_val_loss, dst=0, op=torch.distributed.ReduceOp.SUM)
                    
                    if device == 0:
                        all_gpus_avg_train_loss = avg_train_loss / world_size
                        print(f"All_GPUs_Train_losses: {all_gpus_avg_train_loss.item():.4f}")
                        all_gpus_avg_val_loss = avg_val_loss / world_size
                        print(f"All_GPUs_Val_losses: {all_gpus_avg_val_loss.item():.4f}")
                        

                        
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
                    self._save_snapshot(epoch=epoch, model=model, optimizer=optimizer, step=step)

                
                # batch = {k: v.to(self.local_rank) for k, v in batch.items()}
                idx = batch['input_ids'].to(device)

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

                if device == 0 and step % 1000 == 0 and step != 0:

                    print("Generating text...")
                    prompt = "Once upon a time"
                    generated_text = greedy_decode(model, self.tokenizer, prompt, max_length=50)
                    # generated_text = beam_search(model, tokenizer, prompt, beam_width=5, max_length=50, temperature=1.0)
                    print(f" Step: {step} | Generated Text: {generated_text}")
                    # save_to_file(generated_text)
             
                
            
            # break
        # Cleanup
        if device == 0:
            # writer.close()
            wandb.finish()
            
        self.cleanup()


    world_size = torch.cuda.device_count()
    print(f"World size: {world_size}")
    
    

def main():
    
    
     # Parse command-line arguments
    parser = argparse.ArgumentParser()
    # Add arguments for self.model_args fields
    parser.add_argument("--epochs", type=int, default=ModelArgs.epochs, help="Number of training epochs.")
    parser.add_argument("--block_size", type=int, default=ModelArgs.block_size, help="Block size for the model.")
    parser.add_argument("--batch_size", type=int, default=ModelArgs.batch_size, help="Batch size for training.")
    parser.add_argument("--embeddings_dims", type=int, default=ModelArgs.embeddings_dims, help="Embedding dimensions.")
    parser.add_argument("--attn_dropout", type=float, default=ModelArgs.attn_dropout, help="Attention dropout rate.")
    parser.add_argument("--no_of_heads", type=int, default=ModelArgs.no_of_heads, help="Number of attention heads.")
    parser.add_argument("--dropout", type=float, default=ModelArgs.dropout, help="Dropout rate.")
    parser.add_argument("--val_epochs", type=int, default=ModelArgs.val_epochs, help="Number of validation epochs.")
    parser.add_argument("--max_lr", type=float, default=ModelArgs.max_lr, help="Maximum learning rate.")
    parser.add_argument("--no_of_decoder_layers", type=int, default=ModelArgs.no_of_decoder_layers, help="Number of decoder layers.")
    parser.add_argument("--weight_decay_optim", type=float, default=ModelArgs.weight_decay_optim, help="Weight decay for optimizer.")
    parser.add_argument("--beta_1", type=float, default=ModelArgs.beta_1, help="Beta1 for Adam optimizer.")
    parser.add_argument("--beta_2", type=float, default=ModelArgs.beta_2, help="Beta2 for Adam optimizer.")
    parser.add_argument("--no_kv_heads", type=int, default=ModelArgs.no_kv_heads, help="Number of key/value heads.")
    
    args = parser.parse_args()
    
    
    model_args = ModelArgs(
        epochs=args.epochs,
        block_size=args.block_size,
        batch_size=args.batch_size,
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
        vocab_size=args.vocab_size
    )

    trainer = Trainer(model_args)
    
    trainer.train()
    
if __name__ == '__main__':
    main()