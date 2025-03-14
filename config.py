import argparse
from dataclasses import dataclass
import torch

@dataclass
class ModelArgs:
    pre_trained_model_dir: str = None
    fine_tuned_model_dir: str = None
    epochs: int = 4
    beta: float = 0.1
    block_size: int = 256
    batch_size: int = 128
    inference = None
    embeddings_dims: int = 512
    attn_dropout: float = 0.1
    no_of_heads: int = 8
    dropout: float = 0.1
    val_epochs: int = 2
    max_lr: float = 6e-4
    no_of_decoder_layers: int = 16  
    weight_decay_optim: float = 0.1
    beta_1: float = 0.9
    beta_2: float = 0.95
    clip: float = 1.0
    device: str = 'cuda'
    no_kv_heads: int = 2
    vocab_size: int = 50304 
    eps: float = 1e-5
    dtype: str = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    save_checkpoint_dir: str = "checkpoints"
    prompt: str = "Once upon a time"

   
    save_checkpoint_iter: int = 50
    total_iters: int = 20000
    eval_iters: int = 50
    eval_check: int = 100
    warmup_iters: int = 700
    min_lr: float = 0.1 * max_lr  
    lr_decay_iters: int = 20000
    total_batch_size: int = 524288
    micro_batch_size: int = batch_size
    gradient_accumulation_steps: int = total_batch_size // (micro_batch_size * (block_size * torch.cuda.device_count()))