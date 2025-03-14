# 185860
import random
from uu import encode
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
# from torchtune.modules import RMSNorm
from tokenizers import Tokenizer
from pathlib import Path
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers.models.prophetnet.modeling_prophetnet import ProphetNetDecoderModelOutput
import wandb
from tqdm import tqdm
from functools import partial

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import os


from datasets import load_dataset, concatenate_datasets

train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
val_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="test")


def setup(rank=None, world_size=None):
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    init_process_group("nccl")
    # torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    
def cleanup():
    destroy_process_group()




@dataclass
class ModelArgs:
    
    epochs = 4
    beta = 0.1
    block_size = 256
    batch_size = 32
    embeddings_dims = 512
    attn_dropout = 0.1
    no_of_heads = 8 
    dropout = 0.1
    val_epochs = 2
    max_lr = 1e-6
    no_of_decoder_layers = 16 
    weight_decay_optim = 0.1
    beta_1 = 0.9
    beta_2 = 0.95
    clip = 1.0
    device = 'cuda'
    no_kv_heads = 2
    vocab_size = 50304 
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'



from pathlib import Path
data_path = Path('data')
data_path.mkdir(exist_ok=True)







def _save_snapshot(model, optimizer, scheduler, epoch, step):
    snapshot = {
        "MODEL_STATE": model.module.state_dict(),
        "OPTIMIZER_STATE": optimizer.state_dict(),
        # "SCHEDULER_STATE": scheduler.state_dict(),  # NEW: Save scheduler state
        "EPOCHS_RUN": epoch,
        "STEP_RUN": step
    }
    torch.save(snapshot, f"DPO_model_{step}.pt")
    print(f"Epoch: {epoch} | Step: {step} | Snapshot saved.")

def _load_snapshot(snapshot_path, model, optimizer, scheduler):
    snapshot = torch.load(snapshot_path)
    model.load_state_dict(snapshot["MODEL_STATE"])
    # optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
    # scheduler.load_state_dict(snapshot["SCHEDULER_STATE"])  # Load scheduler state
    epoch = snapshot["EPOCHS_RUN"]
    step = snapshot["STEP_RUN"]
    print(f"Resuming from Epoch {epoch}, Step {step}")
    return epoch, step




tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        max_length=ModelArgs.block_size,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )


def prepare_dataset(split, device, batch_size):


    def collate_fn(batch):
        # Extract text data

        merged_chosen_prompts = []
        merged_rejected_prompts = []

        for sample in batch:

            # print(sample)
            encodings = {}
            # Extract and merge chosen response
            prompt = sample['chosen'][0]['content']
            chosen_data = sample['chosen'][1]['content']
            chosen_data = "### Instruction: " + "Follow the given instructions carefully." + "\n\n" + "### Input: " + prompt + "\n\n" + "### Response: " + chosen_data
            # Extract and merge rejected response
            rejected_data = sample['rejected'][1]['content']
            rejected_data = "### Instruction: " + "Follow the given instructions carefully." + "\n\n" +  "### Input: " + prompt + "\n\n" + "### Response: " + rejected_data

            
          
            merged_chosen_prompts.append(chosen_data)


            merged_rejected_prompts.append(rejected_data)

        tokenized_win_prompt = tokenizer(merged_chosen_prompts, max_length = ModelArgs.block_size, padding='max_length', truncation=True,  return_tensors="pt").to(device)
        tokenized_win_prompt['input_ids'][: , -1] = tokenizer.eos_token_id 
        tokenized_lose_prompt = tokenizer(merged_rejected_prompts, max_length = ModelArgs.block_size, truncation=True, padding='max_length', return_tensors="pt").to(device)
        tokenized_lose_prompt['input_ids'][: , -1] = tokenizer.eos_token_id 

        encodings['chosen'] = tokenized_win_prompt
        encodings['rejected'] = tokenized_lose_prompt

        return encodings

    dataloader = None
    if(split == 'train'):
        data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=DistributedSampler(train_dataset,  shuffle=True),
        collate_fn=collate_fn,
        drop_last=True,
        shuffle=False
    )
    elif(split == 'val'):
        data_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=DistributedSampler(val_dataset, shuffle=True),
        collate_fn=collate_fn,
        drop_last=True,
        shuffle=False
    )

    return data_loader

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - ModelArgs.block_size, (ModelArgs.batch_size,))
    x = torch.stack([data[i:i+ModelArgs.block_size] for i in ix])
    y = torch.stack([data[i+1:i+ModelArgs.block_size+1] for i in ix])
    x, y = x.to(ModelArgs.device), y.to(ModelArgs.device)
    return x, y

from torch.utils.data import Dataset

class TokenDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size  # Ensure valid indexing

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)




def create_sequences(data, block_size):
    sequences = []

    for seq in data:
        len(seq)
        if len(seq) < block_size:
            # while(len(sequence) < block_size):
                # sequence = data[i:i + block_size + 1]
           
                # Pad the sequence if it's shorter than block_size
            padding_length = block_size - len(seq)
            seq = torch.cat([seq, torch.full((padding_length,), tokenizer.encode('[PAD]').ids[0], dtype=torch.long)])

        else:
            if len(seq) > block_size:
                seq = seq[:block_size]
       
        sequences.append(seq)
    out = torch.tensor(sequences, dtype=torch.long)
    return out



# Define collate_fn
def collate_fn(split , batch):
    block_size = ModelArgs.block_size
    batch_size = len(batch)
    if(split == 'train'):
        data = train_data_tensor
    elif(split == 'test'):
        data = val_data_tensor
    ix = torch.randint(len(data) - ModelArgs.block_size, (ModelArgs.batch_size,))
    x = torch.stack([data[i:i+ModelArgs.block_size] for i in ix])
    y = torch.stack([data[i+1:i+ModelArgs.block_size+1] for i in ix])


    return x, y
    

class Normalization(nn.Module):
    def __init__(
        self,

        embeddings_dims: int = ModelArgs.embeddings_dims
    ):
        super().__init__()
        self.rmsnorm_layer = torch.nn.RMSNorm(normalized_shape=embeddings_dims)


    def forward(self, x):

        # Ensure the weight is on the same device as x
        weight = self.rmsnorm_layer.weight.to(x.device)
        return F.rms_norm(x, self.rmsnorm_layer.normalized_shape, weight, self.rmsnorm_layer.eps)




# import numpy as np
class RotaryEmbeddings(nn.Module):
    def __init__(
        self,
         device,
        embeddings_dims: int = ModelArgs.embeddings_dims,
        block_size: int = ModelArgs.block_size,
        batch_size: int = ModelArgs.batch_size
    ):
        super().__init__()

        self.embeddings_dims = embeddings_dims
        self.block_size = block_size
        self.batch_size = batch_size
        self.theta = 0
        self.device=device



    
    def apply_rope(self, seq):
        batch_size, seq_len, embeds_dims = seq.shape
        # print(seq.shape)
        # print(self.embeddings_dims)
        # self.matrix = torch.zeros((seq_len, self.embeddings_dims, self.embeddings_dims), dtype=torch.float32,  requires_grad=False,  device = self.device)

        positions = torch.arange(0 , embeds_dims, 2, dtype=torch.float32,  device = self.device).unsqueeze(0)
        # dims = torch.arange(1, self.embeddings_dims // 2,  dtype=torch.float32)
        theta = 10000 ** (-2 * (positions) / embeds_dims)
        angles = positions * theta
        angles = angles.expand(seq_len, -1) # because this thing needs to be applied to every sequence in the batch but with embeds dims halved
        x_reshaped = seq.view(batch_size, seq_len, embeds_dims // 2, 2)
        
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)


        out = torch.stack([x_reshaped[..., 0]*cos_angles - (x_reshaped[...,1] * sin_angles), x_reshaped[...,1] * cos_angles + x_reshaped[..., 0] * sin_angles], dim=-1)
        out = out.view(batch_size, seq_len, embeds_dims)
        return out

    def forward(self, x):

        res = self.apply_rope(x)
        return res 
        # else:
            # return self.x_reshaped
    
class RotaryAttentionHead(nn.Module):
    def __init__(
        self,
         device,
        embeddings_dims: int = ModelArgs.embeddings_dims,
        no_of_heads: int = ModelArgs.no_of_heads,
        attn_dropout: int = ModelArgs.attn_dropout
    ):
        super().__init__()
        self.head_size = embeddings_dims // no_of_heads
        self.query = nn.Linear(in_features=embeddings_dims, out_features=self.head_size,  bias=False, dtype=torch.float32,  device = device)
        self.key = nn.Linear(in_features=embeddings_dims, out_features=self.head_size,  bias=False, dtype=torch.float32,  device = device)
        self.value = nn.Linear(in_features=embeddings_dims, out_features=self.head_size,  bias=False, dtype=torch.float32,  device = device)
        self.rope = RotaryEmbeddings(embeddings_dims=self.head_size,  device = device)
        self.dropout = nn.Dropout(p = attn_dropout)
        self.device = device
    def forward(self,x):
        # print(x.shape)
        # print("X is: ", x)
        batch, block_size, embeddings_dims = x.shape
        query = self.query(x)
        # print(query)
        key = self.key(x)
        values = self.value(x)
        # matrix = self.rotary_matrix(block_size)
        rotary_q = self.rope(query)
        rotary_k = self.rope(key)
        
        # print(matrix.shape)
        # print(query.shape)
        masked = torch.tril(torch.ones((block_size, block_size),  requires_grad=False,  device = self.device))
        # rotary_query = matrix @ query.permute(1,2,0) # (B,T, C,C) @ (B,T,C) -> (B,C,T) = (B,T,C,T)
        # rotary_key = matrix @ key.permute(1,2,0)  #  (B,T, C,C  ) @ (B,T,C) -> (B,C,T) = (B,T,C,T)
        weights = rotary_q.permute(2,0,1) @ rotary_k.permute(2,0,1).transpose(-2, -1)#(B,T,C,T) @ (B,T,C,T) = (T,C,C,T)
        weights_masked = weights.masked_fill(masked == 0, float('-inf'))
        scaled_weights = weights_masked / (torch.sqrt(torch.tensor(key.shape[-1])))
        scaled_weights = F.softmax(scaled_weights, dim=-1)
        value = scaled_weights @ values
        out = self.dropout(value)
        return out



class MQA(nn.Module):
    def __init__(
        self,
        device,
        no_of_q_heads: int,
        embeddings_dims: int = ModelArgs.embeddings_dims,
        block_size: int = ModelArgs.block_size,
        

    ):
        super().__init__()


        # self.no_of_q_heads = no_of_heads // no_of_kv_heads
        # self.no_of_q_heads = no_of_q_heads
        self.no_of_kv_heads = 2 # I want to have a kv for each pair of query heads 
        self.head_size = embeddings_dims // no_of_q_heads
        # self.kv_head_size = (embeddings_dims // self.no_of_kv_heads) * 2
        self.rotary= RotaryEmbeddings(embeddings_dims=self.head_size,  device = device)
        # self.rotary_k = RotaryEmbeddings(embeddings_dims=self.kv_head_size,  device = device)
        # self.query = nn.Linear(in_features=embeddings_dims, out_features=self.head_size,  bias=False)
        self.key = nn.Linear(in_features=embeddings_dims, out_features=self.head_size,  dtype=torch.float32, bias=False,  device = device)
        self.value = nn.Linear(in_features=embeddings_dims, out_features=self.head_size,  dtype=torch.float32, bias=False,  device = device)
        self.dropout = nn.Dropout(p = ModelArgs.attn_dropout)
        self.linear_layer = nn.Linear(in_features=self.head_size * self.no_of_kv_heads, out_features=embeddings_dims,  dtype=torch.float32, bias=False,  device = device)
        self.device = device
        self.multi_query = nn.ModuleList([nn.Linear(in_features=embeddings_dims, out_features=self.head_size,  bias=False,  device = self.device) for _ in range(self.no_of_kv_heads)])

    def scaled_dot_product(self, q, k, v, block_size):

            # masked = torch.tril(torch.ones((block_size, block_size),  requires_grad=False,  device = self.device))
            q = self.rotary(q)
            masked_table = torch.tril(torch.ones((block_size, block_size),  requires_grad=False,  device = self.device))
            # rotary_query = matrix @ q.permute(1,2,0) # (B,T, C,C) @ (B,T,C) -> (B,C,T) = (B,T,C,T)
            # rotary_key = matrix @ k.permute(1,2,0)  #  (B,T, C,C  ) @ (B,T,C) -> (B,C,T) = (B,T,C,T)
            # print("Query: ", q.shape)
            # print("Keys: ", k.shape)
            # print(q.permute(2,0,1).shape)
            # print(k.permute(2,0,1).transpose(-2, -1).shape)
            # weights = q.permute(2,0,1) @ k.permute(2,0,1).transpose(-2, -1)#(B,T,C,T) @ (B,T,C,T) = (T,C,C,T)
            # weights = q @ k.permute(2,1,0)
            # print(weights.shape)
            # print(masked.shape)
            weights = q @ torch.transpose(k, dim0=-2, dim1=-1) * (k.shape[-1] ** -0.5)
            masked_values = weights.masked_fill(masked_table[: block_size, : block_size] == 0, float('-inf'))
            weights_normalized = nn.functional.softmax(masked_values, dim=-1) #Normalize along the embeddings dimension for all the tokens
            weights_normalized = self.dropout(weights_normalized)
            out = weights_normalized @ v
            return out

    def forward(self,x):
        # print("MQA: ", x.shape)
        batch, block_size, embeddings_dims = x.shape

        # query = self.query(x)
        # matrix = self.rotary_matrix(block_size)


        key = self.key(x)
        values = self.value(x)
        # print("Keys: ", key.shape)
        # print("Values: ", values.shape)
        # rotary_value = self.rotary(values)
        rotary_key = self.rotary(key)
        multi_query_concat = torch.cat([self.scaled_dot_product(query(x), rotary_key, values, block_size) for query in self.multi_query], dim=-1)
        # print("Multi query: ", multi_query_concat.shape)

        linear_layer= self.linear_layer(multi_query_concat)
        # out = self.dropout(linear_layer)
        return linear_layer


class GQA(nn.Module):
    def __init__(
        self,
         device,
        embeddings_dims: int = ModelArgs.embeddings_dims,
        block_size: int = ModelArgs.block_size,
        # no_of_q_heads: int = ModelArgs.no_of_heads,
        mqa_heads: int = ModelArgs.no_kv_heads
    ):
        super().__init__()

        # self.no_of_kv_heads = no_of_kv_heads
        self.no_of_q_heads = ModelArgs.no_of_heads // mqa_heads
        # self.head_dim = embeddings_dims // self.no_kv_heads
        self.dropout = nn.Dropout(p = ModelArgs.attn_dropout)
        self.linear_layer = nn.Linear(in_features=embeddings_dims * self.no_of_q_heads, out_features=embeddings_dims , dtype=torch.float32,  bias=False,  device = device)
        self.device = device
        self.mqa = nn.ModuleList([MQA(no_of_q_heads=self.no_of_q_heads, embeddings_dims=embeddings_dims, device = self.device, block_size=block_size) for _ in range(self.no_of_q_heads)])
        # self.mqa = MQA(no_of_q_heads=self.no_of_q_heads, device=self.device, embeddings_dims=embeddings_dims, block_size=block_size)
    def forward(self,x):

        batch, block_size, embeddings_dims = x.shape

        # res = self.mqa(x)
        grouped_query_concat = torch.cat([group(x) for group in self.mqa], dim=-1)

        linear_layer= self.linear_layer(grouped_query_concat) #Basically MQA is made into GQA with no_of_q_heads and this class right here is just to consolidate everything into one
        out = self.dropout(linear_layer)
        return out


class Swish(nn.Module):
    def __init__(
        self,
        device,
        block_size: int = ModelArgs.block_size,
        embeddings_dims: int = ModelArgs.embeddings_dims
    ):
        super().__init__()

        self.sig = torch.nn.Sigmoid()


    def forward(self, x):
        swish = x * self.sig(x)

        return swish



class SWiGLU(nn.Module):
    def __init__(
        self,
        device,
        block_size: int = ModelArgs.block_size,
        embeddings_dims: int = ModelArgs.embeddings_dims
    ):
        super().__init__()
        self.hidden_dims = int(2 * ( 4 * embeddings_dims) / 3)
        self.swish = Swish(block_size=block_size, embeddings_dims=embeddings_dims, device=device)
        self.linear_layer1 = nn.Linear(in_features=embeddings_dims, out_features=self.hidden_dims,  bias=False, dtype=torch.float32,  device = device)
        self.linear_layer2 = nn.Linear(in_features=embeddings_dims, out_features=self.hidden_dims,  bias=False, dtype=torch.float32,  device = device)
        self.linear_layer3 = nn.Linear(in_features=self.hidden_dims, out_features=embeddings_dims,  bias=False, dtype=torch.float32,  device = device)




    def forward(self, x):
        swish_res = self.swish(self.linear_layer1(x))
        x_V = self.linear_layer2(x)
        res = torch.mul(swish_res, x_V)
        out = self.linear_layer3(res)
        return out



class FFN(nn.Module):
    def __init__(self,
                  device,
                  embeddings_dims: int = ModelArgs.embeddings_dims,
                  block_size: int = ModelArgs.block_size,
                  vocab_size: int = ModelArgs.vocab_size,
                   dropout = ModelArgs.dropout

                 ):
        super().__init__()

        # self.linear_layer = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims,  dtype=torch.float32,  device = device)
        self.swiglue = SWiGLU(block_size=block_size, embeddings_dims=embeddings_dims,  device = device)
        self.dropout = nn.Dropout(p = dropout)
    def forward(self, x):

        x = self.swiglue(x)
        # x = self.linear_layer(x)
        x = self.dropout(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self,
                  device,
                embeddings_dims: int = ModelArgs.embeddings_dims,
                dropout = ModelArgs.dropout,
                block_size: int = ModelArgs.block_size,
                vocab_size: int = ModelArgs.vocab_size,

                 ) :
        super().__init__()


        self.feedforward_network = FFN(embeddings_dims=embeddings_dims, block_size=block_size, vocab_size=vocab_size,  device = device)
        self.gqa = GQA(embeddings_dims=embeddings_dims, block_size=block_size, mqa_heads=2,  device = device)
        # self.norm = Normalization(embeddings_dims=embeddings_dims)
        self.norm1 = Normalization(embeddings_dims=embeddings_dims)
        self.norm2 = Normalization(embeddings_dims=embeddings_dims)
        self.dropout = nn.Dropout(p = dropout)
    def forward(self, x):

        x = x + self.gqa(self.norm1(x))
        x = x + self.feedforward_network(self.norm2(x))
        return x


class Llama(nn.Module):
    def __init__(self,
                device,
                  embeddings_dims: int = ModelArgs.embeddings_dims,
                  no_of_decoder_layers: int = ModelArgs.no_of_decoder_layers,
                  block_size: int = ModelArgs.block_size,
                  vocab_size: int = ModelArgs.vocab_size,
                  dropout = ModelArgs.dropout

                 ) :
        super().__init__()

        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embeddings_dims,  dtype=torch.float32,  device = device)
        self.decoder = nn.Sequential(*[DecoderLayer(embeddings_dims=embeddings_dims, block_size=block_size, vocab_size=vocab_size, dropout=dropout,  device = device) for _ in range(no_of_decoder_layers)])
        self.linear_layer = nn.Linear(in_features=embeddings_dims, out_features=vocab_size,  dtype=torch.float32,  device = device)
        self.dropout = nn.Dropout(p = dropout)
        # self.norm = Normalization(embeddings_dims)
        
        
        #weight tying
        self.embeddings.weight = self.linear_layer.weight
    
        self.apply(self._init_weights)

    def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
               
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
               
                     
                    
    def forward(self, x):
        x = self.embeddings(x)
        x = self.dropout(x)
        x = self.decoder(x)
        # x = self.norm(x)
        x = self.linear_layer(x)
        # out = self.norm(x)
        return x






# from andrej karapathy github
def topk_sampling(model, prompt, device, max_length=50, top_k=50, temperature=1.0):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated_tokens = []
    ModelArgs.inference=True
    for _ in range(max_length):
        with torch.no_grad():
            outputs = model.module(input_ids)
            logits = outputs[:, -1, :]
            
            probs = F.softmax(logits, dim=-1)
            
            # Top-k filtering
            top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
            
            
            # Apply temperature scaling
            # probs = probs / temperature
            
            # Sample from top-k
            next_token = torch.multinomial(top_k_probs, num_samples=1)
            
            # generated_tokens.append(next_token.item())
            
            xcol = torch.gather(top_k_indices, -1, next_token)
            input_ids = torch.cat([input_ids, xcol], dim=1) #1 because is it the dimension of the sequence
            
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

def beam_search(model, tokenizer, prompt, beam_width=5, max_length=50, temperature=1.0):
    device = next(model.module.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)['input_ids']
    beam_scores = torch.zeros(beam_width, device=device)
    beam_sequences = input_ids.repeat(beam_width, 1)

    for _ in range(max_length):
        outputs = model(beam_sequences)
        logits = outputs[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, beam_width, dim=-1)

        # Expand beams
        beam_scores = beam_scores.unsqueeze(-1) + torch.log(top_probs)
        beam_scores = beam_scores.view(-1)
        top_indices = top_indices.view(-1)

        # Select top beams
        beam_scores, top_beams = torch.topk(beam_scores, beam_width)
        beam_sequences = torch.cat([beam_sequences[top_beams // beam_width], top_indices[top_beams].unsqueeze(-1)], dim=-1)

    # Return the best sequence
    best_sequence = beam_sequences[0]
    return tokenizer.decode(best_sequence, skip_special_tokens=True)




def find_unused_parameters(model):
    unused = []
    for name, param in model.named_parameters():
        if param.grad is None:
            unused.append(name)
    return unused

def greedy_decode(
    model, 
    tokenizer, 
    prompt, 
    device,
    max_length=50, 
    repetition_penalty=1.2, 
    context_window=10, 
    temperature=1.0, 
    eos_token_id=None,
    
):
    # model.eval()
    # device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)['input_ids']
    generated_tokens = []
    eos_token_id = eos_token_id or tokenizer.eos_token_id  # Use EOS token if provided
    
    for _ in range(max_length):
        with torch.no_grad():
            outputs = model.module(input_ids)
            logits = outputs[:, -1, :]  # Get logits for the last token

            # Apply temperature scaling
            # if temperature != 1.0:
                # logits = logits / temperature

            # Apply repetition penalty
            # if repetition_penalty != 1.0 and len(generated_tokens) > 0:
                # for token in set(generated_tokens[-context_window:]):  # Penalize recent tokens
                    # logits[0, token] /= repetition_penalty

            # Greedy selection
            next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
            generated_tokens.append(next_token.item())

            # Stop if EOS token is generated
            # if next_token.item() == eos_token_id:
            #     break

            # Append the new token to the input
            input_ids = torch.cat([input_ids, next_token], dim=1)

    # Decode the generated tokens
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)



def save_to_file(text):
    
    with open('generations.txt', 'a') as f:
        f.writelines(text + "\n\n")
        
    
#Train the  model


# writer = SummaryWriter(log_dir="runs/experiment")

from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR

# Warmup phase for 2000 steps
def warmup_fn(step):
    if step < 2000:
        return step / 2000  # LR gradually increases
    return 1.0


from torch.optim.lr_scheduler import LambdaLR



def remove_prefix(state_dict, prefix):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]  # Remove the prefix
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict



def trapezoidal_lr_scheduler(optimizer, max_lr, total_steps, warmup_steps, plateau_steps, decay_steps):
    """
    Trapezoidal learning rate scheduler:
    - Increases linearly for `warmup_steps` steps.
    - Remains constant for `plateau_steps` steps.
    - Decreases linearly for `decay_steps` steps.
    """
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return float(step) / float(max(1, warmup_steps))
        elif step < warmup_steps + plateau_steps:
            # Constant plateau
            return 1.0
        else:
            # Linear decay
            decay_step = step - (warmup_steps + plateau_steps)
            return max(0.0, float(decay_steps - decay_step) / float(max(1, decay_steps)))

    return LambdaLR(optimizer, lr_lambda)


torch.set_float32_matmul_precision('high')

scaler = torch.amp.GradScaler(enabled=(ModelArgs.dtype == 'float16'))

save_chechpoint_iter = 50
total_iters = 3000
eval_iters = 50
eval_check = 100
warmup_iters = 100
min_lr = 0.1 * ModelArgs.max_lr
lr_decay_iters = 3000
total_batch_size = 131072
micro_batch_size = ModelArgs.batch_size
gradient_accumulation_steps = total_batch_size // (micro_batch_size * (ModelArgs.block_size * torch.cuda.device_count()))

# learning rate decay scheduler (cosine with warmup) from https://github.com/karpathy/nanoGPT/blob/master/train.py

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return ModelArgs.max_lr * (it + 1) / (warmup_iters + 1)
    else:
        return ModelArgs.max_lr
    # 2) if it > lr_decay_iters, return min learning rate
    # if it > lr_decay_iters:
        # return min_lr
    # 3) in between, use cosine decay down to min learning rate
    # decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    # assert 0 <= decay_ratio <= 1
    # coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
    # return min_lr + coeff * (ModelArgs.max_lr - min_lr)


def train():


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
  
    dict_model = torch.load('snapshot_fine_tuned_model_900.pt')
    dict_model['MODEL_STATE'] = remove_prefix(dict_model['MODEL_STATE'], '_orig_mod.')
    ref_model.load_state_dict(dict_model['MODEL_STATE'])

    dict_model = torch.load('snapshot_fine_tuned_model_900.pt')
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




world_size = torch.cuda.device_count()
print(f"World size: {world_size}")
train()

