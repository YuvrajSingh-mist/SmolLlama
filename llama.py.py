# 185860
import random
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


# import wandb
# wandb.login()


# from torch.utils.tensorboard import SummaryWriter


from datasets import load_dataset
# use name="sample-10BT" to use the 10BT sample
fw_train = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=False)
# print(fw_train)
# Select only 1000 rows from the dataset
# fw_train = fw_train.select(range(10000000))

# print(fw_train)
# Split the dataset into training and validation sets
fw_train = fw_train.train_test_split(test_size=0.2)
# print(fw_train)

# Access the splits
# train_dataset = train_val_split['train']
# val_dataset = train_val_split['test']

# train_dataset = fw_train.train_test_split(test_size=0.2)


def setup(rank=None, world_size=None):
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    init_process_group("nccl")
    # torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    
def cleanup():
    destroy_process_group()



@dataclass
class ModelArgs:
    #Hyperparameters
    
    epochs = 5
    block_size = 128
    batch_size = 64
    embeddings_dims = 786
    attn_dropout = 0.1
    no_of_heads = 6 #IMP needs to be thoroughly calculated
    dropout = 0.1
    # epochs = 100
    val_epochs = 2
    max_lr = 2e-4
    no_of_decoder_layers = 6 #IMP needs to be thoroughly calculated
    weight_decay_optim = 0.1
    beta_1 = 0.9
    beta_2 = 0.95
    clip = 1.0
    device = 'cuda'
    no_kv_heads = 2
    vocab_size = 50258


from pathlib import Path
data_path = Path('data')
data_path.mkdir(exist_ok=True)
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
# !cp input.txt data/input.txt



#Datasets

# Using tinyshakespeare

# with open('data/input.txt', 'r', encoding='utf-8') as f:
#     text = f.read()


# Load the tokenizer
# tokenizer = Tokenizer.from_file("bpe_tokenizer_30k.json")

# Encode and decode functions
# encode = lambda s: tokenizer.encode(s).ids
# decode = lambda l: tokenizer.decode(l)



def _save_snapshot(model, optimizer, epoch, step):
    snapshot = {}
    snapshot["MODEL_STATE"] =model.module.state_dict()
    snapshot["OPTIMIZER_STATE"]= optimizer.state_dict(),
    snapshot["EPOCHS_RUN"] = epoch
    snapshot["STEP_RUN"] = step
    torch.save(snapshot, "snapshot.pt")
    print(f"Epoch: {epoch} | step {step} | Training snapshot saved at snapshot.pt")

def _load_snapshot(epoch, model, optimizer, step, snapshot_path):
    snapshot = torch.load(snapshot_path)
    model = model.load_state_dict(snapshot["MODEL_STATE"])
    optimizer - optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"]),
    # snapshot["EPOCHS_RUN"] = epoch
    # snapshot["STEP_RUN"] = step
    print(f"Resuming training from snapshot at Epoch {epoch}")
    return (model, optimizer)


#Subword level tokenization

#Loading custom trained BPE
# Load the tokenizer
# tokenizer = Tokenizer.from_file("data/bpe_tokenizer_tinyshakespeare_1k.json")
# vocab_size = tokenizer.get_vocab_size()
# Encode and decode functions
# encode = lambda s: tokenizer.encode(s).ids
# decode = lambda l: tokenizer.decode(l)





###############################################################################
#Character level tokenization

# # here are all the unique characters that occur in this text
# chars = sorted(list(set(text)))
# vocab_size = len(chars)


# # create a mapping from characters to integers
# stoi = { ch: i for i,ch in enumerate(chars) }
# itos = { i:ch for i,ch in enumerate(chars) }
# encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
# decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


# Convert the dataset to Hugging Face Dataset format
# train_hf_dataset = Dataset.from_dict({"text": train_dataset['train']['text']})
# val_hf_dataset = Dataset.from_dict({"text": train_dataset['test']['text']})

# Tokenize the dataset using the `map` function


# from google.colab import userdata
# HF_TOKEN = userdata.get('HF_TOKEN')

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", hf_token = '...')
# tokenizer.pad_token = tokenizer.eos_token
# if tokenizer.pad_token is None:
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
print("ADDED THE PADDING TOKEN: ", tokenizer.pad_token_id)

# model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        max_length=ModelArgs.block_size,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )


## Load the tokenizer
# tokenizer = Tokenizer.from_file("bpe_tokenizer_30k.json")

# # Tokenization functions
# def encode_train(examples):
#     tokens = []
#     for example in examples['text']:
#         out = tokenizer.encode(example).ids
#         tokens.append(out)  # Append the tokenized sequence (do not flatten)
#     return {"tokens": tokens}

# def encode_val(examples):
#     tokens = []
#     for example in examples['text']:
#         out = tokenizer.encode(example).ids
#         tokens.append(out)  # Append the tokenized sequence (do not flatten)
#     return {"tokens": tokens}

# Apply tokenization with batching
# train_data = train_dataset['train'].map(tokenize_function, batched=True, batch_size=8000, remove_columns=['id', 'dump', 'url', 'date', 'file_path', 'language', 'language_score', 'token_count'], num_proc=8)
# val_data = train_dataset['test'].map(tokenize_function, batched=True, batch_size=8000, remove_columns=['id', 'dump', 'url', 'date', 'file_path', 'language', 'language_score', 'token_count'], num_proc=8)

# # # Extract tokens from the processed datasets
# # train_tokens = train_data['tokens']
# # val_tokens = val_data['tokens']

# # Flatten the tokenized data
# # train_tokens = [token_id for seq in train_data['input_ids'] for token_id in seq]
# # val_tokens = [token_id for seq in val_data['input_ids'] for token_id in seq]

# try:
#     train_tensors = [torch.tensor(seq) for seq in tqdm(train_data['input_ids'], desc="Converting train_data to tensors")]
#     train_data_tensor = torch.cat(train_tensors)
# except Exception as e:
#     print(f"Error during tensor conversion: {e}")

# try:
#     train_tensors = [torch.tensor(seq) for seq in tqdm(val_data['input_ids'], desc="Converting train_data to tensors")]
#     val_data_tensor = torch.cat(train_tensors)
# except Exception as e:
#     print(f"Error during tensor conversion: {e}")
# print("Train tokens count: ", train_data_tensor)
# print("Val tokens count: ", val_data_tensor)


def prepare_dataset(split, batch_size):
    # Load a subset of the C4 dataset with a glob pattern for specific training files
    # dataset = load_dataset("allenai/c4", data_files=["en/c4-train.00001-of-01024.json.gz"], trust_remote_code=True)

    # Initialize tokenizer
    # tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def collate_fn(batch):
        # Extract text data
        texts = [item["text"] for item in batch]

        # Set the pad token if it isn't set already
        # if tokenizer.pad_token is None:
        #     tokenizer.pad_token = tokenizer.eos_token

        # Tokenize text data
        encoding = tokenizer(texts, max_length = ModelArgs.block_size, padding='max_length', truncation=True, return_tensors="pt")
        encoding["labels"] = encoding["input_ids"].clone()  # Use `input_ids` as labels
        encoding["labels"][:, :-1] = encoding["input_ids"][:, 1:]  # Shift right
        encoding["labels"][:, -1] = tokenizer.pad_token_id    # Ignore the last token (no target for it)
        # Return tokenized input tensors
        return encoding

    # Create DistributedSampler for proper shuffling and partitioning across processes
    # dist_sampler = DistributedSampler(fw_train["text"], shuffle=True)

    # Create DataLoader with custom collate_fn
    # print(fw_dataset)
    dataloader = None
    if(split == 'train'):
        data_loader = DataLoader(
        fw_train['train'],
        batch_size=batch_size,
        sampler=DistributedSampler(fw_train['train'], shuffle=True),
        collate_fn=collate_fn,
        drop_last=True,
        shuffle=False
    )
    elif(split == 'val'):
        data_loader = DataLoader(
        fw_train['test'],
        batch_size=batch_size,
        sampler=DistributedSampler(fw_train["test"], shuffle=True),
        collate_fn=collate_fn,
        drop_last=True,
        shuffle=False
    )

    return data_loader
# Convert to tensors
# train_data_tensor = torch.tensor(train_tokens, dtype=torch.long)
# val_data_tensor = torch.tensor(val_tokens, dtype=torch.long)

# # Debug output
# print("Number of train tokens:", len(train_data_tensor))
# print("Number of validation tokens:", len(val_data_tensor))


# def create_sequences(data, block_size):
#     sequences = []

#     for seq in data:
#         if len(seq) < block_size:
#             # while(len(sequence) < block_size):
#                 # sequence = data[i:i + block_size + 1]
           
#                 # Pad the sequence if it's shorter than block_size
#             padding_length = block_size - len(seq)
#             seq = torch.cat([seq, torch.full((padding_length,), tokenizer.pad_token_id, dtype=torch.long)])
#         sequences.append(seq)
#     out = torch.tensor(sequences, dtype=torch.long)
#     return out

# train_data = create_sequences(train_data['input_ids'], ModelArgs.block_size)
# val_data = create_sequences(val_data['input_ids'], ModelArgs.block_size)


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


# train_rows = 11895089
# encoded_data = torch.tensor(encode(fw_train['text']), dtype=torch.long)
# train_data = train_data[:train_rows]
# val_data = val_data[train_rows:]

# train_dataset = TokenDataset(train_data_tensor, ModelArgs.block_size)
# val_dataset = TokenDataset(val_data_tensor, ModelArgs.block_size)
# encoded_data = torch.tensor(encode(text), dtype=torch.long)

# print(train_data)
# print(val_data)
# train_dataset = TextDataset(train_data, ModelArgs.block_size)
# val_dataset = TextDataset(val_data, ModelArgs.block_size)

# print(train_dataset)
# print(val_dataset)


# # Convert the tokenized data into a list of sequences
# train_sequences = [train_data[i:i + ModelArgs.block_size] for i in range(0, len(train_data) - ModelArgs.block_size)]
# val_sequences = [val_data[i:i + ModelArgs.block_size] for i in range(0, len(val_data) - ModelArgs.block_size)]

# Define collate_fn
# def collate_fn(batch):
#     block_size = ModelArgs.block_size
#     batch_size = len(batch)
#     x = torch.zeros((batch_size, block_size), dtype=torch.long)
#     y = torch.zeros((batch_size, block_size), dtype=torch.long)
#     for i, sequence in enumerate(batch):
#         print("Shape x: ", sequence[:-1].shape)
#         print("Shape of y: ", len(sequence[1:]))
#         x[i] = sequence[:-1]  # Input is all tokens except the last one
#         y[i] = sequence[1:]   # Target is all tokens except the first one
#     return x, y



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
            # while(len(sequence) < block_size):
                # sequence = data[i:i + block_size + 1]
           
                # Pad the sequence if it's shorter than block_size
            # padding_length = block_size - len(seq)
            # seq = torch.cat([seq, torch.full((padding_length,), tokenizer.encode('[PAD]').ids[0], dtype=torch.long)])
        sequences.append(seq)
    out = torch.tensor(sequences, dtype=torch.long)
    return out

# train_data = create_sequences(train_data_flat['input_ids'], ModelArgs.block_size)
# val_data = create_sequences(val_data['input_ids'], ModelArgs.block_size)


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

    # print("Shape of x: ", len(x))
    # print("Length of y: ", len(y))
    # x, y = x.to(ModelArgs.device), y.to(ModelArgs.device)
    # x = torch.zeros((batch_size, block_size), dtype=torch.long)
    # y = torch.zeros((batch_size, block_size), dtype=torch.long)
    # for i, sequence in enumerate(batch):
    #     print("Seq: ", sequence)
    #     print("Shape x: ", sequence[:-1].shape)
    #     print("Shape of y: ", len(sequence[1:]))
    #     x[i] = sequence[:-1]  # Input is all tokens except the last one
    #     y[i] = sequence[1:]   # Target is all tokens except the first one
    return x, y
    

class Normalization(nn.Module):
    def __init__(
        self,

        embeddings_dims: int = ModelArgs.embeddings_dims
    ):
        super().__init__()
        self.rmsnorm_layer = torch.nn.RMSNorm(normalized_shape=embeddings_dims)


    def forward(self, x):

        x = self.rmsnorm_layer(x)
        return x



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


    # def init_matrix(self, seq_len):
    #         self.matrix = torch.zeros((seq_len, self.embeddings_dims, self.embeddings_dims), dtype=torch.float32,  requires_grad=False)
    #         for pos in range(seq_len):
    #             for j in range(1, self.embeddings_dims // 2):
    #                 self.theta = 10000 ** (-2*(pos-1) / self.embeddings_dims)
    #                 self.matrix[pos, 2*j + 1, 2*j + 1] = np.cos((pos*self.theta))
    #                 self.matrix[pos, 2*j + 1, j + 1] = -np.sin((pos* self.theta))
    #                 self.matrix[pos, 2*j , 2*j ] = -np.cos((pos* self.theta))
    #                 self.matrix[pos, 2*j + 1, 2*j + 1] = np.sin((pos* self.theta))
    #         return self.matrix
        self.device=device

    def init_matrix(self, seq_len):
        self.matrix = torch.zeros((seq_len, self.embeddings_dims, self.embeddings_dims), dtype=torch.float32,  requires_grad=False,  device = self.device)

        positions = torch.arange(seq_len,  dtype=torch.float32,  device = self.device).unsqueeze(1)
        # dims = torch.arange(1, self.embeddings_dims // 2,  dtype=torch.float32)
        theta = 10000 ** (-2 * (positions - 1) / self.embeddings_dims)
        angles = positions * theta

        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)

        indices = torch.arange(self.embeddings_dims,  dtype=torch.int64,  device = self.device)
        # print(indices)
        # print(indices.shape)
        # print(indices[::2])
        even_indices = indices[::2]
        odd_indices = indices[1::2]

        self.matrix[:, even_indices, even_indices] = cos_angles
        self.matrix[:, odd_indices, odd_indices] = sin_angles
        self.matrix[:, odd_indices, even_indices] = -sin_angles
        self.matrix[:, even_indices, odd_indices] = cos_angles

        return self.matrix

    def forward(self, x):
        # B,T,C = x.shape
        # print("MATRIX:",x)
        if(x > self.block_size or x < self.block_size):
            matrix = self.init_matrix(x)
            return matrix
        else:
            matrix = self.init_matrix(self.block_size)

            return matrix


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
        self.query = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims,  bias=False, dtype=torch.float32,  device = device)
        self.key = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims,  bias=False, dtype=torch.float32,  device = device)
        self.value = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims,  bias=False, dtype=torch.float32,  device = device)
        self.rotary_matrix = RotaryEmbeddings(embeddings_dims=embeddings_dims,  device = device)
        self.dropout = nn.Dropout(p = attn_dropout)
        self.device = device
    def forward(self,x):
        # print(x.shape)
        batch, block_size, embeddings_dims = x.shape
        query = self.query(x)
        # print(query)
        key = self.key(x)
        values = self.value(x)
        matrix = self.rotary_matrix(block_size)

        # print(matrix.shape)
        # print(query.shape)
        masked = torch.tril(torch.ones((block_size, block_size),  requires_grad=False,  device = self.device))
        rotary_query = matrix @ query.permute(1,2,0) # (B,T, C,C) @ (B,T,C) -> (B,C,T) = (B,T,C,T)
        rotary_key = matrix @ key.permute(1,2,0)  #  (B,T, C,C  ) @ (B,T,C) -> (B,C,T) = (B,T,C,T)
        weights = rotary_query.permute(2,0,1) @ rotary_key.permute(2,0,1).transpose(-2, -1)#(B,T,C,T) @ (B,T,C,T) = (T,C,C,T)
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
        embeddings_dims: int = ModelArgs.embeddings_dims,
        block_size: int = ModelArgs.block_size,
        no_of_kv_heads: int = ModelArgs.no_of_heads,
        no_of_heads: int = ModelArgs.no_of_heads,

    ):
        super().__init__()

        self.no_of_kv_heads = no_of_kv_heads
        self.no_of_q_heads = no_of_heads // no_of_kv_heads
        self.head_size = embeddings_dims // self.no_of_q_heads
        self.rotary_matrix = RotaryEmbeddings(embeddings_dims=embeddings_dims,  device = device)
        # self.query = nn.Linear(in_features=embeddings_dims, out_features=self.head_size,  bias=False)
        self.key = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims,  dtype=torch.float32, bias=False,  device = device)
        self.value = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims,  dtype=torch.float32, bias=False,  device = device)
        self.dropout = nn.Dropout(p = ModelArgs.attn_dropout)
        self.linear_layer = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims,  dtype=torch.float32, bias=False,  device = device)
        self.device = device
        self.multi_query = nn.ModuleList([nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims,  bias=False,  device = self.device) for _ in range(self.no_of_q_heads)])

    def scaled_dot_product(self, q, k, v, block_size, matrix):

            # masked = torch.tril(torch.ones((block_size, block_size),  requires_grad=False,  device = self.device))

            masked = torch.tril(torch.ones((block_size, block_size),  requires_grad=False,  device = self.device))
            rotary_query = matrix @ q.permute(1,2,0) # (B,T, C,C) @ (B,T,C) -> (B,C,T) = (B,T,C,T)
            rotary_key = matrix @ k.permute(1,2,0)  #  (B,T, C,C  ) @ (B,T,C) -> (B,C,T) = (B,T,C,T)
            weights = rotary_query.permute(2,0,1) @ rotary_key.permute(2,0,1).transpose(-2, -1)#(B,T,C,T) @ (B,T,C,T) = (T,C,C,T)
            weights_masked = weights.masked_fill(masked == 0, float('-inf'))
            scaled_weights = weights_masked / (torch.sqrt(torch.tensor(k.shape[-1])))
            scaled_weights = F.softmax(scaled_weights, dim=-1)
            value = scaled_weights @ v
            out = self.dropout(value)
            return value

    def forward(self,x):
        # print("MQA: ", x.shape)
        batch, block_size, embeddings_dims = x.shape

        # query = self.query(x)
        matrix = self.rotary_matrix(block_size)


        key = self.key(x)
        values = self.value(x)

        multi_query_concat = torch.cat([self.scaled_dot_product(query(x), key, values, block_size, matrix) for query in self.multi_query], dim=-1)


        linear_layer= self.linear_layer(multi_query_concat)
        out = self.dropout(linear_layer)
        return out


class GQA(nn.Module):
    def __init__(
        self,
         device,
        embeddings_dims: int = ModelArgs.embeddings_dims,
        block_size: int = ModelArgs.block_size,
        no_of_q_heads: int = ModelArgs.no_of_heads,
        no_of_kv_heads: int = ModelArgs.no_kv_heads
    ):
        super().__init__()

        self.no_of_kv_heads = no_of_kv_heads
        self.no_of_q_heads = no_of_q_heads
        self.dropout = nn.Dropout(p = ModelArgs.attn_dropout)
        self.linear_layer = nn.Linear(in_features=embeddings_dims * self.no_of_kv_heads, out_features=embeddings_dims , dtype=torch.float32,  bias=False,  device = device)
        self.device = device
        self.mqa = nn.ModuleList([MQA(embeddings_dims=embeddings_dims, device = self.device, block_size=block_size) for _ in range(self.no_of_kv_heads)])

    def forward(self,x):

        batch, block_size, embeddings_dims = x.shape


        grouped_query_concat = torch.cat([group(x) for group in self.mqa], dim=-1)

        linear_layer= self.linear_layer(grouped_query_concat)
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

        self.swish = Swish(block_size=block_size, embeddings_dims=embeddings_dims, device=device)
        self.linear_layer1 = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims,  bias=False, dtype=torch.float32,  device = device)
        self.linear_layer2 = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims,  bias=False, dtype=torch.float32,  device = device)
        self.linear_layer3 = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims,  bias=False, dtype=torch.float32,  device = device)




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

        self.linear_layer = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims,  dtype=torch.float32,  device = device)
        self.swiglue = SWiGLU(block_size=block_size, embeddings_dims=embeddings_dims,  device = device)
        self.dropout = nn.Dropout(p = dropout)
    def forward(self, x):

        x = self.swiglue(x)
        x = self.linear_layer(x)
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
        self.gqa = GQA(embeddings_dims=embeddings_dims, block_size=block_size, no_of_kv_heads=ModelArgs.no_kv_heads, no_of_q_heads=ModelArgs.no_of_heads,  device = device)
        # self.norm = Normalization(embeddings_dims=embeddings_dims)
        self.norm1 = Normalization(embeddings_dims=embeddings_dims)
        self.norm2 = Normalization(embeddings_dims=embeddings_dims)
        self.dropout = nn.Dropout(p = dropout)
    def forward(self, x):

        x = self.norm1(x + self.gqa(x))
        x = self.norm2(x + self.feedforward_network(x))
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
    def forward(self, x):
        x = self.embeddings(x)
        x = self.dropout(x)
        x = self.decoder(x)
        # x = self.norm(x)
        x = self.linear_layer(x)
        # out = self.norm(x)
        return x


# device = "cuda" if torch.cuda.is_available() else "cpu"
# # device = "cpu"
# ModelArgs.device = device
# model = Llama(device=ModelArgs.device, embeddings_dims=ModelArgs.embeddings_dims, block_size=ModelArgs.block_size, vocab_size=ModelArgs.vocab_size, dropout=ModelArgs.dropout)
# model = model.to(ModelArgs.device)

#Printing a summary of the architecture
# !pip install torchinfo
# from torchinfo import summary
# # idx, targets = get_batch('test')
# idx = torch.randint(
#         low=0,
#         high=ModelArgs.vocab_size,
#         size=(ModelArgs.batch_size, ModelArgs.block_size),
#         dtype=torch.long
#     )
# # sample_idx = random.randint(range(len(train_dataset)))
# # idx, targets = train_dataset[0]
# idx = idx.to(ModelArgs.device)
# # targets = targets.to(ModelArgs.device)
# summary(model=model,
#         input_data=idx,
#         # input_size=(ModelArgs.batch_size, ModelArgs.block_size, ModelArgs.embeddings_dims),
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"])


def find_unused_parameters(model):
    unused = []
    for name, param in model.named_parameters():
        if param.grad is None:
            unused.append(name)
    return unused



def beam_search(model, tokenizer, input_text, beam_width=5, max_length=50, temperature=1.0):

    model.eval()  # Set model to evaluation mode
    device = next(model.parameters()).device  # Get the device of the model

    # Encode the input text (without padding)
    input_ids = tokenizer(input_text, max_length = ModelArgs.block_size, truncation=True, padding='max_length', return_tensors="pt").to(device)['input_ids']
    # batch_size = input_ids.size(0)

    # Initialize beams
    beams = [(input_ids, 0)]  # (sequence, cumulative log probability)

    for _ in range(max_length):
        new_beams = []
        for beam_seq, beam_score in beams:
            # Stop if the sequence reaches max_length
            if beam_seq.size(1) >= max_length:
                new_beams.append((beam_seq, beam_score))
                continue

            # Get the last token in the beam
            last_token = beam_seq[:, -1:]

            # Forward pass to get logits
            with torch.no_grad():
                outputs = model(beam_seq)
                logits = outputs[:, -1, :] / temperature  # Apply temperature scaling

            # Get top-k tokens and their probabilities
            log_probs = F.log_softmax(logits, dim=-1)
            topk_log_probs, topk_tokens = torch.topk(log_probs, beam_width, dim=-1)

            # Expand beams
            for i in range(beam_width):
                token = topk_tokens[0, i].unsqueeze(0).unsqueeze(0)
                log_prob = topk_log_probs[0, i].item()
                new_seq = torch.cat([beam_seq, token], dim=-1)
                new_score = beam_score + log_prob
                new_beams.append((new_seq, new_score))

        # Keep only the top-k beams
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_width]

        # Stop if all beams end with the EOS token
        if all(beam_seq[:, -1].item() == tokenizer.eos_token_id for beam_seq, _ in beams):
            break

    # Select the beam with the highest score
    best_beam_seq, best_beam_score = beams[0]

    # Decode the generated sequence
    generated_text = tokenizer.decode(best_beam_seq[0], skip_special_tokens=True)
    return generated_text


def greedy_decode(model, tokenizer, prompt, max_length=50):
    device = next(model.parameters()).device
    block_size = ModelArgs.block_size

    # Encode the prompt without adding extra padding
    input_ids = tokenizer(prompt, max_length=block_size, truncation=True, padding=False, return_tensors="pt").to(device)['input_ids']
    
    # Initialize the window with the input_ids
    window = input_ids.clone()

    # Get the length of the prompt
    prompt_length = input_ids.shape[1]

    # We will generate up to (max_length - prompt_length) new tokens.
    generated_tokens = []
    for _ in range(max_length):
        # Forward pass through the model
        outputs = model(window)
        
        # Get logits for the last token position in the window
        logits = outputs[:, -1, :]  # Shape: (1, vocab_size)
        
        # Greedy decode: choose the token with the maximum probability
        next_token = torch.argmax(logits, dim=-1).unsqueeze(0)  # Shape: (1, 1)
        
        # If the model produces the EOS token or padding token, we stop generation.
        if next_token.item() == tokenizer.eos_token_id:
            break

        # Append the generated token (for later decoding)
        generated_tokens.append(next_token.item())
        
        # Shift the window: remove the first token and append the new token
        window = torch.cat([window[:, 1:], next_token], dim=1)

    # Combine the original prompt tokens with generated tokens
    full_sequence = input_ids[0].tolist() + generated_tokens
    decoded_text = tokenizer.decode(full_sequence, skip_special_tokens=True)
    
    return decoded_text
def save_to_file(text):
    
    with open('generations.txt', 'a') as f:
        f.writelines(text + "\n\n")
        
    
#Train the  model


# writer = SummaryWriter(log_dir="runs/experiment")


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
                loader.sampler.set_epoch(step)
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
            if device == 0 and step % 500 == 0 and step != 0:
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
        writer.close()
    #     wandb.finish()
    cleanup()


world_size = torch.cuda.device_count()
print(f"World size: {world_size}")
train()

