
import torch.nn.functional as F

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


from datasets import load_dataset
from torch.utils.data import DataLoader
from tokenizer import Tokenizer
from config import ModelArgs



tokenizer = Tokenizer().ready_tokenizer()


import argparse

def load_selected_dataset(tinystories, fw, dpo):
    if tinystories:
        train_dataset = load_dataset("roneneldan/TinyStories", split="train")
        val_dataset = load_dataset("roneneldan/TinyStories", split="validation")
    elif fw:
        train_dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=False)
        val_dataset = train_dataset.train_test_split(test_size=0.01)
    elif dpo:
        train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
        val_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="test")
    else:
        raise ValueError("At least one dataset flag must be set to True.")
    
    print("Dataset loaded!")
    return train_dataset, val_dataset

# print("Dataset loaded!")

# train_dataset, val_dataset = load_selected_dataset(tinystories, fw, dpo)



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
    print("Device is: ", device)

    def collate_fn(batch):
        # Extract text data
        texts = [item ["text"] for item in batch]


        input_encodings = tokenizer(texts, max_length = ModelArgs.block_size, padding='max_length', truncation=True, return_tensors="pt")

        input_encodings["labels"] = input_encodings["input_ids"].clone() 
        
        input_encodings["labels"][:, :-1] = input_encodings["input_ids"][:, 1:]  
        input_encodings["labels"][:, -1] = tokenizer.eos_token_id  

        return input_encodings


    dataloader = None
    if(tinystories):
        if(split == 'train'):
            data_loader = DataLoader(
            train_dataset,
            # generator=generator,
            batch_size=batch_size,
             
            sampler=DistributedSampler(train_dataset, shuffle=True),
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
    elif(fw):
        if(split == 'train'):
            data_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            
            
            sampler=DistributedSampler(train_dataset, shuffle=True),
            collate_fn=collate_fn,
            drop_last=True,
            shuffle=False
    )
        elif(split == 'val'):
            data_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
                # generator=generator,
            sampler=DistributedSampler(val_dataset, shuffle=True),
            collate_fn=collate_fn,
              
            drop_last=True,
            shuffle=False
        )
    return data_loader



def prepare_dataset_dpo(split, device, batch_size):


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a dataset for training")
    parser.add_argument("--tinystories", action="store_true", help="Use TinyStories dataset")
    parser.add_argument("--fw", action="store_true", help="Use FineWeb dataset")
    parser.add_argument("--dpo", action="store_true", help="Use DPO dataset")

    args = parser.parse_args()
    
    train_dataset, val_dataset = load_selected_dataset(args.tinystories, args.fw, args.dpo)
