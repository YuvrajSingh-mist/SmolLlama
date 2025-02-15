
import torch.nn.functional as F

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


from datasets import load_dataset
from torch.utils.data import DataLoader
from tokenizer import Tokenizer
from config import ModelArgs



tokenizer = Tokenizer().ready_tokenizer()

# use name="sample-10BT" to use the 10BT sample
fw_train = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=False)

fw_train = fw_train.train_test_split(test_size=0.2)



def prepare_dataset(split, batch_size):

    def collate_fn(batch):
        # Extract text data
        texts = [item["text"] for item in batch]



        # Tokenize text data
        encoding = tokenizer(texts, max_length = ModelArgs.block_size, padding='max_length', truncation=True, return_tensors="pt")
        encoding["labels"] = encoding["input_ids"].clone()  # Use `input_ids` as labels
        encoding["labels"][:, :-1] = encoding["input_ids"][:, 1:]  # Shift right
        encoding["labels"][:, -1] = tokenizer.pad_token_id    # Ignore the last token (no target for it)
        # Return tokenized input tensors
        return encoding


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
