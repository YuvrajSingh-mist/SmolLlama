
from transformers import AutoTokenizer
import os


class Tokenizer:
    
    def __init__(self) -> None:
        
        self.tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", hf_token = '...')

        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def ready_tokenizer(self):
        
        return self.tokenizer
    
    



