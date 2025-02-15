from transformers import AutoTokenizer, AutoModelForCausalLM


class Tokenizer:
    
    def __init__(self) -> None:
        
        self.tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", hf_token = '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')

        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def ready_tokenizer(self):
        
        return self.tokenizer
    
    



