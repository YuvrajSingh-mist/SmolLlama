from config import ModelArgs
from model import Llama
import torch
import torch.nn.functional as F
from tokenizer import Tokenizer
import argparse


tokenizer = Tokenizer()
tokenizer = tokenizer.ready_tokenizer()


def remove_hashtag_lines(text):
    """Removes lines that contain hashtags from the given text."""
    lines = text.split("\n")
    cleaned_lines = [line for line in lines if "#" not in line]
    return "\n".join(cleaned_lines)


def remove_prefix(state_dict, prefix):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]  # Remove the prefix
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def topk_sampling(model, prompt, device, max_length=50, top_k=50, temperature=1.0):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated_tokens = []
    # ModelArgs.inference=True
    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(input_ids)
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

def main():

    torch.set_float32_matmul_precision('high')

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default=''' Follow the given instructions carefully. My mom is about to retire from her 10 long years of service to a company. write me a message saying how grateful we are for her service to our company. ''')
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    # parser.add_argument("--repetition_penalty", type=float, default=1.2)
    args = parser.parse_args()
    
    model = Llama(device=ModelArgs.device, embeddings_dims=ModelArgs.embeddings_dims, no_of_decoder_layers=ModelArgs.no_of_decoder_layers, block_size=ModelArgs.block_size, vocab_size=ModelArgs.vocab_size, dropout=ModelArgs.dropout)
    # model = torch.compile(model)
    model = model.to(ModelArgs.device)

    dict_model = torch.load('DPO_model_1650.pt')
    dict_model['MODEL_STATE'] = remove_prefix(dict_model['MODEL_STATE'], '_orig_mod.')
    model.load_state_dict(dict_model['MODEL_STATE'])
    model.eval()
    print("Model ready")
    # prompt = 'Its a secret'

    with torch.no_grad():
        generated_text = topk_sampling(model, args.prompt, max_length=args.max_length, top_k=args.top_k, temperature=args.temperature, device=ModelArgs.device)
        # generated_text = remove_hashtag_lines(generated_text)
        print("Generated: ", generated_text)
        # generated_text = beam_search(model, tokenizer, args.prompt, beam_width=5, max_length=50, temperature=1.0)
        # print(args.prompt + generated_text)


if __name__ == '__main__':
    main()
