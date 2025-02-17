from config import ModelArgs
from model import Llama
import torch
import torch.nn.functional as F
from tokenizer import Tokenizer
import argparse


tokenizer = Tokenizer()
tokenizer = tokenizer.ready_tokenizer()



def improved_greedy_decode_with_temperature(
    model, 
    tokenizer, 
    prompt, 
    max_length=50, 
    repetition_penalty=1.2, 
    context_window=10, 
    temperature=1.0, 
    eos_token_id=None
):
    """
    Improved greedy decoding with:
    - Repetition penalty
    - Temperature scaling
    - Contextual coherence
    - Early stopping
    """
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)['input_ids']
    generated_tokens = []
    eos_token_id = eos_token_id or tokenizer.eos_token_id  # Use EOS token if provided

    for _ in range(max_length):
        outputs = model(input_ids)
        logits = outputs[:, -1, :]  # Get logits for the last token

        # Apply temperature scaling
        if temperature != 1.0:
            logits = logits / temperature

        # Apply repetition penalty
        if repetition_penalty != 1.0 and len(generated_tokens) > 0:
            for token in set(generated_tokens[-context_window:]):  # Penalize recent tokens
                logits[0, token] /= repetition_penalty

        # Greedy selection
        next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
        generated_tokens.append(next_token.item())

        # Stop if EOS token is generated
        if next_token.item() == eos_token_id:
            break

        # Append the new token to the input
        input_ids = torch.cat([input_ids, next_token], dim=1)

    # Decode the generated tokens
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_length", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    args = parser.parse_args()
    
    model = Llama(device=ModelArgs.device, embeddings_dims=ModelArgs.embeddings_dims, no_of_decoder_layers=ModelArgs.no_of_decoder_layers, block_size=ModelArgs.block_size, vocab_size=ModelArgs.vocab_size, dropout=ModelArgs.dropout)
    model = model.to(ModelArgs.device)

    dict_model = torch.load('snapshot2.pt')
    model.load_state_dict(dict_model['MODEL_STATE'])

    # print("Model ready")
    # prompt = 'Its a secret'


    generated_text = improved_greedy_decode_with_temperature(
            model, 
            tokenizer, 
            args.prompt, 
            args.max_length, 
            args.repetition_penalty, 
            temperature=args.temperature,
            context_window=args.max_length // 2,
            
        )

    print(args.prompt + generated_text)


if __name__ == '__main__':
    main()