from config import ModelArgs
from model import Llama
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import wandb
import torch.nn.functional as F
from tokenizer import Tokenizer



tokenizer = Tokenizer()

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




model = Llama(device=ModelArgs.device, embeddings_dims=ModelArgs.embeddings_dims, no_of_decoder_layers=ModelArgs.no_of_decoder_layers, block_size=ModelArgs.block_size, vocab_size=ModelArgs.vocab_size, dropout=ModelArgs.dropout)
model = model.to(ModelArgs.device)

dict_model = torch.load('snapshot.pt')
model.load_state_dict(dict_model['MODEL_STATE'])

print("Model ready")
prompt = 'I was once'

# input_ids = tokenizer(prompt, max_length = ModelArgs.block_size, truncation=True, padding=False, return_tensors='pt')['input_ids']
# input_ids = input_ids.to(ModelArgs.device)

generated_text = greedy_decode(model, tokenizer=tokenizer, max_length=50, prompt=prompt)
print(generated_text)