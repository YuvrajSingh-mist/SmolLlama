from config import ModelArgs
from model import Llama
import torch
import torch.nn.functional as F
from tokenizer import Tokenizer



tokenizer = Tokenizer()
tokenizer = tokenizer.ready_tokenizer()

def beam_search_decoding(model, tokenizer, input_text, beam_width=5, max_length=50):

    # Tokenize the input text
    input_ids = tokenizer(input_text, max_length=ModelArgs.block_size, truncation=True, return_tensors="pt")# Convert input text to token IDs
    # print(input_ids)
    # input_length = input_ids.shape[1]  # Length of the input sequence
    input_ids = input_ids['input_ids'].to(ModelArgs.device) 
    # Initialize the beam with the input sequence
    sequences = [[input_ids, 0.0]]  # Each sequence is a list of token IDs and its score

    for _ in range(max_length):
        all_candidates = []
        for seq, score in sequences:
            # Get the last token in the sequence
            last_token = seq[-1]

            # Predict the next token probabilities
            with torch.no_grad():
                # print(seq)
                outputs = model(seq)
                next_token_logits = outputs[:, -1, :]  # Shape: (1, vocab_size)
                next_token_probs = torch.softmax(next_token_logits, dim=-1).squeeze()  # Convert to probabilities

            # Get the top `beam_width` tokens and their log probabilities
            top_tokens = torch.topk(next_token_probs, beam_width).indices
            top_probs = torch.log(torch.topk(next_token_probs, beam_width).values)

            # Create new candidates
            for token, prob in zip(top_tokens, top_probs):
                new_seq = seq + token
                new_score = score + prob
                all_candidates.append((new_seq, new_score))

        # Sort all candidates by score and select the top `beam_width` sequences
        ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        sequences = ordered[:beam_width]

    # Decode the best sequence back to text
    best_sequence_ids = sequences[0][0]  # Get the token IDs of the best sequence
    best_sequence_text = tokenizer.decode(best_sequence_ids, skip_special_tokens=True)

    return best_sequence_text




def greedy_decode(model, tokenizer, prompt, max_length=50):
    """
    Greedy decoding: Always selects the most likely next token.
    """
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)['input_ids']
    generated_tokens = []

    for _ in range(max_length):
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]  # Get logits for the last token
        next_token = torch.argmax(logits, dim=-1).unsqueeze(0)  # Greedy selection
        generated_tokens.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token], dim=1)  # Append the new token

    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


def temperature_sampling(logits, temperature=1.0):
    """
    Applies temperature scaling to logits before sampling.
    """
    if temperature == 0:  # Greedy decoding
        return torch.argmax(logits, dim=-1)
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token


def top_k_sampling(logits, k=50):
    """
    Top-k sampling: Samples from the top-k most likely tokens.
    """
    top_k_logits, top_k_indices = torch.topk(logits, k)
    probs = F.softmax(top_k_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return top_k_indices.gather(-1, next_token)


def top_p_sampling(logits, p=0.9):
    """
    Top-p (nucleus) sampling: Samples from the smallest set of tokens whose cumulative probability >= p.
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token


def decode_with_strategy(model, tokenizer, prompt, max_length=50, strategy="greedy", temperature=1.0, top_k=50, top_p=0.9):
    """
    Decodes text using the specified strategy:
    - "greedy": Always selects the most likely next token.
    - "temperature": Applies temperature scaling to logits.
    - "top_k": Samples from the top-k most likely tokens.
    - "top_p": Samples from the smallest set of tokens whose cumulative probability >= p.
    """
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)['input_ids']
    generated_tokens = []

    for _ in range(max_length):
        outputs = model(input_ids)
        logits = outputs[:, -1, :]  # Get logits for the last token

        if strategy == "greedy":
            next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
        elif strategy == "temperature":
            next_token = temperature_sampling(logits, temperature=temperature)
        elif strategy == "top_k":
            next_token = top_k_sampling(logits, k=top_k)
        elif strategy == "top_p":
            next_token = top_p_sampling(logits, p=top_p)
        else:
            raise ValueError(f"Unknown decoding strategy: {strategy}")

        generated_tokens.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token], dim=1)  # Append the new token

    return tokenizer.decode(generated_tokens, skip_special_tokens=True)

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


def yet_another_greedy_decode(model, tokenizer, prompt, max_length=50):
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
        
        # # If the model produces the EOS token or padding token, we stop generation.
        # if next_token.item() == tokenizer.eos_token_id:
        #     break

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

dict_model = torch.load('snapshot2.pt')
model.load_state_dict(dict_model['MODEL_STATE'])

print("Model ready")
prompt = 'Its a secret'

# input_ids = tokenizer(prompt, max_length = ModelArgs.block_size, truncation=True, padding=False, return_tensors='pt')['input_ids']
# input_ids = input_ids.to(ModelArgs.device)

# generated_text = greedy_decode(model, tokenizer=tokenizer, max_length=50, prompt=prompt)
# generated_text1 = decode_with_strategy(model, tokenizer, prompt, max_length=50, strategy="greedy", temperature=0.7, top_k=50, top_p=0.9)
generated_text = improved_greedy_decode_with_temperature(
        model, 
        tokenizer, 
        prompt, 
        max_length=50, 
        repetition_penalty=1.2, 
        context_window=10,
        temperature=0.7  # Lower temperature for more deterministic output
    )

# generated_text2 = yet_another_greedy_decode(model, tokenizer, prompt, max_length=50)
print(generated_text)
# print(generated_text2)