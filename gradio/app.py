import gradio as gr
from transformers import pipeline
from config import ModelArgs
from inference import remove_prefix
from model import Llama
import torch
from inference_sft import topk_sampling
import os
import subprocess
import re
from tokenizer import Tokenizer
import torch.nn.functional as F
import shutil

# Define model paths
model_paths = {
    "SFT": "weights/fine_tuned/models--YuvrajSingh9886--smol-llama-finetuned/snapshots/35df7811c322dab5f8df56f6a1f61b5bcf6a7339/snapshot_fine_tuned_model_900.pt",
    "DPO": "weights/DPO/models--YuvrajSingh9886--smol-llama-dpo/snapshots/13b9f059cf4630e6079e8822ea7cb7a703d77ed8/DPO_model_1650.pt",
    "Pretrained": "weights/pretrained/models--YuvrajSingh9886--smol-llama-base/snapshots/8e50a94e3c83649e48eb02549558130c87e75a87/snapshot_6750.pt"
}



ACCESS_TOKEN = os.getenv("GDRIVE_ACCESS_TOKEN")


# def download_models():
for i in model_paths:
    subprocess.run(["python", "download_model_weight.py", "--model_type", i.lower()], check=True)

# download_models()

tk = Tokenizer()
tk = tk.ready_tokenizer()



def beam_search(model, prompt, device, max_length=50, beam_width=5, top_k=50, temperature=1.0):
    input_ids = tk.encode(prompt, return_tensors='pt').to(device)
    
    # Initialize beams with initial input repeated beam_width times
    beams = input_ids.repeat(beam_width, 1)
    beam_scores = torch.zeros(beam_width).to(device)  # Initialize scores

    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(beams)
            logits = outputs[:, -1, :]  # Get last token logits
            
            # Apply temperature scaling
            scaled_logits = logits / temperature
            
            # Calculate log probabilities
            log_probs = F.log_softmax(scaled_logits, dim=-1)
            
            # Get top k candidates for each beam
            topk_log_probs, topk_indices = torch.topk(log_probs, top_k, dim=-1)
            
            # Generate all possible candidates
            expanded_beams = beams.repeat_interleave(top_k, dim=0)
            new_tokens = topk_indices.view(-1, 1)
            candidate_beams = torch.cat([expanded_beams, new_tokens], dim=1)
            
            # Calculate new scores for all candidates
            expanded_scores = beam_scores.repeat_interleave(top_k)
            candidate_scores = expanded_scores + topk_log_probs.view(-1)
            
            # Select top beam_width candidates
            top_scores, top_indices = candidate_scores.topk(beam_width)
            beams = candidate_beams[top_indices]
            beam_scores = top_scores

    # Select best beam
    best_idx = beam_scores.argmax()
    best_sequence = beams[best_idx]
    return tk.decode(best_sequence, skip_special_tokens=True)

# Function to load the selected model
def load_model(model_type):
    model_path = model_paths[model_type]

    # Check if the model exists; if not, download it
    # if not os.path.exists(model_path):
    #     shutil.rmtree(model_path)
    #     os.mkdir(model_path)
    #     print(f"{model_type} Model not found! Downloading...")
    #     subprocess.run(["python", "download_model_weight.py", f"--{model_type.lower()}"], check=True)
    # else:
    #     print(f"{model_type} Model found, skipping download.")

    # Load the model
    model = Llama(
        device=ModelArgs.device, 
        embeddings_dims=ModelArgs.embeddings_dims, 
        no_of_decoder_layers=ModelArgs.no_of_decoder_layers, 
        block_size=ModelArgs.block_size, 
        vocab_size=ModelArgs.vocab_size, 
        dropout=ModelArgs.dropout
    )
    model = model.to(ModelArgs.device)

    dict_model = torch.load(model_path, weights_only=False)
    dict_model['MODEL_STATE'] = remove_prefix(dict_model['MODEL_STATE'], '_orig_mod.')
    model.load_state_dict(dict_model['MODEL_STATE'])
    model.eval()

    return model


# download_models()
current_model = load_model("SFT")
        
       



def clean_prompt(text):
    
    text = re.sub(r'### Instruction:\s*', '', text)
    text = re.sub(r'### Input:\s*', '', text)
    text = re.sub(r'### Response:\s*', '', text)
    return text.strip()

def answer_question(model_type, prompt, temperature, top_k, max_length):
    global current_model
    # Reload model if the selected model type is different
    if model_type == "Base (Pretrained)":
        model_type = "Pretrained"
    if model_paths[model_type] != model_paths.get(current_model, None):
        current_model = load_model(model_type)


    formatted_prompt = f"### Instruction: Answer the following query. \n\n ### Input: {prompt}.\n\n ### Response: "
    
    with torch.no_grad():
        # if decoding_method == "Beam Search":
        #     generated_text = beam_search(current_model, formatted_prompt, device=ModelArgs.device, 
        #                                    max_length=max_length, beam_width=5, top_k=top_k, temperature=temperature)
        # else:
        generated_text = topk_sampling(current_model, formatted_prompt, max_length=max_length, 
                                           top_k=top_k, temperature=temperature, device=ModelArgs.device)
        generated_text = clean_prompt(generated_text)
        return generated_text


iface = gr.Interface(
    fn=answer_question,
    inputs=[
        gr.Dropdown(choices=["SFT", "DPO", "Base (Pretrained)"], value="DPO", label="Select Model"),
        # gr.Dropdown(choices=["Top-K Sampling", "Beam Search"], value="Top-K Sampling", label="Decoding Method"),
        gr.Textbox(label="Prompt", lines=5),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.8, step=0.1, label="Temperature"),
        gr.Slider(minimum=50,maximum = ModelArgs.vocab_size, value=50, step=1, label="Top-k"),
        gr.Slider(minimum=10, maximum=ModelArgs.block_size, value=256, step=1, label="Max Length")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="SmolLlama",
    description="Enter a prompt, select a model (SFT or DPO) and the model will generate an answer."
)

# Launch the Gradio app
if __name__ == "__main__":
    iface.launch()
