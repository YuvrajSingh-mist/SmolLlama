import gradio as gr
from transformers import pipeline
from config import ModelArgs
from inference import remove_prefix
from model import Llama
import torch
from inference_sft import topk_sampling
import os
import subprocess

# Define model paths
model_paths = {
    "SFT": "weights/fine_tuned/fine_tuned_model.pt",
    "DPO": "weights/DPO/dpo_model.pt"
}

for i in model_paths:

    subprocess.run(["python", "download_model_weight.py", f"--{i.lower()}"])

# Function to load the selected model
def load_model(model_type):
    model_path = model_paths[model_type]

    # Check if the model exists; if not, download it
    if not os.path.exists(model_path):
        print(f"{model_type} Model not found! Downloading...")
        subprocess.run(["python", "download_model_weight.py", f"--{model_type.lower()}"], check=True)
    else:
        print(f"{model_type} Model found, skipping download.")

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

    dict_model = torch.load(model_path)
    dict_model['MODEL_STATE'] = remove_prefix(dict_model['MODEL_STATE'], '_orig_mod.')
    model.load_state_dict(dict_model['MODEL_STATE'])
    model.eval()

    return model

current_model = load_model("SFT")

def answer_question(model_type, prompt, temperature, top_k, max_length):
    global current_model

    # Reload model if the selected type is different
    if model_paths[model_type] != model_paths.get(current_model, None):
        current_model = load_model(model_type)

    formatted_prompt = f'''
    ### Instruction: You are a helpful AI Assistant for question answering.
    ### Input : {prompt}

    ### Response: 
    '''
    
    with torch.no_grad():
        generated_text = topk_sampling(current_model, formatted_prompt, max_length=max_length, top_k=top_k, temperature=temperature, device=ModelArgs.device)
        return generated_text

iface = gr.Interface(
    fn=answer_question,
    inputs=[
        gr.Dropdown(choices=["SFT", "DPO"], value="SFT", label="Select Model"),
        gr.Textbox(label="Prompt", lines=5), 
        gr.Slider(minimum=0.1, maximum=1.0, value=0.8, step=0.1, label="Temperature"),
        gr.Slider(minimum=50, value=50, step=1, label="Top-k"),
        gr.Slider(minimum=10, maximum=ModelArgs.block_size, value=256, step=1, label="Max Length")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="SmolLlama",
    description="Enter a prompt, select a model (SFT or DPO), and the model will generate an answer."
)

# Launch the Gradio app
if __name__ == "__main__":
    iface.launch()
