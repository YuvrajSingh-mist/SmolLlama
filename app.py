import gradio as gr
from transformers import pipeline
from config import ModelArgs
from inference import remove_prefix
from model import Llama
import torch
from inference import topk_sampling
import os
import subprocess

model_path = "weights/DPO/dpo_model.pt"

# Check if the model exists; if not, download it
if not os.path.exists(model_path):
    print("Model not found! Downloading...")
    subprocess.run(["python", "download_model_weight.py", "--dpo"], check=True)
else:
    print("Model found, skipping download.")

# Load the model
model = Llama(device=ModelArgs.device, embeddings_dims=ModelArgs.embeddings_dims, no_of_decoder_layers=ModelArgs.no_of_decoder_layers, block_size=ModelArgs.block_size, vocab_size=ModelArgs.vocab_size, dropout=ModelArgs.dropout)
model = model.to(ModelArgs.device)

dict_model = torch.load(model_path)
dict_model['MODEL_STATE'] = remove_prefix(dict_model['MODEL_STATE'], '_orig_mod.')
model.load_state_dict(dict_model['MODEL_STATE'])
model.eval()


def answer_question(prompt, temperature, top_k, max_length):
    with torch.no_grad():
        generated_text = topk_sampling(model, prompt, max_length=max_length, top_k=top_k, temperature=temperature, device=ModelArgs.device)
        # print("Gnerated: ", generated_text)
        return generated_text
# Create Gradio interface
iface = gr.Interface(
    fn=answer_question,
    inputs = [
        gr.Textbox(label="Prompt", lines=5), 
        gr.Slider(minimum=0.1, maximum=1.0, value=0.8, step=0.1, label="Temperature"),
        gr.Slider(minimum=50, value=50, step=1, label="Top-k"),
        gr.Slider(minimum=10, maximum=ModelArgs.block_size, value=256, step=1, label="Max Length")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="SmolLlama",
    description="Enter a prompt, and the model will try to answer it."
)

# Launch the Gradio app
if __name__ == "__main__":
    iface.launch()
