import gradio as gr
import torch
from transformers import GemmaTokenizer, AutoModelForCausalLM

# Load model & tokenizer
model_id = "google/gemma-3-1b-it"
tokenizer = GemmaTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

# Inference function
def generate_caption(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=64)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Gradio interface
gr.Interface(
    fn=generate_caption,
    inputs=gr.Textbox(label="Describe the image in text"),
    outputs=gr.Textbox(label="Gemma response"),
    title="🖼️ Gemma-3B Simulated VLM",
    description="Enter a natural language description of an image. The model will respond as if it had seen the image."
).launch()
