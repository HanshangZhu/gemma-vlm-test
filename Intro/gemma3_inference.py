from transformers import AutoTokenizer, AutoModelForCausalLM ,GemmaTokenizer
import torch

model_id = "google/gemma-3-1b-it" # Specify the model ID,which can be found on Hugging Face's model hub

# Load tokenizer and model
tokenizer = GemmaTokenizer.from_pretrained(model_id) # tokenizer is used to convert text to tokens
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

# Define your prompt
prompt = "Describe the image of a sunset over the mountains."

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Generate output
outputs = model.generate(**inputs, max_new_tokens=50)

# Decode and print the output
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
