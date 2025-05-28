import torch
from PIL import Image
from transformers import (
    PaliGemmaForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)

model_name = "google/paligemma-3b-pt-224"

# Load processor
processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

# Use bitsandbytes 4-bit quantization
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Load model with quantization
model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.float16
)

# Ensure model is in evaluation mode
model.eval()

# Load and resize image
image_path = "/home/hz/Pictures/Screenshots/Screenshot from 2025-05-28 01-25-51.png"
image = Image.open(image_path).convert("RGB")
img_resized = image.resize((224, 224))  # Optional for visualization
img_resized.show()  # Display the image (optional)

################################################################################
# Prepare multimodal prompt with <image> token
prompt = '''<image> Produce a chain of thought reasoning for the following image and answer the question:
Given your end effector can hold the eraser, and you can move it in any direction.
Given the image, if you are a robotic arm, how would you erase the whiteboard?
identify the steps you would take to accomplish this task, considering the tools and actions available to you.

'''
######################################################################

# Preprocess inputs
inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
print(inputs.keys) # check input keys

# Generate output
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=2000, temperature=0)

# Decode result
caption = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
print("🖼️ Caption:", caption)
