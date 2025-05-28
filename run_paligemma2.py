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
image_path = "test_imgs/task_clean_whiteboard.png"
image = Image.open(image_path).convert("RGB")
img_resized = image.resize((224, 224))  # Optional for visualization
img_resized.show()  # Display the image (optional)

################################################################################
# Prepare multimodal prompt with <image> token
prompt = """<image> Given the tools visible, reason step-by-step how to use the eraser to clean the whiteboard. 
Explain each robotic action. First answer what the eraser is, where it is located, and how to use it."""

######################################################################

# Preprocess inputs
inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
#print(inputs.keys) # check input keys

# Generate output
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=2000, temperature=0.2)

# Decode result
caption = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
print("🖼️ Caption:", caption)
