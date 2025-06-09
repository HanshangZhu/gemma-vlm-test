# your_script.py
from unified_tinyvla import UnifiedTinyVLAModel, run_inference

# Load model
model = UnifiedTinyVLAModel(
    model_path="VLM_weights/Llava-Pythia-400M",
    mode="action"
)

# Run inference
outputs = run_inference(
    image_path="test_imgs/task_clean_whiteboard.png",
    prompt="What do you see in this image?",
    model_path="VLM_weights/Llava-Pythia-400M",
    mode="action"
)

# Process outputs
print("Chain of Thought:", outputs['chain_of_thought'])
if 'actions' in outputs:
    print("Actions shape:", outputs['actions'].shape)
    print("First action sequence:", outputs['actions'][0])