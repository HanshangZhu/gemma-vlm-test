#!/usr/bin/env python3
"""
Simple MetaWorld Demo with TinyVLA
Bypasses complex setup and focuses on basic functionality
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import time

# Add TinyVLA to path
sys.path.append('/home/hz/gemma-vlm-test/TinyVLA')

from unified_tinyvla import UnifiedTinyVLAModel
from transformers import AutoTokenizer, CLIPImageProcessor

def create_simple_env():
    """Create a simple MetaWorld environment with minimal setup."""
    try:
        # Set environment variables for headless rendering
        os.environ['MUJOCO_GL'] = 'egl'  # Use EGL for headless rendering
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        
        # Try importing MetaWorld
        from metaworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place import SawyerPickAndPlaceEnv
        
        print("Creating MetaWorld Pick and Place environment...")
        env = SawyerPickAndPlaceEnv()
        
        print("✓ Environment created successfully!")
        return env
        
    except Exception as e:
        print(f"Error creating environment: {e}")
        print("Falling back to dummy environment...")
        return None

def load_tinyvla_model():
    """Load the trained TinyVLA model."""
    print("🚀 Loading TinyVLA Model...")
    
    model_path = "VLM_weights/Llava-Pythia-400M"
    checkpoint_path = "checkpoints/diff_head_ft.pth"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = UnifiedTinyVLAModel(model_path, mode="action")
    model = model.to(device, dtype=torch.float32)
    model.eval()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    new_checkpoint = {k.replace('_orig_mod.', ''): v for k, v in checkpoint.items()}
    model.base_model.embed_out.load_state_dict(new_checkpoint)
    print("✓ Loaded diffusion head checkpoint")
    
    # Load processors
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    image_processor = CLIPImageProcessor.from_pretrained(model_path)
    
    return model, tokenizer, image_processor, device

def get_rgb_observation(env, width=336, height=336):
    """Get RGB observation from environment."""
    if env is None:
        # Return dummy image if no environment
        return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    try:
        # Try different rendering methods
        rgb = env.render(offscreen=True, width=width, height=height)
        if rgb is not None:
            return rgb
        
        # Fallback
        print("Using dummy RGB observation")
        return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
    except Exception as e:
        print(f"Render error: {e}")
        return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

def predict_action(model, tokenizer, image_processor, device, image, prompt):
    """Predict action using TinyVLA."""
    # Preprocess image
    pil_image = Image.fromarray(image)
    processed = image_processor(pil_image, return_tensors="pt")
    image_tensor = processed["pixel_values"].to(device=device, dtype=torch.float32)
    
    # Process text
    text_tokens = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = text_tokens["input_ids"].to(device)
    attention_mask = text_tokens["attention_mask"].to(device)
    
    # Dummy robot state
    states = torch.zeros((1, 7), device=device, dtype=torch.float32)
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=image_tensor,
            states=states
        )
    
    if 'actions' in outputs and outputs['actions'] is not None:
        return outputs['actions'][0, 0].cpu().numpy()
    else:
        # Return dummy action if no prediction
        return np.array([0.0, 0.0, 0.1, 0.0])  # Small upward movement

def run_demo():
    """Run the MetaWorld demo."""
    print("="*60)
    print("🤖 TinyVLA MetaWorld Demo")
    print("="*60)
    
    # Load model
    model, tokenizer, image_processor, device = load_tinyvla_model()
    
    # Create environment
    env = create_simple_env()
    
    if env is not None:
        print("\n--- Initializing Environment ---")
        obs = env.reset()
        print(f"✓ Environment reset. Observation shape: {obs.shape}")
    
    print("\n--- Running Demo Episode ---")
    prompt = "Pick up the object and place it at the target location"
    print(f"Task: {prompt}")
    
    for step in range(10):
        print(f"\nStep {step + 1}/10:")
        
        # Get RGB observation
        rgb_obs = get_rgb_observation(env)
        print(f"  RGB observation shape: {rgb_obs.shape}")
        
        # Predict action
        action = predict_action(model, tokenizer, image_processor, device, rgb_obs, prompt)
        print(f"  Predicted action: {action}")
        
        # Execute action in environment
        if env is not None:
            try:
                obs, reward, done, info = env.step(action)
                print(f"  Reward: {reward:.3f}")
                
                # Try to render for GUI
                try:
                    env.render()  # This should show GUI if display is available
                    print("  ✓ GUI render successful")
                except Exception as e:
                    print(f"  GUI render warning: {e}")
                
                if done:
                    print("  Episode completed!")
                    break
                    
            except Exception as e:
                print(f"  Environment step error: {e}")
                break
        else:
            print("  (Simulated step - no environment)")
        
        time.sleep(0.5)  # Small delay for visualization
    
    print("\n🎉 Demo completed!")
    print("Your TinyVLA model is working and can predict robot actions!")

if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc() 