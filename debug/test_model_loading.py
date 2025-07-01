#!/usr/bin/env python3
"""
🧪 Test Model Loading with Diffusion Head Weights

This script tests that the TinyVLA model loads correctly with trained diffusion head weights.
"""

import os
import torch
import numpy as np
from PIL import Image
from tinyvla_loader import load_tinyvla

def test_model_loading():
    """Test loading the TinyVLA model with diffusion head weights."""
    
    print("🧪 Testing TinyVLA Model Loading with Diffusion Head")
    print("=" * 60)
    
    # Check if required files exist
    print("🔍 Checking required files...")
    
    required_files = [
        "VLA_weights/Llava-Pythia-400M",
        "VLA/diff_head/diffusion_head_latest.bin",
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - NOT FOUND")
    
    # Check for checkpoint directories
    checkpoint_dirs = [
        "VLA_weights/full_training_adapter/step_10000",
        "VLA_weights/full_training_adapter/step_9500", 
        "VLA_weights/full_training_adapter/step_9000",
        "VLA_weights/full_training_adapter/step_500",
    ]
    
    available_checkpoint = None
    for checkpoint_dir in checkpoint_dirs:
        if os.path.exists(checkpoint_dir):
            print(f"   ✅ {checkpoint_dir}")
            if available_checkpoint is None:
                available_checkpoint = checkpoint_dir
        else:
            print(f"   ❌ {checkpoint_dir} - NOT FOUND")
    
    if available_checkpoint is None:
        print("❌ No training checkpoints found!")
        return False
    
    print(f"🎯 Using checkpoint: {available_checkpoint}")
    
    # Attempt to load the model
    print("\n🚀 Loading TinyVLA model...")
    try:
        vla_model = load_tinyvla(
            base_model_path="VLA_weights/Llava-Pythia-400M",
            lora_checkpoint_path=available_checkpoint,
            diffusion_weights_path="VLA/diff_head/diffusion_head_latest.bin",
            stats_path="VLA_weights/full_training_adapter/stats.pkl"
        )
        print("✅ Model loaded successfully!")
        
        # Test model inference
        print("\n🔬 Testing model inference...")
        
        # Create dummy inputs
        dummy_image = Image.new('RGB', (336, 336), color='red')
        dummy_state = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])  # 7D state
        dummy_instruction = "pick up the red block"
        
        # Test prediction
        action = vla_model.predict_action(dummy_image, dummy_state, dummy_instruction)
        
        print(f"   ✅ Prediction successful!")
        print(f"   📊 Action shape: {action.shape}")
        print(f"   📊 Action values: {action}")
        
        # Check for NaN/Inf
        if np.isnan(action).any() or np.isinf(action).any():
            print("   ⚠️ Warning: Action contains NaN/Inf values!")
            return False
        else:
            print("   ✅ Action values are healthy (no NaN/Inf)")
        
        # Check action range
        if np.abs(action).max() > 10.0:
            print(f"   ⚠️ Warning: Action values seem extreme (max: {np.abs(action).max():.3f})")
        else:
            print(f"   ✅ Action values in reasonable range (max: {np.abs(action).max():.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("\n🎉 All tests passed! Model loading works correctly.")
    else:
        print("\n💥 Tests failed! Check the errors above.") 