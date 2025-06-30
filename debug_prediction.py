#!/usr/bin/env python3
"""
🔍 Debug Model Prediction Issues

This script debugs the tensor dimension mismatch error in model prediction.
"""

import os
import torch
import numpy as np
from PIL import Image
from tinyvla_loader import load_tinyvla

def debug_model_prediction():
    """Debug the model prediction to find tensor dimension issues."""
    
    print("🔍 Debugging Model Prediction Issues")
    print("=" * 60)
    
    # Load the model
    print("🚀 Loading model...")
    try:
        vla_model = load_tinyvla(
            base_model_path="VLA_weights/Llava-Pythia-400M",
            lora_checkpoint_path="VLA_weights/full_training_adapter/step_10000",
            diffusion_weights_path="VLA/diff_head/diffusion_head_latest.bin",
            stats_path="VLA_weights/full_training_adapter/stats.pkl"
        )
        print("✅ Model loaded!")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return
    
    # Create test inputs
    print("\n🧪 Creating test inputs...")
    dummy_image = Image.new('RGB', (336, 336), color='red')
    dummy_state = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])  # 7D state
    dummy_instruction = "pick up the red block"
    
    print(f"   Image size: {dummy_image.size}")
    print(f"   State shape: {dummy_state.shape}")
    print(f"   Instruction: {dummy_instruction}")
    
    # Prepare inputs manually to debug each step
    print("\n🔧 Preparing model inputs...")
    try:
        inputs = vla_model._prepare_inputs(dummy_image, dummy_state, dummy_instruction)
        print("✅ Input preparation successful!")
        
        for key, value in inputs.items():
            if torch.is_tensor(value):
                print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"   {key}: {type(value)}")
                
    except Exception as e:
        print(f"❌ Input preparation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run model forward pass with detailed debugging
    print("\n🧠 Running model forward pass...")
    try:
        with torch.no_grad():
            outputs = vla_model.model(**inputs)
            print("✅ Model forward pass successful!")
            
            print(f"   Output type: {type(outputs)}")
            print(f"   Output attributes: {dir(outputs)}")
            
            # Check each possible output attribute
            if hasattr(outputs, 'actions'):
                actions = outputs.actions
                if actions is not None:
                    print(f"   actions: shape={actions.shape}, dtype={actions.dtype}")
                    print(f"   actions range: [{actions.min().item():.4f}, {actions.max().item():.4f}]")
                else:
                    print("   actions: None")
            
            if hasattr(outputs, 'prediction_logits'):
                pred_logits = outputs.prediction_logits
                if pred_logits is not None:
                    print(f"   prediction_logits: shape={pred_logits.shape}, dtype={pred_logits.dtype}")
                    print(f"   prediction_logits range: [{pred_logits.min().item():.4f}, {pred_logits.max().item():.4f}]")
                else:
                    print("   prediction_logits: None")
            
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
                if logits is not None:
                    print(f"   logits: shape={logits.shape}, dtype={logits.dtype}")
                    print(f"   logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
                else:
                    print("   logits: None")
            
            if hasattr(outputs, 'loss'):
                loss = outputs.loss
                if loss is not None:
                    print(f"   loss: {loss.item():.4f}")
                else:
                    print("   loss: None")
                    
    except Exception as e:
        print(f"❌ Model forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Debug action extraction process
    print("\n🎯 Debugging action extraction...")
    try:
        # Try to extract actions following the same logic as predict_action
        predicted_actions = None
        
        if hasattr(outputs, 'actions') and outputs.actions is not None:
            predicted_actions = outputs.actions
            print(f"   Using 'actions' output: {predicted_actions.shape}")
            
        elif hasattr(outputs, 'prediction_logits') and outputs.prediction_logits is not None:
            predicted_actions = outputs.prediction_logits
            print(f"   Using 'prediction_logits' output: {predicted_actions.shape}")
            
        elif hasattr(outputs, 'logits') and outputs.logits is not None:
            # 🔥 FIXED: Handle diffusion model outputs in logits field
            logits = outputs.logits
            print(f"   Found logits: {logits.shape}")
            
            # Check if this looks like action predictions (not language logits)
            if len(logits.shape) >= 2 and logits.shape[-1] <= 10:
                predicted_actions = logits
                print(f"   Using logits as action predictions: {predicted_actions.shape}")
            else:
                print(f"   Logits shape {logits.shape} doesn't look like actions")
                return
            
        else:
            print("   No valid action outputs found")
            return
        
        # Debug the tensor processing step by step
        print(f"   Original tensor shape: {predicted_actions.shape}")
        print(f"   Original tensor device: {predicted_actions.device}")
        print(f"   Original tensor dtype: {predicted_actions.dtype}")
        
        # Convert to CPU
        actions_cpu = predicted_actions.cpu()
        print(f"   After .cpu(): shape={actions_cpu.shape}")
        
        # Convert to numpy
        actions_np = actions_cpu.numpy()
        print(f"   After .numpy(): shape={actions_np.shape}, dtype={actions_np.dtype}")
        
        # Try different indexing strategies
        print(f"   Number of dimensions: {len(actions_np.shape)}")
        
        if len(actions_np.shape) == 3:
            print(f"   3D tensor - trying [0, 0, :] indexing...")
            try:
                action = actions_np[0, 0, :]
                print(f"   Success! Action shape: {action.shape}, values: {action}")
            except Exception as e:
                print(f"   Failed: {e}")
                print(f"   Tensor dimensions: {actions_np.shape}")
                # Try alternative indexing
                try:
                    action = actions_np[0, :, 0] if actions_np.shape[2] == 1 else actions_np[0, 0, :]
                    print(f"   Alternative indexing worked! Action: {action}")
                except Exception as e2:
                    print(f"   Alternative indexing also failed: {e2}")
                    
        elif len(actions_np.shape) == 2:
            print(f"   2D tensor - trying [0, :] indexing...")
            try:
                action = actions_np[0, :]
                print(f"   Success! Action shape: {action.shape}, values: {action}")
            except Exception as e:
                print(f"   Failed: {e}")
                
        elif len(actions_np.shape) == 1:
            print(f"   1D tensor - using directly...")
            action = actions_np
            print(f"   Action shape: {action.shape}, values: {action}")
            
        else:
            print(f"   Unexpected tensor shape: {actions_np.shape}")
            
    except Exception as e:
        print(f"❌ Action extraction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_model_prediction() 