#!/usr/bin/env python3
"""
A diagnostic script to check the compatibility of action head weights.
"""
import torch
import sys
import os
import json
from datetime import datetime

# --- VLA Integration ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'TinyVLA')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'TinyVLA', 'llava_pythia')))

from llava_pythia.model.language_model.pythia.llava_pythia import LlavaPythiaForCausalLM, LlavaPythiaConfig

def load_checkpoint_weights(checkpoint_path):
    """Load weights from various checkpoint formats."""
    try:
        if checkpoint_path.endswith('.safetensors'):
            # Load SafeTensors format
            from safetensors.torch import load_file
            weights = load_file(checkpoint_path)
            print(f"✅ Loaded SafeTensors checkpoint")
        elif checkpoint_path.endswith('.bin') or checkpoint_path.endswith('.pth'):
            # Load PyTorch binary format
            weights = torch.load(checkpoint_path, map_location='cpu')
            print(f"✅ Loaded PyTorch checkpoint")
            
            # Handle training_state.pt which contains nested structure
            if 'model' in weights:
                print(f"📦 Extracting model weights from training state")
                weights = weights['model']
            elif 'state_dict' in weights:
                print(f"📦 Extracting state dict from training state")
                weights = weights['state_dict']
        else:
            print(f"❌ Unsupported checkpoint format: {checkpoint_path}")
            return None
        return weights
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None

def analyze_diffusion_weights(checkpoint_path, model_diffusion_head):
    """Analyze the health and compatibility of diffusion head weights."""
    print(f"\n🔍 Analyzing diffusion weights from: {os.path.basename(checkpoint_path)}")
    
    weights = load_checkpoint_weights(checkpoint_path)
    if weights is None:
        return False
        
    # Filter diffusion head weights
    diffusion_keys = [k for k in weights.keys() if 'embed_out' in k or 'diffusion' in k]
    
    if not diffusion_keys:
        print("❌ No diffusion head weights found in checkpoint!")
        return False
        
    print(f"📋 Found {len(diffusion_keys)} diffusion-related parameters")
    
    # Get model's diffusion head structure for comparison
    model_keys = set(model_diffusion_head.state_dict().keys())
    ckpt_diffusion_keys = set([k.replace('embed_out.', '') for k in diffusion_keys if 'embed_out.' in k])
    
    # Key compatibility analysis
    matching_keys = model_keys & ckpt_diffusion_keys
    missing_in_ckpt = model_keys - ckpt_diffusion_keys
    extra_in_ckpt = ckpt_diffusion_keys - model_keys
    
    print(f"🔗 Key compatibility:")
    print(f"  ✅ Matching keys: {len(matching_keys)}/{len(model_keys)} ({len(matching_keys)/len(model_keys)*100:.1f}%)")
    if missing_in_ckpt:
        print(f"  ⚠️ Missing in checkpoint: {len(missing_in_ckpt)} keys")
        print(f"     Examples: {list(missing_in_ckpt)[:3]}")
    if extra_in_ckpt:
        print(f"  ℹ️ Extra in checkpoint: {len(extra_in_ckpt)} keys")
        
    # Weight health analysis
    total_params = 0
    nan_params = 0
    inf_params = 0
    extreme_params = 0
    weight_stats = {}
    
    print(f"\n🏥 Weight health analysis:")
    for key in diffusion_keys:
        param = weights[key]
        if not isinstance(param, torch.Tensor):
            continue
            
        # Count issues
        param_nan = torch.isnan(param).sum().item()
        param_inf = torch.isinf(param).sum().item()
        param_extreme = (torch.abs(param) > 100).sum().item()
        
        total_params += param.numel()
        nan_params += param_nan
        inf_params += param_inf
        extreme_params += param_extreme
        
        # Store statistics
        weight_stats[key] = {
            'shape': list(param.shape),
            'min': float(param.min().item()),
            'max': float(param.max().item()),
            'mean': float(param.mean().item()),
            'std': float(param.std().item()),
            'nan_count': param_nan,
            'inf_count': param_inf,
            'extreme_count': param_extreme
        }
        
    # Overall health report
    print(f"  📊 Total parameters: {total_params:,}")
    print(f"  🚨 NaN parameters: {nan_params} ({nan_params/total_params*100:.3f}%)")
    print(f"  ♾️ Inf parameters: {inf_params} ({inf_params/total_params*100:.3f}%)")
    print(f"  ⚡ Extreme parameters (>100): {extreme_params} ({extreme_params/total_params*100:.3f}%)")
    
    # Determine health status
    if nan_params > 0:
        print(f"  💀 STATUS: CRITICAL - Contains NaN values!")
        return False
    elif inf_params > 0:
        print(f"  💀 STATUS: CRITICAL - Contains Inf values!")
        return False
    elif extreme_params > total_params * 0.1:
        print(f"  ⚠️ STATUS: UNSTABLE - Too many extreme values!")
        return False
    elif extreme_params > total_params * 0.01:
        print(f"  ⚠️ STATUS: WARNING - Some extreme values present")
        return True
    else:
        print(f"  ✅ STATUS: HEALTHY - All weights in normal range")
        return True

def find_checkpoint_files(base_dir):
    """Find all checkpoint files in step-based directory structure."""
    checkpoint_files = []
    
    if not os.path.exists(base_dir):
        return checkpoint_files
        
    # Look for step directories
    for item in os.listdir(base_dir):
        if item.startswith('step_'):
            step_dir = os.path.join(base_dir, item)
            if os.path.isdir(step_dir):
                # Look for checkpoint files in step directory - prioritize diffusion head, then adapter models
                step_files = os.listdir(step_dir)
                if 'diffusion_head.bin' in step_files:
                    checkpoint_files.append(os.path.join(step_dir, 'diffusion_head.bin'))
                elif 'adapter_model.safetensors' in step_files:
                    checkpoint_files.append(os.path.join(step_dir, 'adapter_model.safetensors'))
                elif 'adapter_model.bin' in step_files:
                    checkpoint_files.append(os.path.join(step_dir, 'adapter_model.bin'))
                elif 'training_state.pt' in step_files:
                    checkpoint_files.append(os.path.join(step_dir, 'training_state.pt'))
    
    return sorted(checkpoint_files)

def check_weights(policy_config):
    model_path = policy_config["model_path"]
    action_head_path = policy_config.get("action_head_path", None)

    print("🤖 Loading VLA config...")
    try:
        config = LlavaPythiaConfig.from_pretrained(model_path, trust_remote_code=True)
        print(f"✅ Config loaded successfully from: {model_path}")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return

    print("🔧 Applying VLA config modifications for action head...")
    # Apply same configuration as in train_lora.py
    config.action_head_type = 'droid_diffusion'
    config.action_dim = 4
    config.state_dim = 7
    config.chunk_size = 20
    config.concat = 'token_cat'
    config.mm_use_im_start_end = True

    print("🧠 Loading VLM with modified config...")
    try:
        # Use from_pretrained method like in train_lora.py
        policy = LlavaPythiaForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.float32,
            device_map="cpu",  # Keep on CPU for checking
            trust_remote_code=True,
        )
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Check if action head (diffusion head) is properly initialized
    if hasattr(policy, 'embed_out') and policy.embed_out is not None:
        print(f"✅ Action head (embed_out) found: {type(policy.embed_out)}")
        
        # Get action head state dict
        action_head_state = policy.embed_out.state_dict()
        print(f"📋 Action head has {len(action_head_state)} parameter keys")
        
        # Show some key parameters
        for name, param in list(action_head_state.items())[:5]:
            print(f"  🔹 {name}: {param.shape} ({param.dtype})")
        if len(action_head_state) > 5:
            print(f"  ... and {len(action_head_state) - 5} more parameters")
            
    else:
        print("❌ No action head (embed_out) found in model!")
        return

    # Check for existing checkpoints if path provided
    if action_head_path and os.path.exists(action_head_path):
        analyze_diffusion_weights(action_head_path, policy.embed_out)

    # Check all training checkpoints
    print("\n🔍 Checking all training checkpoints for diffusion head weights...")
    checkpoint_dirs = [
        ("VLA_weights/full_training_adapter", "Full Parameter Training"),
        ("VLA_weights/lora_adapter", "LoRA Adapter Training"),
        ("VLA_weights/lora_adapter_optimized", "Optimized LoRA Adapter"),
        ("VLA_weights/full_training_bs1", "Full Training BS=1"),
        ("VLA/diff_head", "Dedicated Diffusion Head Weights")
    ]
    
    healthy_checkpoints = []
    
    for ckpt_dir, description in checkpoint_dirs:
        print(f"\n📂 {description}: {ckpt_dir}")
        
        # Special handling for VLA/diff_head directory
        if ckpt_dir == "VLA/diff_head":
            if os.path.exists(ckpt_dir):
                diff_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.bin')]
                if diff_files:
                    print(f"   Found {len(diff_files)} diffusion head files")
                    # Check the latest file
                    latest_file = os.path.join(ckpt_dir, "diffusion_head_latest.bin")
                    if os.path.exists(latest_file):
                        print(f"\n   🔍 Checking latest: diffusion_head_latest.bin")
                        is_healthy = analyze_diffusion_weights(latest_file, policy.embed_out)
                        if is_healthy:
                            healthy_checkpoints.append(latest_file)
                    else:
                        # Check the most recent numbered file
                        numbered_files = [f for f in diff_files if 'step_' in f]
                        if numbered_files:
                            latest_numbered = sorted(numbered_files)[-1]
                            latest_path = os.path.join(ckpt_dir, latest_numbered)
                            print(f"\n   🔍 Checking latest: {latest_numbered}")
                            is_healthy = analyze_diffusion_weights(latest_path, policy.embed_out)
                            if is_healthy:
                                healthy_checkpoints.append(latest_path)
                else:
                    print(f"   ❌ No diffusion head files found")
            else:
                print(f"   ❌ Directory doesn't exist yet")
            continue
        
        checkpoint_files = find_checkpoint_files(ckpt_dir)
        
        if not checkpoint_files:
            print(f"   ❌ No checkpoint files found")
            continue
            
        print(f"   Found {len(checkpoint_files)} checkpoint files")
        
        # Check latest checkpoint (usually highest step number)
        latest_checkpoint = checkpoint_files[-1]
        print(f"\n   🔍 Checking latest: {os.path.basename(os.path.dirname(latest_checkpoint))}/{os.path.basename(latest_checkpoint)}")
        is_healthy = analyze_diffusion_weights(latest_checkpoint, policy.embed_out)
        if is_healthy:
            healthy_checkpoints.append(latest_checkpoint)
    
    # Summary
    print(f"\n" + "="*60)
    print(f"📊 SUMMARY:")
    print(f"✅ Healthy checkpoints: {len(healthy_checkpoints)}")
    for ckpt in healthy_checkpoints:
        step_dir = os.path.basename(os.path.dirname(ckpt))
        file_name = os.path.basename(ckpt)
        ckpt_type = os.path.basename(os.path.dirname(os.path.dirname(ckpt)))
        print(f"   ✓ {ckpt_type}/{step_dir}/{file_name}")
    
    if len(healthy_checkpoints) == 0:
        print(f"🚨 WARNING: No healthy diffusion head checkpoints found!")
        print(f"   This suggests training may have failed or produced corrupted weights.")
    else:
        print(f"🎉 SUCCESS: Found {len(healthy_checkpoints)} usable diffusion head checkpoints!")

if __name__ == "__main__":
    policy_config = {
        'model_path': 'VLA_weights/Llava-Pythia-400M',
        'action_head_path': None,  # Will check all training checkpoints
    }
    check_weights(policy_config) 