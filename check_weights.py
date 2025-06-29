#!/usr/bin/env python3
"""
A diagnostic script to check the compatibility of action head weights.
"""
import torch
import sys
import os

# --- VLA Integration ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'TinyVLA')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'TinyVLA', 'llava-pythia')))

from llava_pythia.model.language_model.pythia.llava_pythia import LlavaPythiaForCausalLM, LlavaPythiaConfig

def check_weights(policy_config):
    model_path = policy_config["model_path"]
    action_head_path = policy_config["action_head_path"]

    print("🤖 Loading VLA config...")
    config = LlavaPythiaConfig.from_pretrained(model_path, trust_remote_code=True)

    print("🔧 Applying VLA config modifications for action head...")
    config.action_head_type = 'droid_diffusion'
    config.action_dim = 4
    config.state_dim = 7
    config.chunk_size = 20
    config.concat = 'token_cat'
    config.mm_use_im_start_end = True

    print("🧠 Loading VLM with modified config (structure only)...")
    # We use from_config to avoid loading the full weights, as we only need the structure
    policy = LlavaPythiaForCausalLM(config)

    if os.path.exists(action_head_path):
        print(f"💪 Loading action head checkpoint from: {action_head_path}")
        action_head_weights = torch.load(action_head_path, map_location="cpu")
        
        # --- DEBUG: Key comparison ---
        model_keys = policy.embed_out.state_dict().keys()
        ckpt_keys = action_head_weights.keys()
        
        matching_keys = ckpt_keys & model_keys
        missing_in_ckpt = model_keys - ckpt_keys
        missing_in_model = ckpt_keys - model_keys

        print("\n--- Action Head Weight Analysis ---")
        print(f"  Keys in checkpoint file: {len(ckpt_keys)}")
        print(f"  Keys in model's action head layer: {len(model_keys)}")
        print(f"  ✅ Matching keys: {len(matching_keys)}")
        
        if len(missing_in_ckpt) > 0:
            print(f"  ⚠️ Keys in model but NOT in ckpt (will use random init): {len(missing_in_ckpt)}")
            # print("   -> ", sorted(list(missing_in_ckpt))) # Uncomment for full list
        
        if len(missing_in_model) > 0:
            print(f"  ⚠️ Keys in ckpt but NOT in model (will be ignored): {len(missing_in_model)}")
            # print("   -> ", sorted(list(missing_in_model))) # Uncomment for full list

        if len(matching_keys) == 0 and len(ckpt_keys) > 0:
            print("\n🚨 CRITICAL: ZERO matching keys found.")
            print("This means the action head is operating with COMPLETELY RANDOM weights.")
        elif len(missing_in_ckpt) > 0:
             print("\n🔍 WARNING: Partial match. Some weights will be random.")
        else:
            print("\n🎉 SUCCESS: All action head weights seem to match!")
            
        print("------------------------------------\n")

    else:
        print(f"❌ ERROR: Action head checkpoint not found at {action_head_path}.")

if __name__ == "__main__":
    policy_config = {
        'model_path': 'VLM_weights/Llava-Pythia-400M',
        'action_head_path': 'checkpoints/diff_head_raw_final.pth',
    }
    check_weights(policy_config) 