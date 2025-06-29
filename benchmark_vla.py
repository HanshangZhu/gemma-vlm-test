#!/usr/bin/env python3
"""
Benchmark script for TinyVLA on MetaWorld v3 Pick-Place.
"""
import numpy as np
import torch
import metaworld
import time
import sys
import os
import pickle
from PIL import Image

# --- VLA Integration ---
# Add TinyVLA paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'TinyVLA')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'TinyVLA', 'llava-pythia')))

# Imports from our working test script
from transformers import AutoTokenizer, CLIPImageProcessor
from llava_pythia.conversation import conv_templates
from llava_pythia.mm_utils import tokenizer_image_token, get_model_name_from_path
from llava_pythia.model import *
from llava_pythia.model.language_model.pythia.llava_pythia import LlavaPythiaForCausalLM, LlavaPythiaConfig

class VLA_Policy:
    """
    A self-contained policy class to handle VLA model loading,
    pre-processing, and inference.
    """
    def __init__(self, policy_config):
        self.load_policy(policy_config)
        self.load_stats(policy_config["stats_path"])

    def load_stats(self, stats_path):
        print(f"📊 Loading normalization stats from: {stats_path}")
        with open(stats_path, 'rb') as f:
            self.norm_stats = pickle.load(f)
        print("--- Loaded Normalization Stats ---")
        for key in self.norm_stats.keys():
            print(f"  - Found key: {key}")
        print("---------------------------------")

    def load_policy(self, policy_config):
        model_path = policy_config["model_path"]
        action_head_path = policy_config["action_head_path"]

        # 1. Load config and tokenizer
        print("🤖 Loading VLA config and tokenizer...")
        config = LlavaPythiaConfig.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

        # 2. Modify config before loading the model
        print("🔧 Applying VLA config modifications...")
        config.action_head_type = 'droid_diffusion'
        config.action_dim = 4 # CORRECTED: From checkpoint error; was 7
        config.state_dim = 7 # CORRECTED: From checkpoint error; was 4
        config.chunk_size = 20
        config.concat = 'token_cat'
        config.mm_use_im_start_end = True

        # 3. Load VLM with modified config
        print("🧠 Loading VLM with modified config...")
        self.policy = LlavaPythiaForCausalLM.from_pretrained(
            model_path, config=config, use_safetensors=True, torch_dtype=torch.float32
        ).to("cuda")

        # 4. Load Action Head Checkpoint
        if os.path.exists(action_head_path):
            print(f"💪 Loading action head from: {action_head_path}")
            action_head_weights = torch.load(action_head_path, map_location="cuda")
            
            # --- DEBUG: Key comparison ---
            model_keys = self.policy.embed_out.state_dict().keys()
            ckpt_keys = action_head_weights.keys()
            
            matching_keys = ckpt_keys & model_keys
            missing_in_ckpt = model_keys - ckpt_keys
            missing_in_model = ckpt_keys - model_keys

            print("--- Action Head Weight Analysis ---")
            print(f"  Keys in checkpoint: {len(ckpt_keys)}")
            print(f"  Keys in model layer: {len(model_keys)}")
            print(f"  Matching keys: {len(matching_keys)}")
            if len(missing_in_ckpt) > 0:
                print(f"  ⚠️ Keys in model but NOT in ckpt (will be random): {len(missing_in_ckpt)}")
            if len(missing_in_model) > 0:
                print(f"  ⚠️ Keys in ckpt but NOT in model (will be ignored): {len(missing_in_model)}")
            print("------------------------------------")
            
            self.policy.embed_out.load_state_dict(action_head_weights, strict=False)
            print("✅ Action head loaded successfully.")
        else:
            print(f"⚠️ WARNING: Action head checkpoint not found at {action_head_path}. Using random weights.")

        # 5. Load image processor and add special tokens
        self.image_processor = CLIPImageProcessor.from_pretrained(model_path)
        from llava_pythia.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        self.context_len = getattr(self.policy.config, "max_sequence_length", 2048)
        
        self.policy.eval()
        self.device = self.policy.device
        self.dtype = self.policy.dtype
        print(f"✅ VLA Policy ready on device: {self.device}")

    def get_action(self, image_obs, state_obs, instruction):
        """
        Get an action from the VLA model.
        """
        with torch.no_grad():
            # 1. Pre-process inputs
            # Normalize the state observation using our generated stats
            norm_state_obs = (state_obs - self.norm_stats['qpos_mean']) / self.norm_stats['qpos_std']
            
            # The model expects two images, so we duplicate the single observation
            image_tensor = self.image_processor(image_obs, return_tensors='pt')['pixel_values']
            image_tensor = image_tensor.to(self.device)

            # State tensor
            state_tensor = torch.tensor(norm_state_obs, device=self.device).unsqueeze(0)

            # Language instruction
            from llava_pythia.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
            conv = conv_templates['pythia'].copy()
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + instruction
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt() + " <|endoftext|>"
            input_ids = tokenizer_image_token(prompt, self.tokenizer, -200, return_tensors='pt').unsqueeze(0).cuda()
            
            data_dict = dict(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                images=image_tensor, # The model arch expects separate 'images' and 'images_r'
                images_r=image_tensor,
                states=state_tensor
            )
            
            # 2. Run two-step inference
            vlm_output = self.policy.forward(**data_dict, output_hidden_states=True, eval=True)
            global_cond = vlm_output.hidden_states[-1][:, -1]

            noisy_actions = torch.randn(
                (1, self.policy.config.chunk_size, self.policy.config.action_dim),
                device=self.device
            )
            timestep = torch.tensor([0], device=self.device)
            
            actions = self.policy.embed_out(noisy_actions, timestep, global_cond=global_cond)

            # 3. Post-process action
            # Un-normalize the action from [-1, 1] range to the environment's action range
            actions = actions.cpu().numpy()
            action_min = self.norm_stats['action_min']
            action_max = self.norm_stats['action_max']
            unnorm_actions = (actions + 1) / 2 * (action_max - action_min) + action_min

            # For metaworld, we only need the first 4 dims (x,y,z,gripper) of the first action
            action_to_take = unnorm_actions[0, 0, :4]
            return action_to_take

# --- MetaWorld Environment Setup ---
def run_benchmark(policy):
    print("🤖 Loading MetaWorld v2 Pick-Place Task...")
    ml1 = metaworld.ML1('pick-place-v2', seed=42)
    env = ml1.train_classes['pick-place-v2']()
    task = list(ml1.train_tasks)[0]
    env.set_task(task)
    print("✅ MetaWorld Ready")

    # --- Main Benchmark Loop ---
    success_rate = 0
    for i in range(10): # Run 10 episodes
        print(f"\n🎬 Episode {i + 1}/10")
        obs = env.reset()
        print(f" Initial observation shape: {obs.shape}")
        print(f" Initial observation (first 10 elements): {obs[:10]}")
        done = False
        step = 0
        
        # Reset instruction for each episode
        instruction = "pick up the red block and place it on the green target"
        
        while not done and step < 500: # Max 500 steps per episode
            img_array = env.render(offscreen=True)
            image = Image.fromarray(img_array)
            
            # --- Correct State Vector Construction ---
            # The observation is 39-dim, but the model expects 7-dim.
            # We use the first 7 dimensions, consistent with our stat calculation.
            state_from_env = obs[:7]

            action = policy.get_action(image, state_from_env, instruction)
            
            obs, reward, done, info = env.step(action)
            success = info['success'] > 0
            step += 1

            if step % 50 == 0:
                print(f"Step {step:3d}: action={action}, reward={reward:6.3f}, success={success:.1f}")

            if success:
                print(f"🎉 SUCCESS! Task completed in {step+1} steps!")
                break
        
        success_rate += success
        print(f"Episode finished. Success: {success}. Total Reward: {reward:.3f}")

    print("\n--- Benchmark Finished ---")
    print(f"Success rate over 10 episodes: {success_rate * 100 / 10:.1f}%")
    env.close()

if __name__ == "__main__":
    try:
        policy_config = {
            'model_path': 'VLM_weights/Llava-Pythia-400M',
            'action_head_path': 'checkpoints/diff_head_raw_final.pth',
            'stats_path': 'metaworld_stats.pkl', # Use our new stats file
        }
        vla_policy = VLA_Policy(policy_config)
        run_benchmark(vla_policy)
        print("\n👋 Benchmark completed!")
    except Exception as e:
        print(f"\n❌ An error occurred during benchmark: {e}")
        import traceback
        traceback.print_exc()
        print("💡 Make sure you have the 'tinyvla' conda environment activated.") 