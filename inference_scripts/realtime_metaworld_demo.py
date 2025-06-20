#!/usr/bin/env python3
"""
Real-time MetaWorld Demo with TinyVLA Direct Diffusion
This script demonstrates real-time robot control using the trained TinyVLA model
by directly calling the diffusion head, bypassing the routing issues.
"""

import os
import sys
import argparse
import numpy as np
import torch
import cv2
from PIL import Image
import time
import random
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import queue

# Add TinyVLA to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TinyVLA'))

from llava_pythia.conversation import conv_templates
from llava_pythia.model.builder import load_pretrained_model
from llava_pythia.mm_utils import tokenizer_image_token, get_model_name_from_path
from llava_pythia.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava_pythia.model import *
from data_utils.datasets import set_seed

# MetaWorld imports
import metaworld
import logging

# Set rendering mode
if os.environ.get('DISPLAY') is None:
    os.environ['MUJOCO_GL'] = 'osmesa'
    print("[INFO] No display found. Using OSMesa for headless rendering.")
else:
    os.environ['MUJOCO_GL'] = 'glfw'
    print("[INFO] Display found. Using GLFW for rendering.")

class RealTimeVLAController:
    """Real-time controller using direct diffusion approach"""
    
    def __init__(self, model_path, checkpoint_path=None, device='cuda'):
        self.device = device
        self.load_model(model_path, checkpoint_path)
        self.action_queue = queue.Queue(maxsize=20)
        self.temporal_actions = None
        self.last_query_time = 0
        self.query_frequency = 1  # Query every frame for responsiveness
        
    def load_model(self, model_path, checkpoint_path):
        """Load the TinyVLA model with direct diffusion access"""
        print("🤖 Loading TinyVLA model...")
        
        # Temporarily force float32 to avoid scheduler issues
        original_float16 = torch.float16
        torch.float16 = torch.float32
        
        try:
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path, None, get_model_name_from_path(model_path), False, False
            )
        finally:
            torch.float16 = original_float16
        
        # Fix noise scheduler
        if hasattr(self.model, 'noise_scheduler'):
            from diffusers.schedulers.scheduling_ddim import DDIMScheduler
            self.model.noise_scheduler = DDIMScheduler(
                num_train_timesteps=100,
                beta_schedule='squaredcos_cap_v2',
                clip_sample=True,
                set_alpha_to_one=True,
                steps_offset=0,
                prediction_type='epsilon'
            )
        
        # Set visual concat mode
        if not hasattr(self.model, 'visual_concat') or self.model.visual_concat == 'None':
            self.model.visual_concat = 'token_cat'
        
        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"📦 Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            checkpoint = {k.replace('_orig_mod.', ''): v for k, v in checkpoint.items()}
            self.model.embed_out.load_state_dict(checkpoint, strict=False)
            print("✅ Checkpoint loaded!")
        
        self.model.eval()
        
        # Load normalization stats if available
        self.stats = None
        stats_path = os.path.join(os.path.dirname(model_path), 'dataset_stats.pkl')
        if os.path.exists(stats_path):
            with open(stats_path, 'rb') as f:
                self.stats = pickle.load(f)
            print("📊 Loaded action normalization stats")
    
    def expand2square(self, pil_imgs, background_color):
        """Expand images to square for model input"""
        batch_size, channels, height, width = pil_imgs.shape
        max_dim = max(height, width)
        expanded_imgs = np.full((batch_size, max_dim, max_dim, channels), background_color, dtype=np.float32)
        
        if height == width:
            expanded_imgs = pil_imgs.permute(0,2,3,1).cpu().numpy()
        elif height > width:
            offset = (max_dim - width) // 2
            expanded_imgs[:, :height, offset:offset + width, :] = pil_imgs.permute(0,2,3,1).cpu().numpy()
        else:
            offset = (max_dim - height) // 2
            expanded_imgs[:, offset:offset + height, :width, :] = pil_imgs.permute(0,2,3,1).cpu().numpy()
        
        expanded_imgs = torch.tensor(expanded_imgs).to(dtype=pil_imgs.dtype, device=pil_imgs.device)
        return expanded_imgs
    
    def preprocess_observation(self, rgb_obs, robot_state, instruction):
        """Preprocess observation for model input"""
        # Convert image to tensor
        curr_image = torch.from_numpy(rgb_obs / 255.0).float().to(self.device)
        curr_image = curr_image.permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW
        
        # Duplicate for dual camera (MetaWorld only has single camera)
        image = self.expand2square(curr_image, tuple(x for x in self.image_processor.image_mean))
        image_tensor = self.image_processor.preprocess(
            image, return_tensors='pt', do_normalize=True, do_rescale=False, do_center_crop=False
        )['pixel_values'].to(self.device, dtype=self.model.dtype)
        
        image_tensor_r = image_tensor.clone()  # Duplicate for right camera
        
        # Process language instruction
        conv = conv_templates['pythia'].copy()
        inp = instruction
        if self.model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt() + " <|endoftext|>"
        
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        attn_mask = input_ids.ne(self.tokenizer.pad_token_id)
        
        # Robot state
        states = torch.from_numpy(robot_state).float().unsqueeze(0).to(self.device, dtype=self.model.dtype)
        
        return image_tensor, image_tensor_r, input_ids, attn_mask, states
    
    def get_action_direct(self, rgb_obs, robot_state, instruction):
        """Get action by directly calling diffusion head"""
        with torch.inference_mode():
            # Preprocess inputs
            image_tensor, image_tensor_r, input_ids, attn_mask, states = self.preprocess_observation(
                rgb_obs, robot_state, instruction
            )
            
            # Get hidden states from model
            input_ids, attention_mask, _, inputs_embeds, _ = self.model.prepare_inputs_labels_for_multimodal(
                input_ids, attn_mask, None, None, image_tensor,
                images_r=image_tensor_r, images_top=None,
                visual_concat=self.model.visual_concat, states=states
            )
            
            outputs = self.model.get_model()(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_hidden_states=True,
                return_dict=True
            )
            
            hidden_states = outputs[0]
            
            # Direct diffusion denoising
            B = 1
            Tp = self.model.num_queries
            action_dim = self.model.action_dim
            
            # Initialize from noise
            noisy_action = torch.randn((B, Tp, action_dim)).to(self.device)
            naction = noisy_action.to(dtype=hidden_states.dtype)
            
            # Denoising loop
            self.model.noise_scheduler.set_timesteps(self.model.num_inference_timesteps)
            
            for k in self.model.noise_scheduler.timesteps:
                noise_pred = self.model.embed_out(
                    naction, k,
                    global_cond=hidden_states,
                    states=states
                )
                
                naction = self.model.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample
            
            return naction

def setup_metaworld_env(task_name='pick-place-v3', render_mode='rgb_array'):
    """Setup MetaWorld environment"""
    print(f"🎮 Setting up MetaWorld environment: {task_name}")
    
    # Map v2 to v3 task names
    v2_to_v3 = {
        'pick-place-v2': 'pick-place-v3',
        'door-open-v2': 'door-open-v3',
        'drawer-open-v2': 'drawer-open-v3',
        'button-press-topdown-v2': 'button-press-topdown-v3',
        'reach-v2': 'reach-v3',
        'push-v2': 'push-v3',
    }
    
    if task_name in v2_to_v3:
        task_name = v2_to_v3[task_name]
    
    benchmark = metaworld.ML1(task_name)
    env = benchmark.train_classes[task_name]()
    task = random.choice(benchmark.train_tasks)
    env.set_task(task)
    
    if hasattr(env, 'env'):
        env = env.env
    
    env.render_mode = render_mode
    env.camera_name = 'corner'
    
    return env

def run_realtime_demo(controller, env, instruction, max_steps=200, show_plot=True):
    """Run real-time demo with visualization"""
    print("\n🚀 Starting real-time demo...")
    print(f"📝 Instruction: '{instruction}'")
    print("Press Ctrl+C to stop\n")
    
    # Reset environment
    obs, info = env.reset()
    
    # Setup visualization if requested
    if show_plot:
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.set_title("Robot Camera View")
        ax1.axis('off')
        ax2.set_title("Action Values")
        ax2.set_ylim(-1.5, 1.5)
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Action Value")
        
        action_history = {'x': [], 'y': [], 'z': [], 'gripper': []}
        time_steps = []
    
    # Temporal action aggregation
    temporal_agg = True
    chunk_size = 20
    all_time_actions = torch.zeros([max_steps, max_steps + chunk_size, 4], dtype=torch.float32).to(controller.device)
    
    total_reward = 0.0
    step = 0
    success = False
    
    try:
        while step < max_steps:
            start_time = time.time()
            
            # Get observation
            rgb = env.render()
            if rgb is None:
                rgb = np.zeros((480, 480, 3), dtype=np.uint8)
            
            robot_state = obs[:7]  # First 7 elements are typically joint positions
            
            # Get action from model
            if step % controller.query_frequency == 0:
                all_actions = controller.get_action_direct(rgb, robot_state, instruction)
                
                # Print action stats
                if step % 10 == 0:
                    print(f"Step {step}: Generated actions range [{all_actions.min():.3f}, {all_actions.max():.3f}]")
            
            # Temporal aggregation
            if temporal_agg:
                all_time_actions[[step], step:step + chunk_size] = all_actions
                actions_for_curr_step = all_time_actions[:, step]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                
                if len(actions_for_curr_step) > 0:
                    # Exponential weighting for temporal smoothing
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).to(controller.device).unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    raw_action = torch.zeros((1, 4)).to(controller.device)
            else:
                raw_action = all_actions[:, step % chunk_size]
            
            # Post-process action
            action = raw_action.squeeze(0).cpu().numpy()
            
            # Denormalize if stats available
            if controller.stats and 'action_mean' in controller.stats:
                action = action * controller.stats['action_std'] + controller.stats['action_mean']
            
            # Clip actions
            action = np.clip(action, -1.0, 1.0)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # Update visualization
            if show_plot:
                # Update camera view
                ax1.clear()
                ax1.imshow(rgb)
                ax1.set_title(f"Robot Camera View (Step {step})")
                ax1.axis('off')
                
                # Update action plot
                time_steps.append(step)
                action_history['x'].append(action[0])
                action_history['y'].append(action[1])
                action_history['z'].append(action[2])
                action_history['gripper'].append(action[3])
                
                ax2.clear()
                ax2.plot(time_steps, action_history['x'], 'r-', label='X', alpha=0.7)
                ax2.plot(time_steps, action_history['y'], 'g-', label='Y', alpha=0.7)
                ax2.plot(time_steps, action_history['z'], 'b-', label='Z', alpha=0.7)
                ax2.plot(time_steps, action_history['gripper'], 'm-', label='Gripper', alpha=0.7)
                ax2.set_ylim(-1.5, 1.5)
                ax2.set_xlabel("Time Step")
                ax2.set_ylabel("Action Value")
                ax2.set_title(f"Actions (Reward: {total_reward:.3f})")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.pause(0.001)
            
            # Print status
            if step % 25 == 0:
                fps = 1.0 / (time.time() - start_time)
                print(f"Step {step}: Reward={total_reward:.3f}, FPS={fps:.1f}, Action={action}")
            
            # Check success
            if info.get("success", False):
                success = True
                print(f"\n🎉 SUCCESS! Task completed at step {step}")
                break
            
            if done:
                print(f"\n Episode ended at step {step}")
                break
            
            step += 1
            
    except KeyboardInterrupt:
        print("\n⏹️  Demo stopped by user")
    
    # Final summary
    print("\n" + "="*50)
    print("📊 DEMO SUMMARY")
    print("="*50)
    print(f"Total Steps: {step}")
    print(f"Total Reward: {total_reward:.3f}")
    print(f"Success: {'✅ Yes' if success else '❌ No'}")
    print(f"Average FPS: {step / (time.time() - start_time):.1f}")
    
    if show_plot:
        plt.ioff()
        plt.show()
    
    return {
        'steps': step,
        'reward': total_reward,
        'success': success
    }

def main():
    parser = argparse.ArgumentParser(description="Real-time MetaWorld Demo with TinyVLA")
    parser.add_argument("--model-path", type=str, default="VLM_weights/Llava-Pythia-400M",
                       help="Path to the TinyVLA model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/TinyVLA-raw_actions_metaworld/diff_head_raw_final.pth",
                       help="Path to diffusion head checkpoint")
    parser.add_argument("--task", type=str, default="pick-place-v3",
                       help="MetaWorld task name")
    parser.add_argument("--instruction", type=str, default="Pick up the object and place it at the target.",
                       help="Language instruction for the robot")
    parser.add_argument("--max-steps", type=int, default=200,
                       help="Maximum number of steps")
    parser.add_argument("--no-plot", action="store_true",
                       help="Disable real-time plotting")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Suppress warnings
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    
    print("="*60)
    print("🤖 TinyVLA Real-Time MetaWorld Demo")
    print("="*60)
    print("This demo uses direct diffusion head access to control")
    print("the robot in real-time, bypassing routing issues.")
    print("="*60)
    
    # Initialize controller
    controller = RealTimeVLAController(
        model_path=args.model_path,
        checkpoint_path=args.checkpoint
    )
    
    # Setup environment
    env = setup_metaworld_env(args.task)
    
    # Run demo
    results = run_realtime_demo(
        controller=controller,
        env=env,
        instruction=args.instruction,
        max_steps=args.max_steps,
        show_plot=not args.no_plot
    )
    
    env.close()

if __name__ == "__main__":
    main() 