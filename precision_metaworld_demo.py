#!/usr/bin/env python3
"""
High-Precision MetaWorld Demo with TinyVLA Diffusion Policy
Features: Longer episodes, adjustable diffusion steps, and detailed analysis
"""

import os
import sys
import argparse
import numpy as np
import torch
import time
import random
import pickle
import threading
import queue
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, List

# Add TinyVLA to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TinyVLA'))

from llava_pythia.conversation import conv_templates
from llava_pythia.model.builder import load_pretrained_model
from llava_pythia.mm_utils import tokenizer_image_token, get_model_name_from_path
from llava_pythia.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava_pythia.model import *
from data_utils.datasets import set_seed

import metaworld
import logging

# Force GLFW for windowed rendering
os.environ['MUJOCO_GL'] = 'glfw'

@dataclass
class DiffusionConfig:
    """Configuration for diffusion policy"""
    num_train_timesteps: int = 100
    num_inference_timesteps: int = 10
    beta_schedule: str = 'squaredcos_cap_v2'
    clip_sample: bool = True
    prediction_type: str = 'epsilon'
    
class PrecisionVLAController:
    """High-precision controller with detailed diffusion control"""
    
    def __init__(self, model_path: str, checkpoint_path: Optional[str] = None, 
                 device: str = 'cuda', diffusion_steps: int = 10):
        self.device = device
        self.diffusion_steps = diffusion_steps
        self.load_model(model_path, checkpoint_path)
        
        # Precision action buffer with timestamps
        self.action_buffer = deque(maxlen=50)  # Larger buffer for smoother control
        self.action_timestamps = deque(maxlen=50)
        self.inference_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=10)
        
        # Performance metrics
        self.inference_times = deque(maxlen=100)
        self.diffusion_metrics = {
            'noise_norms': [],
            'action_magnitudes': [],
            'denoising_progress': []
        }
        
        # Start inference thread
        self.running = True
        self.inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.inference_thread.start()
    
    def load_model(self, model_path: str, checkpoint_path: Optional[str]):
        """Load model with precision optimizations"""
        print(f"🎯 Loading model for high-precision control...")
        print(f"   Diffusion steps: {self.diffusion_steps}")
        
        # Force float32 for precision
        original_float16 = torch.float16
        torch.float16 = torch.float32
        
        try:
            self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
                model_path, None, get_model_name_from_path(model_path), False, False
            )
        finally:
            torch.float16 = original_float16
        
        # Configure diffusion scheduler
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
            self.model.num_inference_timesteps = self.diffusion_steps
        
        # Set visual concat
        if not hasattr(self.model, 'visual_concat') or self.model.visual_concat == 'None':
            self.model.visual_concat = 'token_cat'
        
        # Load checkpoint
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"📦 Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            checkpoint = {k.replace('_orig_mod.', ''): v for k, v in checkpoint.items()}
            self.model.embed_out.load_state_dict(checkpoint, strict=False)
            print("✅ Checkpoint loaded!")
        
        self.model.eval()
        torch.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = True
        
        # Print model info
        print(f"\n📊 Diffusion Model Architecture:")
        print(f"   - UNet channels: [256, 512, 1024]")
        print(f"   - Kernel size: 5")
        print(f"   - Action dimension: {self.model.action_dim}")
        print(f"   - Chunk size: {self.model.num_queries}")
        print(f"   - Parameters: {sum(p.numel() for p in self.model.embed_out.parameters()):,}")
    
    def _inference_worker(self):
        """Background inference with metrics tracking"""
        while self.running:
            try:
                data = self.inference_queue.get(timeout=0.1)
                if data is None:
                    continue
                
                rgb_obs, robot_state, instruction = data
                start_time = time.time()
                
                with torch.inference_mode():
                    actions, metrics = self._run_inference_with_metrics(
                        rgb_obs, robot_state, instruction
                    )
                
                inference_time = time.time() - start_time
                self.inference_times.append(inference_time)
                
                # Store results with timestamp
                timestamp = time.time()
                self.result_queue.put((actions, metrics, timestamp))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Inference error: {e}")
    
    def _run_inference_with_metrics(self, rgb_obs: np.ndarray, robot_state: np.ndarray, 
                                   instruction: str) -> Tuple[np.ndarray, dict]:
        """Run inference and collect detailed metrics"""
        # Image preprocessing
        curr_image = torch.from_numpy(rgb_obs / 255.0).float().to(self.device)
        curr_image = curr_image.permute(2, 0, 1).unsqueeze(0)
        
        # Expand to square
        image = self.expand2square(curr_image)
        image_tensor = self.image_processor.preprocess(
            image, return_tensors='pt', do_normalize=True, 
            do_rescale=False, do_center_crop=False
        )['pixel_values'].to(self.device, dtype=self.model.dtype)
        
        # Language processing (cached)
        if not hasattr(self, '_cached_instruction') or self._cached_instruction != instruction:
            conv = conv_templates['pythia'].copy()
            inp = DEFAULT_IMAGE_TOKEN + '\n' + instruction
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt() + " <|endoftext|>"
            
            self._cached_ids = tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
            ).unsqueeze(0).to(self.device)
            self._cached_instruction = instruction
        
        # Prepare inputs
        input_ids = self._cached_ids
        attn_mask = input_ids.ne(self.tokenizer.pad_token_id)
        states = torch.from_numpy(robot_state).float().unsqueeze(0).to(
            self.device, dtype=self.model.dtype
        )
        
        # Get hidden states
        input_ids, attention_mask, _, inputs_embeds, _ = self.model.prepare_inputs_labels_for_multimodal(
            input_ids, attn_mask, None, None, image_tensor,
            images_r=image_tensor, images_top=None,
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
        
        # Diffusion denoising with metrics
        B = 1
        Tp = self.model.num_queries
        action_dim = self.model.action_dim
        
        # Initialize from noise
        initial_noise = torch.randn((B, Tp, action_dim), device=self.device, dtype=hidden_states.dtype)
        naction = initial_noise.clone()
        
        # Track denoising progress
        denoising_progress = []
        noise_norms = []
        
        self.model.noise_scheduler.set_timesteps(self.model.num_inference_timesteps)
        
        for i, k in enumerate(self.model.noise_scheduler.timesteps):
            # Predict noise
            noise_pred = self.model.embed_out(
                naction, k, global_cond=hidden_states, states=states
            )
            
            # Track metrics
            noise_norms.append(noise_pred.norm().item())
            
            # Denoise step
            prev_naction = naction.clone()
            naction = self.model.noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=naction
            ).prev_sample
            
            # Track denoising progress
            step_change = (naction - prev_naction).norm().item()
            denoising_progress.append(step_change)
        
        # Final actions
        actions = naction.cpu().numpy()
        
        # Compute metrics
        metrics = {
            'initial_noise_norm': initial_noise.norm().item(),
            'final_action_norm': naction.norm().item(),
            'noise_reduction': initial_noise.norm().item() - naction.norm().item(),
            'denoising_steps': len(denoising_progress),
            'denoising_progress': denoising_progress,
            'noise_norms': noise_norms,
            'action_range': (actions.min(), actions.max()),
            'action_std': actions.std()
        }
        
        return actions, metrics
    
    def expand2square(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Expand image to square maintaining aspect ratio"""
        B, C, H, W = img_tensor.shape
        if H == W:
            return img_tensor.permute(0, 2, 3, 1)
        
        max_dim = max(H, W)
        pad_h = (max_dim - H) // 2
        pad_w = (max_dim - W) // 2
        
        # Use reflection padding for better edges
        padded = torch.nn.functional.pad(
            img_tensor, (pad_w, pad_w, pad_h, pad_h), mode='reflect'
        )
        return padded.permute(0, 2, 3, 1)
    
    def request_inference(self, rgb_obs: np.ndarray, robot_state: np.ndarray, 
                         instruction: str) -> bool:
        """Request async inference"""
        try:
            self.inference_queue.put_nowait((rgb_obs, robot_state, instruction))
            return True
        except queue.Full:
            return False
    
    def get_action_with_interpolation(self, current_time: float, 
                                    interpolation: str = 'cubic') -> Tuple[np.ndarray, dict]:
        """Get action with advanced interpolation"""
        # Process new results
        try:
            while True:
                actions, metrics, timestamp = self.result_queue.get_nowait()
                # Add actions to buffer with timestamps
                for i in range(actions.shape[1]):
                    self.action_buffer.append(actions[0, i])
                    self.action_timestamps.append(timestamp + i * 0.05)  # 20Hz actions
                
                # Store latest metrics
                self.diffusion_metrics['noise_norms'].append(metrics['noise_norms'])
                self.diffusion_metrics['action_magnitudes'].append(metrics['final_action_norm'])
        except queue.Empty:
            pass
        
        # Get interpolated action
        if len(self.action_buffer) < 2:
            return np.zeros(4), {}
        
        # Find surrounding actions for interpolation
        if interpolation == 'linear':
            return self.action_buffer[0], {}
        elif interpolation == 'cubic' and len(self.action_buffer) >= 4:
            # Cubic interpolation for smoother motion
            actions = np.array(list(self.action_buffer)[:4])
            t = 0.25  # Interpolate at 1/4 point
            # Catmull-Rom spline
            a = actions[1]
            b = actions[2]
            c = 0.5 * (actions[2] - actions[0])
            d = 0.5 * (actions[3] - actions[1])
            
            action = a + c*t + (3*b - 3*a - 2*c - d)*t*t + (2*a - 2*b + c + d)*t*t*t
            
            # Remove used action
            self.action_buffer.popleft()
            self.action_timestamps.popleft()
            
            return action, {'interpolation': 'cubic'}
        else:
            action = self.action_buffer.popleft()
            self.action_timestamps.popleft()
            return action, {}
    
    def get_metrics_summary(self) -> dict:
        """Get performance metrics summary"""
        if not self.inference_times:
            return {}
        
        return {
            'avg_inference_time': np.mean(self.inference_times),
            'inference_fps': 1.0 / np.mean(self.inference_times) if self.inference_times else 0,
            'buffer_size': len(self.action_buffer),
            'total_inferences': len(self.diffusion_metrics['action_magnitudes']),
            'avg_action_magnitude': np.mean(self.diffusion_metrics['action_magnitudes']) 
                if self.diffusion_metrics['action_magnitudes'] else 0
        }
    
    def stop(self):
        """Clean shutdown"""
        self.running = False
        self.inference_thread.join()

def run_precision_demo(controller: PrecisionVLAController, env, instruction: str,
                      max_steps: int = 500, target_fps: int = 30,
                      log_interval: int = 50) -> dict:
    """Run high-precision demo with detailed logging"""
    print("\n🎯 Starting HIGH-PRECISION demo...")
    print(f"📝 Instruction: '{instruction}'")
    print(f"🔬 Diffusion steps: {controller.diffusion_steps}")
    print(f"⏱️  Max steps: {max_steps}")
    print(f"🎬 Target FPS: {target_fps}")
    print("="*60)
    
    obs, info = env.reset()
    env.render()
    
    # Tracking
    episode_data = {
        'rewards': [],
        'actions': [],
        'observations': [],
        'inference_times': [],
        'success': False,
        'total_steps': 0
    }
    
    fps_buffer = deque(maxlen=30)
    last_inference_time = 0
    inference_interval = 0.05  # 20Hz inference for high precision
    
    total_reward = 0.0
    step = 0
    last_action = np.zeros(4)
    
    # Action smoothing parameters
    action_smoothing = 0.5  # Less smoothing for higher precision
    target_frame_time = 1.0 / target_fps
    
    try:
        while step < max_steps:
            frame_start = time.time()
            
            # Request inference at high frequency
            current_time = time.time()
            if current_time - last_inference_time > inference_interval:
                # Get observation
                old_mode = env.render_mode
                env.render_mode = 'rgb_array'
                env.camera_name = 'corner'
                rgb = env.render()
                env.render_mode = old_mode
                env.camera_name = None
                
                if rgb is not None:
                    success = controller.request_inference(rgb, obs[:7], instruction)
                    if success:
                        last_inference_time = current_time
            
            # Get interpolated action
            raw_action, interp_info = controller.get_action_with_interpolation(
                current_time, interpolation='cubic'
            )
            
            # Smooth action
            action = action_smoothing * last_action + (1 - action_smoothing) * raw_action
            last_action = action
            
            # Clip actions
            action = np.clip(action, -1.0, 1.0)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # Store data
            episode_data['rewards'].append(reward)
            episode_data['actions'].append(action.copy())
            episode_data['observations'].append(obs.copy())
            
            # Render
            env.render()
            
            # Track FPS
            frame_time = time.time() - frame_start
            fps_buffer.append(frame_time)
            
            # Detailed logging
            if step % log_interval == 0 and len(fps_buffer) > 0:
                current_fps = 1.0 / np.mean(fps_buffer)
                metrics = controller.get_metrics_summary()
                
                print(f"\n📊 Step {step:4d}/{max_steps}")
                print(f"   FPS: {current_fps:5.1f} | Reward: {total_reward:7.3f}")
                print(f"   Action: [{action[0]:+.3f}, {action[1]:+.3f}, {action[2]:+.3f}, {action[3]:+.3f}]")
                print(f"   Buffer: {metrics.get('buffer_size', 0):2d} | Inference FPS: {metrics.get('inference_fps', 0):.1f}")
                print(f"   Avg inference time: {metrics.get('avg_inference_time', 0)*1000:.1f}ms")
                
                # Show recent rewards
                recent_rewards = episode_data['rewards'][-50:]
                if recent_rewards:
                    print(f"   Recent reward (50 steps): μ={np.mean(recent_rewards):.3f}, σ={np.std(recent_rewards):.3f}")
            
            # Check success
            if info.get("success", False):
                episode_data['success'] = True
                print(f"\n🎉 SUCCESS! Task completed at step {step}")
                print("🏆 Showing success for 3 seconds...")
                for _ in range(90):  # 3 seconds at 30 FPS
                    env.render()
                    time.sleep(0.033)
                break
            
            if done:
                print(f"\n⏹️  Episode ended at step {step}")
                break
            
            # Maintain FPS
            elapsed = time.time() - frame_start
            if elapsed < target_frame_time:
                time.sleep(target_frame_time - elapsed)
            
            step += 1
            
    except KeyboardInterrupt:
        print("\n⏹️  Demo stopped by user")
    finally:
        controller.stop()
        episode_data['total_steps'] = step
    
    # Final analysis
    print("\n" + "="*70)
    print("🔬 HIGH-PRECISION DEMO ANALYSIS")
    print("="*70)
    
    avg_fps = 1.0 / np.mean(fps_buffer) if fps_buffer else 0
    print(f"📈 Performance:")
    print(f"   Total Steps: {step}")
    print(f"   Total Reward: {total_reward:.3f}")
    print(f"   Average FPS: {avg_fps:.1f}")
    print(f"   Success: {'✅ Yes!' if episode_data['success'] else '❌ No'}")
    
    if episode_data['actions']:
        actions = np.array(episode_data['actions'])
        print(f"\n📊 Action Statistics:")
        print(f"   Range: [{actions.min():.3f}, {actions.max():.3f}]")
        print(f"   Mean: {actions.mean(axis=0)}")
        print(f"   Std:  {actions.std(axis=0)}")
        print(f"   Magnitude: μ={np.linalg.norm(actions, axis=1).mean():.3f}")
    
    if episode_data['rewards']:
        rewards = np.array(episode_data['rewards'])
        print(f"\n💰 Reward Analysis:")
        print(f"   Total: {rewards.sum():.3f}")
        print(f"   Mean: {rewards.mean():.3f}")
        print(f"   Std: {rewards.std():.3f}")
        print(f"   Max: {rewards.max():.3f}")
        print(f"   Positive steps: {(rewards > 0).sum()} ({(rewards > 0).mean()*100:.1f}%)")
    
    return episode_data

def main():
    parser = argparse.ArgumentParser(description="High-Precision MetaWorld Demo")
    parser.add_argument("--model-path", type=str, default="VLM_weights/Llava-Pythia-400M")
    parser.add_argument("--checkpoint", type=str, 
                       default="checkpoints/TinyVLA-raw_actions_metaworld/diff_head_raw_final.pth")
    parser.add_argument("--task", type=str, default="reach-v3",
                       choices=['pick-place-v3', 'reach-v3', 'push-v3', 'door-open-v3'])
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=500,
                       help="Maximum steps (default: 500 for longer episodes)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--diffusion-steps", type=int, default=10,
                       help="Number of diffusion denoising steps (1-50)")
    parser.add_argument("--seed", type=int, default=0)
    
    args = parser.parse_args()
    
    # Auto-generate instruction
    if args.instruction is None:
        task_instructions = {
            'pick-place-v3': "Pick up the red block and place it at the target location.",
            'reach-v3': "Move the robot arm to reach the target position.",
            'push-v3': "Push the object to the goal area.",
            'door-open-v3': "Open the door by pulling the handle."
        }
        args.instruction = task_instructions.get(args.task, "Complete the task.")
    
    set_seed(args.seed)
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    
    print("="*70)
    print("🔬 TinyVLA High-Precision Diffusion Policy Demo")
    print("="*70)
    print("Features:")
    print("  • Adjustable diffusion steps for quality/speed trade-off")
    print("  • Cubic spline action interpolation")
    print("  • Detailed performance metrics")
    print("  • Extended episode length")
    print("="*70)
    
    # Initialize controller
    controller = PrecisionVLAController(
        model_path=args.model_path,
        checkpoint_path=args.checkpoint,
        diffusion_steps=args.diffusion_steps
    )
    
    # Setup environment
    print(f"\n🎮 Setting up MetaWorld environment: {args.task}")
    benchmark = metaworld.ML1(args.task)
    env = benchmark.train_classes[args.task]()
    task = random.choice(benchmark.train_tasks)
    env.set_task(task)
    
    if hasattr(env, 'env'):
        env = env.env
    
    env.render_mode = 'human'
    env.camera_name = None
    
    # Run demo
    episode_data = run_precision_demo(
        controller=controller,
        env=env,
        instruction=args.instruction,
        max_steps=args.max_steps,
        target_fps=args.fps
    )
    
    env.close()
    
    # Offer to save data
    print("\n💾 Demo complete!")
    save = input("Save episode data? (y/n): ")
    if save.lower() == 'y':
        filename = f"episode_{args.task}_{int(time.time())}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(episode_data, f)
        print(f"✅ Saved to {filename}")

if __name__ == "__main__":
    main() 