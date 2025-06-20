#!/usr/bin/env python3
"""
Fast Real-time MetaWorld Demo with TinyVLA - Optimized for smooth rendering
Uses asynchronous inference and action buffering for real-time performance.
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

# Force GLFW for windowed rendering
os.environ['MUJOCO_GL'] = 'glfw'
print("[INFO] Using GLFW for windowed rendering.")

class FastVLAController:
    """Fast controller with asynchronous inference"""
    
    def __init__(self, model_path, checkpoint_path=None, device='cuda'):
        self.device = device
        self.load_model(model_path, checkpoint_path)
        
        # Action buffer for smooth control
        self.action_buffer = deque(maxlen=20)
        self.inference_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=5)
        
        # Start inference thread
        self.running = True
        self.inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.inference_thread.start()
        
        # Pre-generate some random actions for immediate start
        for _ in range(5):
            self.action_buffer.append(np.random.uniform(-0.1, 0.1, 4))
        
    def load_model(self, model_path, checkpoint_path):
        """Load model with optimizations"""
        print("🚀 Loading model with optimizations...")
        
        # Force float32 for scheduler
        original_float16 = torch.float16
        torch.float16 = torch.float32
        
        try:
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path, None, get_model_name_from_path(model_path), False, False
            )
        finally:
            torch.float16 = original_float16
        
        # Fix scheduler
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
            # Reduce inference steps for speed
            self.model.num_inference_timesteps = 5  # Reduced from 10
        
        if not hasattr(self.model, 'visual_concat') or self.model.visual_concat == 'None':
            self.model.visual_concat = 'token_cat'
        
        # Load checkpoint
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"📦 Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            checkpoint = {k.replace('_orig_mod.', ''): v for k, v in checkpoint.items()}
            self.model.embed_out.load_state_dict(checkpoint, strict=False)
        
        self.model.eval()
        
        # Enable torch optimizations
        torch.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = True
        
        # Compile model if available (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            try:
                print("⚡ Compiling model with torch.compile()...")
                self.model = torch.compile(self.model, mode='reduce-overhead')
            except Exception as e:
                print(f"⚠️  torch.compile failed (this is okay): {e}")
        
        self.stats = None
        stats_path = os.path.join(os.path.dirname(model_path), 'dataset_stats.pkl')
        if os.path.exists(stats_path):
            with open(stats_path, 'rb') as f:
                self.stats = pickle.load(f)
    
    def _inference_worker(self):
        """Background thread for inference"""
        while self.running:
            try:
                # Get inference request
                data = self.inference_queue.get(timeout=0.1)
                if data is None:
                    continue
                
                rgb_obs, robot_state, instruction = data
                
                # Run inference
                with torch.inference_mode():
                    actions = self._run_inference(rgb_obs, robot_state, instruction)
                
                # Put results
                self.result_queue.put(actions)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Inference error: {e}")
    
    def expand2square(self, pil_imgs, background_color):
        """Fast image expansion"""
        batch_size, channels, height, width = pil_imgs.shape
        if height == width:
            return pil_imgs
        
        max_dim = max(height, width)
        expanded_imgs = torch.full((batch_size, channels, max_dim, max_dim), 
                                 background_color[0], dtype=pil_imgs.dtype, device=pil_imgs.device)
        
        if height > width:
            offset = (max_dim - width) // 2
            expanded_imgs[:, :, :height, offset:offset + width] = pil_imgs
        else:
            offset = (max_dim - height) // 2
            expanded_imgs[:, :, offset:offset + height, :width] = pil_imgs
        
        return expanded_imgs.permute(0, 2, 3, 1)
    
    def _run_inference(self, rgb_obs, robot_state, instruction):
        """Optimized inference"""
        # Fast preprocessing
        curr_image = torch.from_numpy(rgb_obs / 255.0).float().to(self.device)
        curr_image = curr_image.permute(2, 0, 1).unsqueeze(0)
        
        # Expand and preprocess
        image = self.expand2square(curr_image, tuple(x for x in self.image_processor.image_mean))
        image_tensor = self.image_processor.preprocess(
            image, return_tensors='pt', do_normalize=True, do_rescale=False, do_center_crop=False
        )['pixel_values'].to(self.device, dtype=self.model.dtype)
        
        image_tensor_r = image_tensor  # Reuse instead of clone
        
        # Language processing (cached if same instruction)
        if not hasattr(self, '_cached_instruction') or self._cached_instruction != instruction:
            conv = conv_templates['pythia'].copy()
            inp = DEFAULT_IMAGE_TOKEN + '\n' + instruction
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt() + " <|endoftext|>"
            
            self._cached_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
            self._cached_instruction = instruction
        
        input_ids = self._cached_ids
        attn_mask = input_ids.ne(self.tokenizer.pad_token_id)
        states = torch.from_numpy(robot_state).float().unsqueeze(0).to(self.device, dtype=self.model.dtype)
        
        # Get hidden states
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
        
        # Fast diffusion
        B = 1
        Tp = self.model.num_queries
        action_dim = self.model.action_dim
        
        naction = torch.randn((B, Tp, action_dim), device=self.device, dtype=hidden_states.dtype)
        
        self.model.noise_scheduler.set_timesteps(self.model.num_inference_timesteps)
        
        for k in self.model.noise_scheduler.timesteps:
            noise_pred = self.model.embed_out(naction, k, global_cond=hidden_states, states=states)
            naction = self.model.noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=naction
            ).prev_sample
        
        return naction.cpu().numpy()
    
    def request_inference(self, rgb_obs, robot_state, instruction):
        """Request async inference"""
        try:
            self.inference_queue.put_nowait((rgb_obs, robot_state, instruction))
        except queue.Full:
            pass  # Skip if queue is full
    
    def get_action(self, default_action=None):
        """Get action from buffer or use default"""
        # Check for new inference results
        try:
            while True:
                actions = self.result_queue.get_nowait()
                # Add all actions to buffer
                for i in range(actions.shape[1]):
                    action = actions[0, i]
                    if self.stats and 'action_mean' in self.stats:
                        action = action * self.stats['action_std'] + self.stats['action_mean']
                    self.action_buffer.append(np.clip(action, -1.0, 1.0))
        except queue.Empty:
            pass
        
        # Return action from buffer or default
        if len(self.action_buffer) > 0:
            return self.action_buffer.popleft()
        else:
            return default_action if default_action is not None else np.zeros(4)
    
    def stop(self):
        """Stop inference thread"""
        self.running = False
        self.inference_thread.join()

def setup_metaworld_env(task_name='pick-place-v3'):
    """Setup environment"""
    print(f"🎮 Setting up MetaWorld environment: {task_name}")
    
    benchmark = metaworld.ML1(task_name)
    env = benchmark.train_classes[task_name]()
    task = random.choice(benchmark.train_tasks)
    env.set_task(task)
    
    if hasattr(env, 'env'):
        env = env.env
    
    env.render_mode = 'human'
    env.camera_name = None
    
    return env

def run_fast_demo(controller, env, instruction, max_steps=300, target_fps=30):
    """Run fast real-time demo"""
    print("\n🏃 Starting FAST real-time demo...")
    print(f"📝 Instruction: '{instruction}'")
    print(f"⚡ Target FPS: {target_fps}")
    print("🖼️  MuJoCo window should appear...")
    print("Press Ctrl+C to stop\n")
    
    obs, info = env.reset()
    env.render()
    
    # Performance tracking
    fps_buffer = deque(maxlen=30)
    last_inference_time = 0
    inference_interval = 0.1  # Request inference every 100ms
    
    total_reward = 0.0
    step = 0
    success = False
    last_action = np.zeros(4)
    
    # For smooth control
    action_smoothing = 0.7  # Blend with previous action
    
    target_frame_time = 1.0 / target_fps
    
    try:
        while step < max_steps:
            frame_start = time.time()
            
            # Request inference periodically
            current_time = time.time()
            if current_time - last_inference_time > inference_interval:
                # Get camera image for inference
                old_mode = env.render_mode
                env.render_mode = 'rgb_array'
                env.camera_name = 'corner'
                rgb = env.render()
                env.render_mode = old_mode
                env.camera_name = None
                
                if rgb is not None:
                    controller.request_inference(rgb, obs[:7], instruction)
                    last_inference_time = current_time
            
            # Get action (from buffer or interpolated)
            raw_action = controller.get_action(default_action=last_action)
            
            # Smooth actions for better visual quality
            action = action_smoothing * last_action + (1 - action_smoothing) * raw_action
            last_action = action
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # Render to window
            env.render()
            
            # Track FPS
            frame_time = time.time() - frame_start
            fps_buffer.append(frame_time)
            
            # Print status periodically
            if step % 30 == 0 and len(fps_buffer) > 0:
                current_fps = 1.0 / np.mean(fps_buffer)
                buffer_size = len(controller.action_buffer)
                print(f"Step {step:3d} | FPS: {current_fps:4.1f} | Reward: {total_reward:6.2f} | Buffer: {buffer_size:2d} | Action: [{action[0]:+.2f}, {action[1]:+.2f}, {action[2]:+.2f}, {action[3]:+.2f}]")
            
            # Check success
            if info.get("success", False):
                success = True
                print(f"\n🎉 SUCCESS! Task completed at step {step}")
                for _ in range(60):  # Show success for 2 seconds
                    env.render()
                    time.sleep(0.033)
                break
            
            if done:
                print(f"\n⏹️  Episode ended at step {step}")
                break
            
            # Maintain target FPS
            elapsed = time.time() - frame_start
            if elapsed < target_frame_time:
                time.sleep(target_frame_time - elapsed)
            
            step += 1
            
    except KeyboardInterrupt:
        print("\n⏹️  Demo stopped by user")
    finally:
        controller.stop()
    
    # Summary
    avg_fps = 1.0 / np.mean(fps_buffer) if fps_buffer else 0
    print("\n" + "="*60)
    print("🏁 FAST DEMO SUMMARY")
    print("="*60)
    print(f"🕐 Total Steps: {step}")
    print(f"🏅 Total Reward: {total_reward:.3f}")
    print(f"✨ Success: {'✅ Yes!' if success else '❌ No'}")
    print(f"⚡ Average FPS: {avg_fps:.1f}")
    print("\n💡 Performance Tips:")
    print(f"   - Achieved {avg_fps:.1f} FPS (target was {target_fps})")
    if avg_fps < target_fps * 0.8:
        print("   - Consider reducing inference frequency or diffusion steps")
    else:
        print("   - Great performance! The demo ran smoothly")

def main():
    parser = argparse.ArgumentParser(description="Fast Real-time MetaWorld Demo")
    parser.add_argument("--model-path", type=str, default="VLM_weights/Llava-Pythia-400M")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/TinyVLA-raw_actions_metaworld/diff_head_raw_final.pth")
    parser.add_argument("--task", type=str, default="reach-v3",
                       choices=['pick-place-v3', 'reach-v3', 'push-v3', 'door-open-v3'])
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    parser.add_argument("--seed", type=int, default=0)
    
    args = parser.parse_args()
    
    # Auto-generate instruction
    if args.instruction is None:
        task_instructions = {
            'pick-place-v3': "Pick up the red block and place it at the target.",
            'reach-v3': "Move the arm to reach the target position.",
            'push-v3': "Push the object to the goal.",
            'door-open-v3': "Open the door by pulling the handle."
        }
        args.instruction = task_instructions.get(args.task, "Complete the task.")
    
    set_seed(args.seed)
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    
    print("="*60)
    print("🚀 TinyVLA Fast Real-Time Demo")
    print("="*60)
    print("Optimized for smooth real-time rendering with:")
    print("  • Asynchronous inference")
    print("  • Action buffering")
    print("  • Reduced diffusion steps")
    print("  • Action smoothing")
    print("="*60)
    
    # Initialize
    controller = FastVLAController(
        model_path=args.model_path,
        checkpoint_path=args.checkpoint
    )
    
    env = setup_metaworld_env(args.task)
    
    # Run demo
    run_fast_demo(
        controller=controller,
        env=env,
        instruction=args.instruction,
        max_steps=args.max_steps,
        target_fps=args.fps
    )
    
    env.close()

if __name__ == "__main__":
    main() 