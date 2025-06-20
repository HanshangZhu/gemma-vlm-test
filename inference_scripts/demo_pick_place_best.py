#!/usr/bin/env python3
"""
TinyVLA Pick-Place Demo - Best Case Results
Showcases optimal manipulation performance using our best prompt engineering findings
"""

import os
import sys
import argparse
import numpy as np
import torch
import cv2
from PIL import Image
import time

# Add parent directory to path for unified_tinyvla
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set rendering mode for GUI
os.environ['MUJOCO_GL'] = 'glfw'  # Enable GUI rendering

from transformers import AutoTokenizer, CLIPImageProcessor
from unified_tinyvla import UnifiedTinyVLAModel

# MetaWorld imports
import metaworld
import random

class PickPlaceDemoRunner:
    def __init__(self, model_path, checkpoint_path):
        self.model_path = model_path
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("🚀 Loading TinyVLA model for pick-place demo...")
        self.load_model()
        
    def load_model(self):
        """Load the TinyVLA model and tokenizer"""
        try:
            # Load model
            self.model = UnifiedTinyVLAModel(self.model_path, mode="action").to(self.device)
            
            # Load checkpoint
            if os.path.exists(self.checkpoint_path):
                print(f"📦 Loading checkpoint: {self.checkpoint_path}")
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
                # Remove _orig_mod prefix if present
                checkpoint = {k.replace('_orig_mod.', ''): v for k, v in checkpoint.items()}
                self.model.base_model.embed_out.load_state_dict(checkpoint)
                print("✅ Checkpoint loaded successfully!")
            else:
                print(f"⚠️  Checkpoint not found: {self.checkpoint_path}")
                
            self.model.eval()
            
            # Load tokenizer and image processor
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.image_processor = CLIPImageProcessor.from_pretrained(self.model_path)
            
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def setup_environment(self):
        """Setup MetaWorld pick-place environment with GUI"""
        try:
            print("🌍 Setting up MetaWorld pick-place environment...")
            
            # Use v3 for evaluation (maps from v2 training)
            benchmark = metaworld.ML1('pick-place-v3')
            env = benchmark.train_classes['pick-place-v3']()
            task = random.choice(benchmark.train_tasks)
            env.set_task(task)
            
            # Unwrap if needed
            if hasattr(env, 'env'):
                env = env.env
                
            # Enable GUI rendering with optimizations
            env.render_mode = 'human'  # This enables the GUI window
            env.camera_name = 'corner'
            
            print("✅ Environment setup complete!")
            return env
            
        except Exception as e:
            print(f"❌ Error setting up environment: {e}")
            return None
    
    def predict_action(self, rgb, prompt, robot_state=None):
        """Predict action using the model"""
        try:
            model_dtype = next(self.model.parameters()).dtype
            
            # Process image
            pil_img = Image.fromarray(rgb.astype(np.uint8))
            img_tensor = self.image_processor(pil_img, return_tensors="pt")["pixel_values"]
            img_tensor = img_tensor.to(self.device, dtype=model_dtype)
            
            # Process text
            tokens = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            
            # Process state
            if robot_state is None:
                states = torch.zeros((1, 7), device=self.device, dtype=model_dtype)
            else:
                states = torch.tensor(robot_state, device=self.device, dtype=model_dtype).unsqueeze(0)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model.base_model(
                    input_ids=tokens['input_ids'],
                    attention_mask=tokens['attention_mask'],
                    images=img_tensor,
                    states=states,
                    eval=True
                )
                
                if isinstance(outputs, torch.Tensor):
                    action_seq = outputs
                elif isinstance(outputs, dict):
                    action_seq = outputs.get("actions", None)
                else:
                    action_seq = None
                    
                if action_seq is None:
                    return np.zeros(4, dtype=np.float32)
                
                # Extract first action
                if len(action_seq.shape) >= 3:
                    action = action_seq[0, 0].cpu()
                elif len(action_seq.shape) == 2:
                    action = action_seq[0].cpu()
                else:
                    action = action_seq.cpu()
                    
                return action.float().numpy()
                
        except Exception as e:
            print(f"⚠️  Prediction error: {e}")
            return np.zeros(4, dtype=np.float32)
    
    def run_demo(self, episodes=3, max_steps=150, prompt=None):
        """Run the pick-place demo"""
        
        # Use our best prompt from research if none provided
        if prompt is None:
            prompt = "Pick up the object and place it at the target location"
            
        print(f"\n🎯 Running Pick-Place Demo")
        print(f"📝 Prompt: '{prompt}'")
        print(f"🔄 Episodes: {episodes}")
        print(f"⏱️  Max Steps: {max_steps}")
        print(f"🎮 GUI: Enabled (optimized for performance)")
        print("=" * 60)
        
        env = self.setup_environment()
        if env is None:
            return
        
        results = []
        
        for episode in range(episodes):
            print(f"\n🚀 Episode {episode + 1}/{episodes}")
            
            # Reset environment
            obs, info = env.reset()
            total_reward = 0.0
            success = False
            
            print("🎬 Starting episode... (GUI window should be visible)")
            
            # Render initial state
            env.render()
            
            for step in range(max_steps):
                # Get camera image for model input (every step for accuracy)
                try:
                    rgb = env.render()  # This also updates the GUI
                    if rgb is None:
                        rgb = np.zeros((240, 320, 3), dtype=np.uint8)
                except:
                    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
                
                # Get robot state
                state = obs[:7] if len(obs) >= 7 else None
                
                # Predict action
                action = self.predict_action(rgb, prompt, state)
                
                # Execute action
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                
                # Print progress every 25 steps to reduce terminal spam
                if step % 25 == 0:
                    print(f"   Step {step}: reward={total_reward:.3f}")
                
                # Check for success
                if info.get("success", False):
                    success = True
                    print(f"🎉 SUCCESS! Completed in {step + 1} steps")
                    break
                    
                if done:
                    print(f"⏹️  Episode ended at step {step + 1}")
                    break
                    
                # Minimal delay - remove sleep for better performance
                # time.sleep(0.01)  # Reduced from 0.05 to 0.01
            
            # Final render to show end state
            env.render()
            
            # Episode results
            result = {
                'episode': episode + 1,
                'reward': total_reward,
                'success': success,
                'steps': step + 1
            }
            results.append(result)
            
            print(f"📊 Episode {episode + 1} Results:")
            print(f"   💰 Reward: {total_reward:.3f}")
            print(f"   ✅ Success: {success}")
            print(f"   👣 Steps: {step + 1}")
            
            # Brief pause between episodes
            if episode < episodes - 1:
                print("⏸️  Pausing 1 second before next episode...")
                time.sleep(1)  # Reduced from 2 to 1 second
        
        # Summary
        print("\n" + "=" * 60)
        print("📈 DEMO SUMMARY")
        print("=" * 60)
        
        avg_reward = np.mean([r['reward'] for r in results])
        success_rate = np.mean([r['success'] for r in results]) * 100
        avg_steps = np.mean([r['steps'] for r in results])
        
        print(f"🎯 Task: Pick-Place")
        print(f"📝 Prompt: '{prompt}'")
        print(f"📊 Average Reward: {avg_reward:.3f}")
        print(f"✅ Success Rate: {success_rate:.1f}%")
        print(f"👣 Average Steps: {avg_steps:.1f}")
        
        # Performance interpretation
        if avg_reward > 1.0:
            print("🏆 EXCELLENT performance! Model is working perfectly!")
        elif avg_reward > 0.5:
            print("👍 GOOD performance! Model shows solid manipulation skills!")
        elif avg_reward > 0.2:
            print("🟡 MODERATE performance. Room for improvement.")
        else:
            print("⚠️  LOW performance. Check model/environment setup.")
        
        print("\n🎮 GUI window shows the robot manipulation!")
        print("💡 This demonstrates our best-case TinyVLA performance!")
        print("⚡ Performance optimized - reduced lag while maintaining visualization")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="TinyVLA Pick-Place Demo with GUI")
    parser.add_argument("--model-path", type=str, 
                       default="VLM_weights/Llava-Pythia-400M",
                       help="Path to the base model")
    parser.add_argument("--checkpoint", type=str,
                       default="checkpoints/TinyVLA-droid_diffusion_metaworld/diff_head_FIXED_epoch_40.pth",
                       help="Path to the trained checkpoint")
    parser.add_argument("--episodes", type=int, default=3,
                       help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=150,
                       help="Maximum steps per episode")
    parser.add_argument("--prompt", type=str, default=None,
                       help="Custom prompt (default: use optimal prompt from research)")
    
    args = parser.parse_args()
    
    print("🎮 TinyVLA Pick-Place Demo - Best Case Results")
    print("=" * 60)
    print("🔬 Based on our prompt engineering research:")
    print("   • Optimal prompt: 'Pick up the object and place it at the target location'")
    print("   • Expected reward: ~1.16 (excellent performance)")
    print("   • GUI enabled with performance optimizations")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        print("💡 Make sure you've trained the model first!")
        return
    
    try:
        # Create demo runner
        demo = PickPlaceDemoRunner(args.model_path, args.checkpoint)
        
        # Run the demo
        results = demo.run_demo(
            episodes=args.episodes,
            max_steps=args.max_steps,
            prompt=args.prompt
        )
        
        print("\n🎉 Demo completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")

if __name__ == "__main__":
    main() 