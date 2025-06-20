#!/usr/bin/env python3
"""
TinyVLA Enhanced Pick-Place Demo - Focus on Grasping Success
Analyzes grasping mechanics and provides detailed diagnostics for 50% success rate target
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

class EnhancedPickPlaceDemoRunner:
    def __init__(self, model_path, checkpoint_path):
        self.model_path = model_path
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("🚀 Loading TinyVLA model for enhanced pick-place demo...")
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
                
            # Enable GUI rendering
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
    
    def analyze_grasp_state(self, env, obs):
        """Analyze current grasping state"""
        try:
            # Get object position
            obj_pos = obs[3:6] if len(obs) >= 6 else None
            
            # Get gripper positions
            right_finger = env.get_site_pos('rightEndEffector')
            left_finger = env.get_site_pos('leftEndEffector')
            finger_com = (right_finger + left_finger) / 2
            
            # Calculate distances
            if obj_pos is not None:
                reach_dist = np.linalg.norm(obj_pos - finger_com)
                reach_dist_xy = np.linalg.norm(obj_pos[:-1] - finger_com[:-1])
                height_diff = abs(obj_pos[2] - finger_com[2])
            else:
                reach_dist = reach_dist_xy = height_diff = float('inf')
            
            # Check if object is grasped (sensor data)
            try:
                sensor_data = env.data.sensordata
                is_grasped = (sensor_data[0] > 0) and (sensor_data[1] > 0) if len(sensor_data) >= 2 else False
            except:
                is_grasped = False
            
            # Check if object is lifted
            obj_height = obj_pos[2] if obj_pos is not None else 0
            obj_init_height = getattr(env, 'objHeight', 0.6)  # Default height
            is_lifted = obj_height > (obj_init_height + 0.01)
            
            return {
                'reach_dist': reach_dist,
                'reach_dist_xy': reach_dist_xy,
                'height_diff': height_diff,
                'is_grasped': is_grasped,
                'is_lifted': is_lifted,
                'obj_height': obj_height,
                'finger_com': finger_com,
                'obj_pos': obj_pos
            }
            
        except Exception as e:
            print(f"⚠️  Grasp analysis error: {e}")
            return {
                'reach_dist': float('inf'),
                'reach_dist_xy': float('inf'),
                'height_diff': float('inf'),
                'is_grasped': False,
                'is_lifted': False,
                'obj_height': 0,
                'finger_com': None,
                'obj_pos': None
            }
    
    def run_demo(self, episodes=5, max_steps=200, prompt=None):
        """Run the enhanced pick-place demo"""
        
        # Use our best prompt from research if none provided
        if prompt is None:
            prompt = "Pick up the object and place it at the target location"
            
        print(f"\n🎯 Enhanced Pick-Place Demo - Focus on Grasping")
        print(f"📝 Prompt: '{prompt}'")
        print(f"🔄 Episodes: {episodes}")
        print(f"⏱️  Max Steps: {max_steps}")
        print(f"🎮 GUI: Enabled with detailed diagnostics")
        print(f"🎯 Target: 50% success rate")
        print("=" * 70)
        
        env = self.setup_environment()
        if env is None:
            return
        
        results = []
        grasp_stats = {
            'reached_object': 0,
            'grasped_object': 0,
            'lifted_object': 0,
            'completed_task': 0
        }
        
        for episode in range(episodes):
            print(f"\n🚀 Episode {episode + 1}/{episodes}")
            
            # Reset environment
            obs, info = env.reset()
            total_reward = 0.0
            success = False
            
            # Episode tracking
            max_grasp_achieved = False
            max_lift_achieved = False
            min_reach_dist = float('inf')
            gripper_actions = []
            
            print("🎬 Starting episode... (GUI window should be visible)")
            
            # Render initial state
            env.render()
            
            for step in range(max_steps):
                # Get camera image for model input
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
                
                # Track gripper action (4th element is gripper)
                gripper_actions.append(action[3] if len(action) > 3 else 0)
                
                # Execute action
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                
                # Analyze grasp state
                grasp_info = self.analyze_grasp_state(env, obs)
                
                # Update tracking
                min_reach_dist = min(min_reach_dist, grasp_info['reach_dist'])
                if grasp_info['is_grasped']:
                    max_grasp_achieved = True
                if grasp_info['is_lifted']:
                    max_lift_achieved = True
                
                # Print detailed progress every 50 steps
                if step % 50 == 0:
                    print(f"   Step {step:3d}: reward={total_reward:.3f} | "
                          f"reach={grasp_info['reach_dist']:.3f} | "
                          f"grasp={grasp_info['is_grasped']} | "
                          f"lift={grasp_info['is_lifted']} | "
                          f"gripper={action[3]:.3f}")
                
                # Check for success
                if info.get("success", False):
                    success = True
                    print(f"🎉 SUCCESS! Completed in {step + 1} steps")
                    break
                    
                if done:
                    print(f"⏹️  Episode ended at step {step + 1}")
                    break
            
            # Final render to show end state
            env.render()
            
            # Final grasp analysis
            final_grasp = self.analyze_grasp_state(env, obs)
            
            # Update statistics
            if min_reach_dist < 0.05:  # Reached object
                grasp_stats['reached_object'] += 1
            if max_grasp_achieved:  # Grasped object
                grasp_stats['grasped_object'] += 1
            if max_lift_achieved:  # Lifted object
                grasp_stats['lifted_object'] += 1
            if success:  # Completed task
                grasp_stats['completed_task'] += 1
            
            # Episode results
            result = {
                'episode': episode + 1,
                'reward': total_reward,
                'success': success,
                'steps': step + 1,
                'min_reach_dist': min_reach_dist,
                'grasped': max_grasp_achieved,
                'lifted': max_lift_achieved,
                'avg_gripper': np.mean(gripper_actions),
                'gripper_range': np.max(gripper_actions) - np.min(gripper_actions)
            }
            results.append(result)
            
            print(f"📊 Episode {episode + 1} Detailed Results:")
            print(f"   💰 Reward: {total_reward:.3f}")
            print(f"   ✅ Success: {success}")
            print(f"   👣 Steps: {step + 1}")
            print(f"   🎯 Min Reach Distance: {min_reach_dist:.3f}")
            print(f"   🤏 Grasped Object: {max_grasp_achieved}")
            print(f"   ⬆️  Lifted Object: {max_lift_achieved}")
            print(f"   🔧 Avg Gripper Action: {np.mean(gripper_actions):.3f}")
            print(f"   📏 Gripper Range: {np.max(gripper_actions) - np.min(gripper_actions):.3f}")
            
            # Brief pause between episodes
            if episode < episodes - 1:
                print("⏸️  Pausing 1 second before next episode...")
                time.sleep(1)
        
        # Comprehensive Summary
        print("\n" + "=" * 70)
        print("📈 ENHANCED DEMO SUMMARY")
        print("=" * 70)
        
        avg_reward = np.mean([r['reward'] for r in results])
        success_rate = np.mean([r['success'] for r in results]) * 100
        avg_steps = np.mean([r['steps'] for r in results])
        
        # Calculate success rates for each stage
        reach_rate = (grasp_stats['reached_object'] / episodes) * 100
        grasp_rate = (grasp_stats['grasped_object'] / episodes) * 100
        lift_rate = (grasp_stats['lifted_object'] / episodes) * 100
        complete_rate = (grasp_stats['completed_task'] / episodes) * 100
        
        print(f"🎯 Task: Pick-Place")
        print(f"📝 Prompt: '{prompt}'")
        print(f"📊 Average Reward: {avg_reward:.3f}")
        print(f"👣 Average Steps: {avg_steps:.1f}")
        print()
        print("🔍 MANIPULATION PIPELINE ANALYSIS:")
        print(f"   🎯 Reached Object: {reach_rate:.1f}% ({grasp_stats['reached_object']}/{episodes})")
        print(f"   🤏 Grasped Object: {grasp_rate:.1f}% ({grasp_stats['grasped_object']}/{episodes})")
        print(f"   ⬆️  Lifted Object: {lift_rate:.1f}% ({grasp_stats['lifted_object']}/{episodes})")
        print(f"   ✅ Completed Task: {complete_rate:.1f}% ({grasp_stats['completed_task']}/{episodes})")
        print()
        
        # Diagnosis and recommendations
        print("🔧 DIAGNOSIS & RECOMMENDATIONS:")
        if reach_rate < 80:
            print("   ⚠️  Issue: Poor reaching - model not getting close to object")
            print("   💡 Solution: Check action scaling, increase training on reach tasks")
        elif grasp_rate < 50:
            print("   ⚠️  Issue: Poor grasping - reaching but not closing gripper properly")
            print("   💡 Solution: Focus on gripper action training, check action[3] values")
        elif lift_rate < 50:
            print("   ⚠️  Issue: Poor lifting - grasping but not lifting object")
            print("   💡 Solution: Train on lifting motions, check vertical actions")
        elif complete_rate < 50:
            print("   ⚠️  Issue: Poor placing - lifting but not completing placement")
            print("   💡 Solution: Train on placement precision, longer episodes")
        else:
            print("   ✅ Good performance across all stages!")
        
        # Performance interpretation
        if complete_rate >= 50:
            print(f"\n🎉 TARGET ACHIEVED! {complete_rate:.1f}% success rate (≥50% target)")
        elif complete_rate >= 30:
            print(f"\n👍 GOOD progress! {complete_rate:.1f}% success rate (close to 50% target)")
        elif complete_rate >= 10:
            print(f"\n🟡 MODERATE performance. {complete_rate:.1f}% success rate (needs improvement)")
        else:
            print(f"\n⚠️  LOW performance. {complete_rate:.1f}% success rate (significant issues)")
        
        print("\n🎮 GUI window shows detailed robot manipulation!")
        print("💡 This enhanced demo provides detailed grasping diagnostics!")
        
        return results, grasp_stats

def main():
    parser = argparse.ArgumentParser(description="Enhanced TinyVLA Pick-Place Demo with Grasping Analysis")
    parser.add_argument("--model-path", type=str, 
                       default="VLM_weights/Llava-Pythia-400M",
                       help="Path to the base model")
    parser.add_argument("--checkpoint", type=str,
                       default="checkpoints/TinyVLA-droid_diffusion_metaworld/diff_head_FIXED_epoch_40.pth",
                       help="Path to the trained checkpoint")
    parser.add_argument("--episodes", type=int, default=5,
                       help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=200,
                       help="Maximum steps per episode")
    parser.add_argument("--prompt", type=str, default=None,
                       help="Custom prompt (default: use optimal prompt from research)")
    
    args = parser.parse_args()
    
    print("🎮 Enhanced TinyVLA Pick-Place Demo - Grasping Analysis")
    print("=" * 70)
    print("🔬 Focus on achieving 50% success rate:")
    print("   • Detailed grasping diagnostics")
    print("   • Manipulation pipeline analysis")
    print("   • Action analysis and recommendations")
    print("   • GUI enabled with enhanced feedback")
    print("=" * 70)
    
    # Check if model exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        print("💡 Make sure you've trained the model first!")
        return
    
    try:
        # Create demo runner
        demo = EnhancedPickPlaceDemoRunner(args.model_path, args.checkpoint)
        
        # Run the demo
        results, grasp_stats = demo.run_demo(
            episodes=args.episodes,
            max_steps=args.max_steps,
            prompt=args.prompt
        )
        
        print("\n🎉 Enhanced demo completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")

if __name__ == "__main__":
    main() 