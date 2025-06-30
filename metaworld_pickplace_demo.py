#!/usr/bin/env python3
"""
🤖 MetaWorld Pick-Place Demo - Simple Sequential Approach 🤖

This demo uses a simple sequential approach:
1. Get frame from environment
2. Model predicts action 
3. Step environment
4. Update visual display
5. Repeat

No complex context management needed!

Usage:
    python metaworld_pickplace_demo.py [--episodes N] [--max-steps N]
"""

import argparse
import numpy as np
import time
import sys
import os
from pathlib import Path
import warnings
import glfw  # For resizing the Mujoco viewer window
import mujoco_py  # Access MjViewer constants
import gymnasium.envs.mujoco.mujoco_rendering as mjrender

# Suppress verbose outputs
import contextlib
from io import StringIO

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent))

# Import MetaWorld - this is required
try:
    import metaworld
    print("✅ MetaWorld imported successfully")
except ImportError as e:
    print(f"❌ MetaWorld import failed: {e}")
    print("💡 Please install MetaWorld: pip install git+https://github.com/Farama-Foundation/Metaworld.git")
    sys.exit(1)

# Try to import TinyVLA (optional)
try:
    from tinyvla_loader import load_tinyvla
    HAS_TINYVLA = True
    print("✅ TinyVLA loader available")
except ImportError as e:
    HAS_TINYVLA = False
    print(f"⚠️ TinyVLA not available: {e}")
    print("🎲 Will use random actions instead")

# Optional PIL for image processing
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("⚠️ PIL not available - visual model features limited")

MODEL_RESOLUTION = (320, 180)  # (width, height) native resolution expected by TinyVLA

# ---------------------------------------------------------------------------
# Force MuJoCo's default viewer window to the model-native resolution BEFORE
# the window is ever created.  This avoids fullscreen windows and associated
# GPU overhead on high-DPI monitors.
# ---------------------------------------------------------------------------
mujoco_py.MjViewerBasic._WINDOW_WIDTH  = 320  # width  of GUI window
mujoco_py.MjViewerBasic._WINDOW_HEIGHT = 180  # height of GUI window

# ---------------------------------------------------------------------------
# Monkey-patch gymnasium's WindowViewer so that any new GUI window opens at
# 320 × 180 instead of half the monitor size.
# ---------------------------------------------------------------------------
_OriginalWindowViewer = mjrender.WindowViewer


class _TinyWindowViewer(_OriginalWindowViewer):
    def __init__(self, model, data, **kwargs):
        kwargs.setdefault('width', 320)
        kwargs.setdefault('height', 180)
        super().__init__(model, data, **kwargs)


# Replace the class in the module so future constructions use the tiny version
mjrender.WindowViewer = _TinyWindowViewer

def suppress_tinyvla_verbose():
    """Context manager to suppress TinyVLA verbose output"""
    return contextlib.redirect_stdout(StringIO())


def load_model_safely():
    """
    🤖 Load TinyVLA Model with Comprehensive Error Handling
    
    Returns:
        tuple: (model_object_or_None, descriptive_string)
    """
    if not HAS_TINYVLA:
        return None, "TinyVLA loader not available"
    
    # Try multiple potential model paths
    potential_paths = [
        "VLA_weights/full_training_bs1_final/step_1000",  # Latest completed training
    ]
    
    for model_path in potential_paths:
        if Path(model_path).exists():
            try:
                with suppress_tinyvla_verbose():
                    model = load_tinyvla(lora_checkpoint_path=model_path)
                if model is not None:
                    return model, f"TinyVLA (from {model_path})"
            except Exception as e:
                print(f"⚠️ Model loading failed for {model_path}: {e}")
    
    return None, "Random Heuristic (TinyVLA unavailable)"


def setup_metaworld_environment():
    """
    🌍 Setup MetaWorld Pick-Place-v2 Environment
    
    Creates and configures the MetaWorld environment for pick-place task.
    
    Returns:
        env: MetaWorld environment instance
    """
    print("🌍 Setting up MetaWorld environment...")
    
    # 🎯 ENABLE GUI WINDOW: Use GLFW for visual display  
    if 'DISPLAY' in os.environ:
        os.environ['MUJOCO_GL'] = 'glfw'  # Enable GUI window
    else:
        os.environ['MUJOCO_GL'] = 'egl' 
    
    # Create MetaWorld ML1 benchmark with pick-place-v2 task
    try:
        ml1 = metaworld.ML1('pick-place-v2', seed=42)
        
        # Get the environment class and instantiate it
        env_cls = ml1.train_classes['pick-place-v2']
        env = env_cls()  # Create environment without render_mode parameter
        
        # Set the task
        task = list(ml1.train_tasks)[0]  # Get first training task
        env.set_task(task)
        
        # Test environment by resetting it (handle tuple return)
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            initial_obs, info = reset_result
        else:
            initial_obs = reset_result
        
        # Test rendering capabilities
        try:
            # Test offscreen rendering (for model input)
            offscreen_frame = env.render(offscreen=True, resolution=MODEL_RESOLUTION)
            if offscreen_frame is not None:
                print(f"✅ Offscreen rendering working: {offscreen_frame.shape}")
            
            # Test visual rendering (opens GUI window)  
            env.render(offscreen=False)
            print("✅ Visual rendering working")
            
        except Exception as e:
            print(f"⚠️ Rendering test failed: {e}")
            
        return env
        
    except Exception as e:
        print(f"❌ MetaWorld environment setup failed: {e}")
        raise


def get_action(model, env, obs, step):
    """
    🎯 Simple Sequential Action Prediction
    
    Args:
        model: TinyVLA model instance (or None)
        env: MetaWorld environment  
        obs: Current observation
        step: Current step number
        
    Returns:
        action: Action to take (numpy array)
        action_source: String describing action source
    """
    
    if model is not None:
        try:
            # Simple approach: get frame for model input
            rgb_frame = env.render(offscreen=True, resolution=MODEL_RESOLUTION)
            
            if rgb_frame is not None and HAS_PIL:
                # Convert to PIL Image for model
                if isinstance(rgb_frame, np.ndarray) and rgb_frame.size > 0:
                    if rgb_frame.max() <= 1.0:
                        rgb_frame = (rgb_frame * 255).astype(np.uint8)
                    
                    pil_image = Image.fromarray(rgb_frame)
                    
                    # Get robot state (first 7 elements typically)
                    robot_state = obs[:7] if len(obs) >= 7 else obs
                    
                    # Get model prediction with suppressed output
                    with suppress_tinyvla_verbose():
                        action = model.predict_action(
                            pil_image, 
                            robot_state, 
                            "pick up the red object and place it on the green target"
                        )
                    
                    # Scale up actions for more visible movement
                    action = action * 2.0  # Double the action magnitude
                    
                    # Validate action
                    if not np.isnan(action).any():
                        return action, "TinyVLA Model"
                        
        except Exception as e:
            if step % 50 == 0:  # Only log errors occasionally
                print(f"⚠️ Model prediction failed at step {step}: {e}")
    
    # Fallback to large random actions
    action = env.action_space.sample() * 2.5  # Large random actions for visibility
    return action, "Random"


def run_episode(env, model, episode_num, max_steps):
    """
    🎮 Run Single Episode with Simple Sequential Approach
    
    Args:
        env: MetaWorld environment
        model: TinyVLA model (or None)
        episode_num: Episode number for logging
        max_steps: Maximum steps per episode
        
    Returns:
        dict: Episode results
    """
    print(f"\n🎬 Episode {episode_num} starting...")
    
    # Reset environment (handle tuple return)
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, info = reset_result
    else:
        obs = reset_result
    
    # Track initial robot position for movement verification
    initial_robot_pos = obs[:3] if len(obs) >= 3 else None
    
    # Initial visual render: show starting state
    try:
        env.render(offscreen=False, resolution=MODEL_RESOLUTION)  # Update visual window with initial state
        # Force GLFW window to the same resolution (prevent fullscreen lag)
        if hasattr(env, 'viewer') and getattr(env.viewer, 'window', None) is not None:
            try:
                glfw.set_window_size(env.viewer.window, MODEL_RESOLUTION[0], MODEL_RESOLUTION[1])
            except Exception:
                pass
    except Exception as e:
        print(f"⚠️ Initial visual render failed: {e}")
    
    total_reward = 0
    step = 0
    success = False
    action_sources = []
    
    episode_start_time = time.time()
    
    while step < max_steps:
        # Simple sequential approach:
        # 1. Get action (this will get frame for model input)
        action, action_source = get_action(model, env, obs, step)
        action_sources.append(action_source)
        
        # 2. Step environment
        result = env.step(action)
        
        # Handle different return formats (gym compatibility)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
        
        # 3. Update visual display (simple, no complex timing)
        try:
            env.render(offscreen=False, resolution=MODEL_RESOLUTION)  # Visual window update
            if hasattr(env, 'viewer') and getattr(env.viewer, 'window', None) is not None:
                try:
                    glfw.set_window_size(env.viewer.window, MODEL_RESOLUTION[0], MODEL_RESOLUTION[1])
                except Exception:
                    pass
        except Exception as e:
            if step % 100 == 0:  # Only log render errors occasionally
                print(f"⚠️ Visual render failed at step {step}: {e}")
        
        # 🔍 Verify robot movement periodically
        if step % 50 == 0 and initial_robot_pos is not None:
            current_robot_pos = obs[:3] if len(obs) >= 3 else None
            if current_robot_pos is not None:
                movement = np.linalg.norm(current_robot_pos - initial_robot_pos)
                print(f"🤖 Step {step}: Moved {movement:.3f} units, Action: {action[:4]}")
        
        total_reward += reward
        step += 1
        
        # Check for success
        if info.get('success', False):
            success = True
            print(f"🎉 SUCCESS at step {step}! Task completed!")
            break
            
        # Log progress periodically
        if step % 100 == 0:
            print(f"Step {step}: reward={reward:.3f}, total={total_reward:.3f}, src={action_source}")
        
        if done:
            break
    
    episode_time = time.time() - episode_start_time
    
    # Count action sources
    model_actions = action_sources.count("TinyVLA Model")
    random_actions = action_sources.count("Random")
    
    result = {
        "episode": episode_num,
        "steps": step,
        "total_reward": total_reward,
        "success": success,
        "episode_time": episode_time,
        "model_actions": model_actions,
        "random_actions": random_actions,
        "final_info": dict(info) if info else {}
    }
    
    print(f"📊 Episode {episode_num} completed:")
    print(f"   Steps: {step}, Reward: {total_reward:.3f}, Success: {success}")
    print(f"   Model actions: {model_actions}, Random: {random_actions}")
    
    return result


def main():
    """🚀 Main Demo Function"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="MetaWorld Pick-Place Demo - Simple Sequential Approach")
    parser.add_argument('--episodes', type=int, default=3, help='Number of episodes to run (default: 3)')
    parser.add_argument('--max-steps', type=int, default=500, help='Maximum steps per episode (default: 500)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    args = parser.parse_args()
    
    print("🤖 MetaWorld Pick-Place Demo Starting (Simple Sequential Approach)...")
    print("=" * 70)
    print(f"Episodes: {args.episodes}, Max steps: {args.max_steps}, Seed: {args.seed}")
    print("=" * 70)
    
    # Set random seed
    np.random.seed(args.seed)
    
    try:
        # Load model (optional)
        model, model_info = load_model_safely()
        print(f"🤖 Model: {model_info}")
        
        # Setup MetaWorld environment (required)
        env = setup_metaworld_environment()
        print(f"🌍 Environment ready: MetaWorld pick-place-v2")
        
        # Note about MuJoCo window
        if 'DISPLAY' in os.environ:
            print("🖼️ MuJoCo GUI window will open - simple sequential rendering!")
        
        # Run episodes
        results = []
        total_start_time = time.time()
        
        for episode in range(1, args.episodes + 1):
            try:
                episode_result = run_episode(env, model, episode, args.max_steps)
                results.append(episode_result)
                
                # Brief pause between episodes
                time.sleep(1.0)
                
            except KeyboardInterrupt:
                print("\n⏹️ Demo interrupted by user")
                break
            except Exception as e:
                print(f"❌ Episode {episode} failed: {e}")
                continue
        
        # Summary
        total_time = time.time() - total_start_time
        
        if results:
            successes = sum(r['success'] for r in results)
            avg_reward = np.mean([r['total_reward'] for r in results])
            total_model_actions = sum(r['model_actions'] for r in results)
            total_random_actions = sum(r['random_actions'] for r in results)
            
            print(f"\n📊 FINAL SUMMARY:")
            print(f"   Episodes: {len(results)}, Success rate: {successes}/{len(results)}")
            print(f"   Average reward: {avg_reward:.3f}")
            print(f"   Model actions: {total_model_actions}, Random: {total_random_actions}")
            print(f"   Total time: {total_time:.1f}s")
        
        # Clean up
        if hasattr(env, 'close'):
            env.close()
            
    except KeyboardInterrupt:
        print("\n⏹️ Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1)
    
    print("\n🎉 Demo completed!")


if __name__ == "__main__":
    main() 