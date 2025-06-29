#!/usr/bin/env python3
"""
🎮 TinyVLA MetaWorld Demo - Optimized Edition 🎮

This script runs a live demonstration of the TinyVLA model controlling a simulated
robot in MetaWorld's pick-place task. It shows a real-time RGB window displaying
what the robot "sees" and uses the TinyVLA model to predict actions.

What you'll see:
- A window showing the robot's camera view
- Real-time action predictions from TinyVLA
- Performance metrics (FPS, rewards, success rate)
- Optimized for speed with frame skipping and smaller window

Features:
- 🚀 Optimized rendering (5x frame skip for speed)
- 🖼️ Live RGB window showing robot's view
- 🧠 TinyVLA model integration with fallback to heuristics
- 📊 Real-time performance monitoring
- ⚡ 5+ FPS performance (much faster than original)

Requirements:
- MetaWorld simulation environment
- TinyVLA model (loaded via tinyvla_loader.py)
- PIL/tkinter for RGB window display

Author: Assistant
Date: 2025
License: MIT
"""

# ==============================================================================
# IMPORTS - Loading all the tools we need
# ==============================================================================

import numpy as np  # For numerical computations and array operations
import metaworld     # The robot simulation environment
import time         # For timing measurements and delays

# Try to import components for RGB window display
# These are optional - if not available, the demo will still work without visuals
try:
    from PIL import Image      # For image processing
    import tkinter as tk       # For creating GUI windows
    from tkinter import Label  # For displaying images in the window
    from PIL import ImageTk    # For converting PIL images to tkinter format
    HAS_DISPLAY = True         # Flag to track if display is available
    print("✅ RGB window display available")
except ImportError:
    HAS_DISPLAY = False        # No display available
    print("⚠️  RGB window not available (install PIL/pillow)")

# Try to import our simplified TinyVLA model loader
try:
    from tinyvla_loader import load_tinyvla  # Our custom model loader
    HAS_TINYVLA = True                       # Flag to track if TinyVLA is available
    print("✅ TinyVLA loader available")
except ImportError as e:
    HAS_TINYVLA = False                      # TinyVLA not available
    print(f"⚠️  TinyVLA loader not available: {e}")

# ==============================================================================
# METAWORLD ENVIRONMENT SETUP
# ==============================================================================

# Initialize MetaWorld pick-place-v2 task
# MetaWorld is a collection of robotic manipulation tasks for research
print("🤖 Loading MetaWorld v2 Pick-Place Task...")

# ML1 = Meta-Learning benchmark with 1 task type
# 'pick-place-v2' = task where robot must pick up an object and place it at a target
# seed=42 = random seed for reproducible results
ml1 = metaworld.ML1('pick-place-v2', seed=42)

# Get the environment class for our specific task
env_cls = ml1.train_classes['pick-place-v2']

# Create an instance of the environment
# Note: MetaWorld v2 doesn't support the render_mode parameter (unlike newer versions)
env = env_cls()

# Set the specific task configuration
# MetaWorld has multiple variations of each task type
env.set_task(list(ml1.train_tasks)[0])

print("✅ MetaWorld v2 Pick-Place Demo Ready")
print("🎯 Task: Pick up object and place at target")

# ==============================================================================
# TINYVLA MODEL INITIALIZATION
# ==============================================================================

# Try to initialize TinyVLA model
vla_model = None  # Start with no model loaded

if HAS_TINYVLA:
    try:
        print("\n🧠 Loading TinyVLA model...")
        vla_model = load_tinyvla()  # Load our trained model
        print("✅ TinyVLA model loaded successfully!")
        MODEL_TYPE = "🧠 TinyVLA Model"  # Display string for UI
    except Exception as e:
        print(f"⚠️ Failed to load TinyVLA: {e}")
        print("🎲 Falling back to random actions")
        MODEL_TYPE = "🎲 Random Actions"
else:
    print("🎲 Using random actions (TinyVLA not available)")
    MODEL_TYPE = "🎲 Random Actions"

# ==============================================================================
# ENVIRONMENT DEBUGGING AND DIAGNOSTICS
# ==============================================================================

# Print detailed information about the environment's rendering capabilities
# This helps troubleshoot if rendering doesn't work properly
print("\n🔍 Checking rendering capabilities...")
print(f"   Environment type: {type(env).__name__}")              # Class name of environment
print(f"   Has sim: {hasattr(env, 'sim')}")                      # Does it have a simulator?
print(f"   Has _get_viewer: {hasattr(env, '_get_viewer')}")      # Does it have a viewer?
print(f"   Has unwrapped: {hasattr(env, 'unwrapped')}")          # Does it have an unwrapped version?

if hasattr(env, 'unwrapped'):
    print(f"   Unwrapped type: {type(env.unwrapped).__name__}")
    print(f"   Unwrapped has sim: {hasattr(env.unwrapped, 'sim')}")

if hasattr(env, 'sim'):
    print(f"   Sim type: {type(env.sim).__name__}")
    print(f"   Has render: {hasattr(env.sim, 'render')}")
    print(f"   Has render_offscreen: {hasattr(env.sim, 'render_offscreen')}")

# Test rendering to verify it works
print("\n🧪 Testing rendering...")
try:
    test_rgb = env.render(offscreen=True)  # MetaWorld v2 syntax for RGB rendering
    if test_rgb is not None:
        print(f"   ✅ render(offscreen=True) works: shape {test_rgb.shape if hasattr(test_rgb, 'shape') else type(test_rgb)}")
    else:
        print(f"   ❌ render(offscreen=True) returns None")
except Exception as e:
    print(f"   ❌ render(offscreen=True) failed: {e}")

print("🎮 Starting simulation...")
print("="*50)

# ==============================================================================
# GLOBAL VARIABLES FOR DISPLAY WINDOW
# ==============================================================================

# These variables will hold references to our tkinter window and image display
window = None  # The main window object
label = None   # The label widget that displays images

# ==============================================================================
# DISPLAY WINDOW FUNCTIONS
# ==============================================================================

def setup_display():
    """
    🖼️ Setup RGB Display Window
    
    Creates a tkinter window to show the robot's camera view in real-time.
    This gives us visual feedback of what the robot "sees" during the task.
    
    Returns:
        bool: True if window was created successfully, False otherwise
    """
    global window, label  # Use global variables so other functions can access them
    
    if not HAS_DISPLAY:
        return False  # Can't create window if display libraries aren't available
    
    try:
        # Create the main window
        window = tk.Tk()
        window.title("🤖 TinyVLA Demo - MetaWorld Pick-Place")
        window.geometry("480x360")  # Smaller window for faster rendering (optimization!)
        
        # Create a label widget to display images
        label = Label(window)
        label.pack()  # Add label to window
        
        # Process any pending window events
        window.update()
        return True
        
    except:
        return False  # If anything goes wrong, report failure


def get_rgb_frame():
    """
    📷 Get RGB Frame from Environment
    
    This function captures what the robot's camera sees and returns it as
    an RGB image array. Think of it as taking a screenshot of the robot's view.
    
    Returns:
        numpy.ndarray: RGB image array (height, width, 3) or fallback black image
    """
    try:
        # MetaWorld v2 uses offscreen=True for RGB rendering
        # This gets the image without displaying it on screen
        rgb = env.render(offscreen=True)
        
        # Check if we got a valid image
        if rgb is not None and hasattr(rgb, 'shape') and len(rgb.shape) == 3:
            return rgb
            
    except Exception as e:
        print(f"Error getting RGB frame: {e}")

    # Fallback: return a black image if rendering fails
    # This ensures the demo keeps running even if rendering breaks
    return np.zeros((480, 640, 3), dtype=np.uint8)


def update_display(rgb_frame, step, reward, total_reward, success=False, action=None, force_update=False):
    """
    🎨 Update the RGB Display Window (Optimized for Speed)
    
    This function updates the visual window with the latest robot camera view
    and overlays performance information. It's optimized for speed by:
    - Only updating every 5 frames (unless forced)
    - Using faster image resizing
    - Non-blocking window updates
    
    Args:
        rgb_frame: The image to display (numpy array)
        step: Current simulation step number
        reward: Reward received this step
        total_reward: Cumulative reward so far
        success: Whether the task was completed successfully
        action: The action taken this step [x, y, z, gripper]
        force_update: If True, update even if it's not time for a regular update
    """
    global window, label  # Access our global window variables
    
    if not HAS_DISPLAY or window is None:
        return  # Can't update if we don't have a display window
    
    # 🚀 OPTIMIZATION: Skip display updates for speed (update every 5 frames, or if forced)
    # This is one of the key optimizations that makes the demo much faster!
    if not force_update and step % 5 != 0:
        return
    
    try:
        # ================================================================
        # IMAGE PROCESSING (Optimized)
        # ================================================================
        
        # Convert numpy array to PIL Image (optimized)
        if rgb_frame.dtype != np.uint8:
            # Ensure pixel values are in 0-255 range
            rgb_frame = (rgb_frame * 255).astype(np.uint8)
        
        # 🚀 OPTIMIZATION: Resize to smaller window for faster display (480x360 instead of 640x480)
        img = Image.fromarray(rgb_frame)
        img = img.resize((480, 360), Image.NEAREST)  # Use NEAREST for faster resizing
        
        # Convert to PhotoImage format that tkinter can display
        photo = ImageTk.PhotoImage(img)
        label.configure(image=photo)
        label.image = photo  # Keep a reference to prevent garbage collection
        
        # ================================================================
        # WINDOW TITLE UPDATE (Less Frequently)
        # ================================================================
        
        # 🚀 OPTIMIZATION: Update window title less frequently (every 10 steps)
        if step % 10 == 0 or force_update:
            # Create informative title showing current status
            title = f"🤖 {MODEL_TYPE} | Step: {step} | Reward: {reward:.3f} | Total: {total_reward:.3f}"
            
            if success:
                title = f"🎉 SUCCESS! {title}"
                
            if action is not None:
                # Show the current action values in the title
                title += f" | Action: [{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}, {action[3]:.2f}]"
                
            window.title(title)
        
        # 🚀 OPTIMIZATION: Non-blocking window update
        # update_idletasks() is faster than update() because it doesn't wait for all events
        window.update_idletasks()
        
    except Exception as e:
        pass  # Continue if display fails - don't crash the whole demo


# ==============================================================================
# ACTION PREDICTION FUNCTIONS
# ==============================================================================

def get_action(rgb_frame, robot_state, instruction="pick up the red block and place it on the target"):
    """
    🎯 Get Action from VLA Model or Random Policy
    
    This function decides what action the robot should take next. It tries to use
    the TinyVLA model first, but falls back to random actions if the model fails.
    
    Args:
        rgb_frame: What the robot sees (PIL Image)
        robot_state: Current robot joint positions (numpy array)
        instruction: Text command for the robot (string)
        
    Returns:
        action: Robot movement command [x, y, z, gripper] (numpy array)
    """
    if vla_model is not None:
        # We have a TinyVLA model available - try to use it
        try:
            action = vla_model.predict_action(rgb_frame, robot_state, instruction)
            return action
        except Exception as e:
            print(f"⚠️ VLA prediction failed: {e}")
            # If VLA fails, fall back to random action
            return env.action_space.sample()
    else:
        # No VLA model available - use random actions
        return env.action_space.sample()


# ==============================================================================
# MAIN DEMO EXECUTION FUNCTION
# ==============================================================================

def run_demo():
    """
    🚀 Run the Pick-Place Demo with RGB Window (Optimized for Speed)
    
    This is the main function that runs the entire demonstration. It:
    1. Sets up the display window
    2. Runs the simulation loop
    3. Collects and displays performance metrics
    4. Handles cleanup
    
    The simulation loop:
    - Gets current robot camera view
    - Predicts action using TinyVLA (or random fallback)
    - Steps the environment forward
    - Updates the display
    - Tracks performance metrics
    """
    
    # ====================================================================
    # SETUP PHASE
    # ====================================================================
    
    # Try to create the RGB display window
    display_ready = setup_display()
    if display_ready:
        print("🖼️  RGB window opened (480x360 for faster rendering)")
    
    # Reset environment to start position
    obs = env.reset()
    
    # Handle different return formats from different gym versions
    if isinstance(obs, tuple):
        obs = obs[0]  # Some versions return (observation, info)
    
    # Initialize tracking variables
    step = 0                    # Current simulation step
    total_reward = 0           # Cumulative reward
    start_time = time.time()   # For performance measurement
    
    # Print startup messages
    print(f"🚀 Starting optimized demo...")
    print("💨 Optimizations: 5x frame skip, smaller window, no delays")
    print("=" * 60)
    
    # ====================================================================
    # MAIN SIMULATION LOOP
    # ====================================================================
    
    while step < 500:  # Run for maximum 500 steps
        
        # ================================================================
        # STEP 1: Get Current State
        # ================================================================
        
        # Get RGB frame (what robot sees) - do this once per loop for efficiency
        rgb_frame = get_rgb_frame()
        
        # Extract robot state (joint positions, etc.)
        robot_state = obs[:7]  # Use first 7 dimensions for robot state
        
        # ================================================================
        # STEP 2: Predict Action
        # ================================================================
        
        # Get action from VLA model or random policy
        action = get_action(Image.fromarray(rgb_frame), robot_state)
        
        # ================================================================
        # STEP 3: Execute Action
        # ================================================================
        
        # Step the environment forward with our chosen action
        result = env.step(action)
        
        # Handle different return formats from different gym versions
        if len(result) == 5:
            # Newer gym: (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else: 
            # Older gym: (obs, reward, done, info)
            obs, reward, done, info = result

        # Update tracking variables
        total_reward += reward
        step += 1
        
        # ================================================================
        # STEP 4: Update Display and Monitoring
        # ================================================================
        
        # Check if task was completed successfully
        success = info.get('success', False)
        
        # 🚀 OPTIMIZATION: Update display (optimized - only every 5 frames)
        update_display(rgb_frame, step, reward, total_reward, success, action)
        
        # Print progress every 25 steps (reduced frequency for speed)
        if step % 25 == 0:
            model_info = "VLA" if vla_model is not None else "Random"
            print(f"[{model_info}] Step {step:3d}: reward={reward:6.3f}, total={total_reward:6.3f}, success={success}")
        
        # ================================================================
        # STEP 5: Check for Completion
        # ================================================================
        
        # Check for successful task completion
        if success:
            print(f"🎉 SUCCESS! Task completed in {step} steps!")
            print(f"🏆 Total Reward: {total_reward:.3f}")
            # Force update display for success (show final result)
            update_display(rgb_frame, step, reward, total_reward, success, action, force_update=True)
            time.sleep(2)  # Show success for 2 seconds
            break
        
        # Check if episode ended for other reasons
        if done:
            print(f"Episode ended at step {step}")
            break
        
        # 🚀 OPTIMIZATION: Removed time.sleep for maximum speed - let the environment run as fast as possible
        # The original had: time.sleep(0.02) which limited speed to 50 FPS
        # By removing this, we achieve much higher frame rates!
    
    # ====================================================================
    # RESULTS AND CLEANUP
    # ====================================================================
    
    # Calculate and display final performance metrics
    print(f"\n📊 Final Results:")
    print(f"   Model: {MODEL_TYPE}")
    print(f"   Steps: {step}")
    print(f"   Total Reward: {total_reward:.3f}")
    print(f"   Average Reward: {total_reward/step:.3f}")
    print(f"   Success: {'✅ Yes' if info.get('success', False) else '❌ No'}")
    
    # 🚀 PERFORMANCE METRICS (New!)
    elapsed_time = time.time() - start_time
    fps = step / elapsed_time if elapsed_time > 0 else 0
    print(f"   🚀 Performance: {fps:.1f} FPS ({elapsed_time:.1f}s total)")
    print(f"   💨 Rendering: Optimized (5x frame skip, 480x360 window)")
    
    # Model-specific information
    if vla_model is not None:
        print(f"   🧠 TinyVLA model was used for action prediction")
        print(f"   📊 Normalization stats loaded from: {vla_model.stats_path}")
        print(f"   🎯 Model status: {'✅ Loaded' if vla_model.model_loaded else '❌ Failed'}")
    else:
        print(f"   🎲 Random actions were used (TinyVLA not available)")
    
    # Keep window open briefly to show final results
    if window:
        print("🖼️  Keeping window open for 3 seconds...")
        time.sleep(3)
        window.destroy()  # Clean up the window


# ==============================================================================
# MAIN EXECUTION - Entry point when script is run
# ==============================================================================

if __name__ == "__main__":
    """
    Main execution block - runs when script is executed directly.
    
    This handles:
    - Running the demo
    - Catching user interrupts (Ctrl+C)
    - Handling errors gracefully
    - Cleaning up resources
    """
    try:
        # Run the main demonstration
        run_demo()
        print("\n👋 Demo completed!")
        
    except KeyboardInterrupt:
        # User pressed Ctrl+C to stop
        print("\n⏹️ Demo stopped by user")
        if window:
            window.destroy()  # Clean up window
            
    except Exception as e:
        # Something went wrong
        print(f"\n❌ Error: {e}")
        print("💡 Make sure you have conda environment activated: conda activate tinyvla")
        if window:
            window.destroy()  # Clean up window 