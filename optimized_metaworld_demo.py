# ==============================================================================
# IMPORTS - Loading all the tools we need
# ==============================================================================

import numpy as np  # For numerical computations and array operations
import metaworld     # The robot simulation environment
import time         # For timing measurements and delays
import os           # For environment variables

# GPU and display imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAS_MATPLOTLIB = True
    print("✅ Matplotlib available for GPU rendering display")
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  Matplotlib not available")

# Try to import our simplified TinyVLA model loader
try:
    from tinyvla_loader import load_tinyvla  # Our custom model loader
    HAS_TINYVLA = True                       # Flag to track if TinyVLA is available
    print("✅ TinyVLA loader available")
except ImportError as e:
    HAS_TINYVLA = False                      # TinyVLA not available
    print(f"⚠️  TinyVLA loader not available: {e}")

# ==============================================================================
# MUJOCO GPU CONFIGURATION
# ==============================================================================

# Set MuJoCo to use GPU-accelerated offscreen rendering
# This avoids the full-screen viewer issue entirely
os.environ['MUJOCO_GL'] = 'egl'  # Use EGL for GPU-accelerated offscreen rendering
print("🚀 Configuring MuJoCo for GPU-accelerated offscreen rendering")

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
# PROMPT ENGINEERING SYSTEM
# ==============================================================================

class TaskPromptGenerator:
    """
    🎯 Advanced Prompt Engineering for MetaWorld Tasks
    
    This class generates optimized, task-specific instructions for VLA models
    using different prompt engineering strategies and styles.
    """
    
    def __init__(self, prompt_style="detailed"):
        """
        Initialize the prompt generator
        
        Args:
            prompt_style: Style of prompts to generate
                - "simple": Basic task descriptions
                - "detailed": Comprehensive step-by-step instructions  
                - "imperative": Direct command style
                - "conversational": Natural language style
                - "technical": Precise robotic terminology
        """
        self.prompt_style = prompt_style
        
        # Task name mappings for MetaWorld environments
        self.task_mappings = {
            'pick-place-v2': 'pick_place',
            'pick-place': 'pick_place',
            'reach-v2': 'reach',
            'reach': 'reach',
            'push-v2': 'push', 
            'push': 'push',
            'door-open-v2': 'door_open',
            'door-close-v2': 'door_close',
            'drawer-open-v2': 'drawer_open',
            'drawer-close-v2': 'drawer_close',
            'button-press-v2': 'button_press',
            'peg-insert-side-v2': 'peg_insert',
            'window-open-v2': 'window_open',
            'window-close-v2': 'window_close'
        }
        
    def generate_prompt(self, task_name, prompt_style=None):
        """
        🎨 Generate optimized prompt for specific task
        
        Args:
            task_name: MetaWorld task name (e.g., 'pick-place-v2')
            prompt_style: Override default prompt style
            
        Returns:
            str: Optimized instruction prompt for the task
        """
        style = prompt_style or self.prompt_style
        
        # Map environment name to standard task name
        standard_task = self.task_mappings.get(task_name, 'pick_place')
        
        # Generate prompt based on task and style
        prompt_func = getattr(self, f'_generate_{standard_task}_prompt', self._generate_default_prompt)
        return prompt_func(style)
    
    def _generate_pick_place_prompt(self, style):
        """Generate pick-place task prompts"""
        prompts = {
            "simple": "Pick up the object and place it at the target location.",
            
            "detailed": "Carefully approach the red object, grasp it securely with the gripper, lift it up, navigate to the green target area, and gently place the object down at the target position.",
            
            "imperative": "PICK UP THE RED BLOCK. MOVE TO TARGET. PLACE BLOCK DOWN.",
            
            "conversational": "Please pick up the red block that you see and move it to the target location. Make sure to grasp it firmly but gently, then place it precisely on the target.",
            
            "technical": "Execute pick-and-place manipulation: (1) Navigate end-effector to object coordinates, (2) Close gripper around object, (3) Lift object above obstacles, (4) Navigate to target coordinates, (5) Lower and release object at target."
        }
        return prompts.get(style, prompts["detailed"])
    
    def _generate_reach_prompt(self, style):
        """Generate reach task prompts"""
        prompts = {
            "simple": "Reach to the target location.",
            "detailed": "Move the robot arm to reach the target position marked by the goal indicator.",
            "imperative": "REACH THE TARGET POSITION.",
            "conversational": "Please move your arm to reach the target location.",
            "technical": "Navigate end-effector to target coordinates without object manipulation."
        }
        return prompts.get(style, prompts["detailed"])
        
    def _generate_push_prompt(self, style):
        """Generate push task prompts"""
        prompts = {
            "simple": "Push the object to the target location.",
            "detailed": "Use the robot arm to push the object across the surface to the target area.",
            "imperative": "PUSH OBJECT TO TARGET.",
            "conversational": "Please push the object to move it to the target location.",
            "technical": "Apply lateral force to object to translate it to target coordinates."
        }
        return prompts.get(style, prompts["detailed"])
        
    def _generate_door_open_prompt(self, style):
        """Generate door opening prompts"""
        prompts = {
            "simple": "Open the door.",
            "detailed": "Grasp the door handle and pull it to open the door completely.",
            "imperative": "OPEN THE DOOR.",
            "conversational": "Please open the door by pulling the handle.",
            "technical": "Grasp door handle and apply rotational force to open door mechanism."
        }
        return prompts.get(style, prompts["detailed"])
        
    def _generate_button_press_prompt(self, style):
        """Generate button press prompts"""
        prompts = {
            "simple": "Press the button.",
            "detailed": "Locate the button and press it down with sufficient force to activate it.",
            "imperative": "PRESS THE BUTTON.",
            "conversational": "Please press the button to activate it.",
            "technical": "Apply downward force to button to trigger activation mechanism."
        }
        return prompts.get(style, prompts["detailed"])
        
    def _generate_default_prompt(self, style):
        """Fallback prompt for unknown tasks"""
        return "Complete the robotic manipulation task shown in the environment."
        
    def get_available_styles(self):
        """Get list of available prompt styles"""
        return ["simple", "detailed", "imperative", "conversational", "technical"]
        
    def benchmark_prompts(self, task_name):
        """Generate all prompt styles for comparison"""
        results = {}
        for style in self.get_available_styles():
            results[style] = self.generate_prompt(task_name, style)
        return results

# Create global prompt generator
prompt_generator = TaskPromptGenerator(prompt_style="detailed")  # Default to detailed prompts

def get_task_instruction(task_name, style=None):
    """
    🎯 Get optimized instruction for specific task
    
    Args:
        task_name: MetaWorld task name
        style: Prompt style override
        
    Returns:
        str: Task-specific instruction
    """
    return prompt_generator.generate_prompt(task_name, style)

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
print("\n🔍 Checking GPU rendering capabilities...")
print(f"   Environment type: {type(env).__name__}")              # Class name of environment
print(f"   Has sim: {hasattr(env, 'sim')}")                      # Does it have a simulator?
print(f"   Has _get_viewer: {hasattr(env, '_get_viewer')}")      # Does it have a viewer?
print(f"   Has unwrapped: {hasattr(env, 'unwrapped')}")          # Does it have an unwrapped version?
print(f"   MUJOCO_GL: {os.environ.get('MUJOCO_GL', 'default')}")  # GPU backend

if hasattr(env, 'unwrapped'):
    print(f"   Unwrapped type: {type(env.unwrapped).__name__}")
    print(f"   Unwrapped has sim: {hasattr(env.unwrapped, 'sim')}")

if hasattr(env, 'sim'):
    print(f"   Sim type: {type(env.sim).__name__}")
    print(f"   Has render: {hasattr(env.sim, 'render')}")

# Test GPU rendering to verify it works
print("\n🧪 Testing GPU offscreen rendering...")
try:
    test_rgb = env.render(offscreen=True)  # GPU-accelerated offscreen rendering
    if test_rgb is not None:
        print(f"   ✅ GPU offscreen rendering works: shape {test_rgb.shape if hasattr(test_rgb, 'shape') else type(test_rgb)}")
    else:
        print(f"   ❌ GPU offscreen rendering returns None")
except Exception as e:
    print(f"   ❌ GPU offscreen rendering failed: {e}")

# ==============================================================================
# PROMPT ENGINEERING DEMO
# ==============================================================================

# Demonstrate prompt engineering for current task
current_task = 'pick-place-v2'
print(f"\n🎯 Prompt Engineering Demo for '{current_task}':")
print("=" * 50)

# Show all available prompt styles
all_prompts = prompt_generator.benchmark_prompts(current_task)
for style, prompt in all_prompts.items():
    print(f"📝 {style.upper():>12}: {prompt}")

# Select the prompt style to use (you can change this!)
SELECTED_PROMPT_STYLE = "detailed"  # Options: simple, detailed, imperative, conversational, technical
selected_instruction = get_task_instruction(current_task, SELECTED_PROMPT_STYLE)
print(f"\n🎯 SELECTED STYLE: {SELECTED_PROMPT_STYLE}")
print(f"🤖 INSTRUCTION: {selected_instruction}")

print("🎮 Starting GPU-accelerated simulation...")
print("="*50)

# ==============================================================================
# MATPLOTLIB DISPLAY FUNCTIONS (CONTROLLED WINDOW SIZE)
# ==============================================================================

class GPURenderDisplay:
    """
    🖥️ GPU-Accelerated Rendering Display with Controlled Window Size
    
    This class handles displaying GPU-rendered frames in a matplotlib window
    with controlled size (no full screen issues).
    """
    
    def __init__(self, window_width=800, window_height=600):
        """Initialize the display window with specific size"""
        self.window_width = window_width
        self.window_height = window_height
        self.fig = None
        self.ax = None
        self.im = None
        self.setup_complete = False
        
    def setup_display(self):
        """Setup matplotlib display window"""
        if not HAS_MATPLOTLIB:
            return False
            
        try:
            # Create figure with specific size (in inches, DPI=100)
            self.fig, self.ax = plt.subplots(figsize=(self.window_width/100, self.window_height/100), dpi=100)
            self.ax.set_title("🚀 TinyVLA Demo - GPU Accelerated MetaWorld")
            self.ax.axis('off')  # Hide axes for cleaner look
            
            # Set window position and size
            mngr = self.fig.canvas.manager
            mngr.window.wm_geometry(f"{self.window_width}x{self.window_height}+100+100")
            
            plt.ion()  # Interactive mode for real-time updates
            self.setup_complete = True
            return True
            
        except Exception as e:
            print(f"⚠️ Could not setup matplotlib display: {e}")
            return False
    
    def update_display(self, rgb_frame, step, reward, total_reward, success=False):
        """Update display with new frame"""
        if not self.setup_complete:
            return
            
        try:
            # First frame - create image plot
            if self.im is None:
                self.im = self.ax.imshow(rgb_frame)
            else:
                # Update existing image
                self.im.set_array(rgb_frame)
            
            # Update title with performance info and prompt style
            title = f"🚀 {MODEL_TYPE} | Step: {step} | Reward: {reward:.3f} | Total: {total_reward:.3f} | Prompt: {SELECTED_PROMPT_STYLE}"
            if success:
                title = f"🎉 SUCCESS! {title}"
            self.ax.set_title(title, fontsize=10)
            
            # Refresh display
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)  # Very small pause for rendering
            
        except Exception as e:
            pass  # Continue if display fails
    
    def close(self):
        """Close the display window"""
        if self.fig:
            plt.close(self.fig)

# ==============================================================================
# ACTION PREDICTION FUNCTIONS
# ==============================================================================

def get_action(rgb_frame, robot_state, task_name='pick-place-v2', prompt_style=None):
    """
    🎯 Get Action from VLA Model or Random Policy with Task-Specific Prompts
    
    This function decides what action the robot should take next. It uses
    sophisticated prompt engineering to generate task-specific instructions.
    
    Args:
        rgb_frame: What the robot sees (numpy array)
        robot_state: Current robot joint positions (numpy array)
        task_name: MetaWorld task name for prompt generation
        prompt_style: Override prompt style
        
    Returns:
        action: Robot movement command [x, y, z, gripper] (numpy array)
    """
    # Generate task-specific instruction using prompt engineering
    instruction = get_task_instruction(task_name, prompt_style)
    
    if vla_model is not None:
        # We have a TinyVLA model available - try to use it
        try:
            from PIL import Image
            action = vla_model.predict_action(Image.fromarray(rgb_frame), robot_state, instruction)
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
    🚀 Run the Pick-Place Demo with GPU-Accelerated Rendering and Advanced Prompts
    
    This is the main function that runs the entire demonstration. It:
    1. Uses GPU-accelerated offscreen rendering for maximum performance
    2. Uses advanced prompt engineering for task-specific instructions
    3. Displays results in a controlled matplotlib window (no full screen)
    4. Runs the simulation loop with optimized prompts
    5. Collects and displays performance metrics
    """
    
    # ====================================================================
    # SETUP PHASE
    # ====================================================================
    
    print("🚀 Setting up GPU-accelerated rendering with controlled window...")
    
    # Setup GPU rendering display
    display = GPURenderDisplay(window_width=800, window_height=600)
    display_ready = display.setup_display()
    
    if display_ready:
        print("✅ GPU rendering display ready (800x600 window)")
    else:
        print("⚠️ Display not available - running headless")
    
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
    print(f"🚀 Starting GPU-accelerated demo with {SELECTED_PROMPT_STYLE} prompts...")
    print("💨 Optimizations: GPU offscreen rendering, controlled window, advanced prompts")
    print("=" * 60)
    
    # ====================================================================
    # MAIN SIMULATION LOOP
    # ====================================================================
    
    while step < 500:  # Run for maximum 500 steps
        
        # ================================================================
        # STEP 1: GPU-Accelerated Rendering
        # ================================================================
        
        # Get GPU-rendered RGB frame (what robot sees)
        rgb_frame = None
        try:
            rgb_frame = env.render(offscreen=True)  # GPU-accelerated rendering
            if rgb_frame is None:
                rgb_frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Fallback
        except Exception as e:
            print(f"Warning: GPU rendering failed: {e}")
            rgb_frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Fallback
        
        # Extract robot state (joint positions, etc.)
        robot_state = obs[:7]  # Use first 7 dimensions for robot state
        
        # ================================================================
        # STEP 2: Advanced Prompt-Based Action Prediction
        # ================================================================
        
        # Get action using task-specific prompt engineering
        action = get_action(rgb_frame, robot_state, current_task, SELECTED_PROMPT_STYLE)
        
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
        # STEP 4: Display Update (Controlled Window)
        # ================================================================
        
        # Check if task was completed successfully
        success = info.get('success', False)
        
        # 🚀 OPTIMIZATION: Update display every few frames for performance
        if step % 2 == 0 and display_ready:  # Update every 2 steps
            display.update_display(rgb_frame, step, reward, total_reward, success)
        
        # Print progress every 25 steps (reduced frequency for speed)
        if step % 25 == 0:
            model_info = "VLA" if vla_model is not None else "Random"
            print(f"[{model_info}] Step {step:3d}: reward={reward:6.3f}, total={total_reward:6.3f}, success={success} | Prompt: {SELECTED_PROMPT_STYLE}")
        
        # ================================================================
        # STEP 5: Check for Completion
        # ================================================================
        
        # Check for successful task completion
        if success:
            print(f"🎉 SUCCESS! Task completed in {step} steps!")
            print(f"🏆 Total Reward: {total_reward:.3f}")
            print(f"🎯 Used Prompt Style: {SELECTED_PROMPT_STYLE}")
            # Final display update
            if display_ready:
                display.update_display(rgb_frame, step, reward, total_reward, success)
            time.sleep(2)  # Show success for 2 seconds
            break
        
        # Check if episode ended for other reasons
        if done:
            print(f"Episode ended at step {step}")
            break
        
        # 🚀 OPTIMIZATION: Minimal delay for maximum GPU performance
        # GPU rendering is much faster, so we can have smaller delays
        time.sleep(0.005)  # Very small delay for maximum performance
    
    # ====================================================================
    # RESULTS AND CLEANUP
    # ====================================================================
    
    # Calculate and display final performance metrics
    print(f"\n📊 Final Results:")
    print(f"   Model: {MODEL_TYPE}")
    print(f"   Prompt Style: {SELECTED_PROMPT_STYLE}")
    print(f"   Instruction: {selected_instruction}")
    print(f"   Steps: {step}")
    print(f"   Total Reward: {total_reward:.3f}")
    print(f"   Average Reward: {total_reward/step:.3f}")
    print(f"   Success: {'✅ Yes' if info.get('success', False) else '❌ No'}")
    
    # 🚀 PERFORMANCE METRICS
    elapsed_time = time.time() - start_time
    fps = step / elapsed_time if elapsed_time > 0 else 0
    print(f"   🚀 Performance: {fps:.1f} FPS ({elapsed_time:.1f}s total)")
    print(f"   💨 Rendering: GPU-accelerated offscreen (EGL backend)")
    print(f"   🖥️ Display: Controlled matplotlib window (800x600)")
    
    # Model-specific information
    if vla_model is not None:
        print(f"   🧠 TinyVLA model was used for action prediction")
        print(f"   📊 Normalization stats loaded from: {vla_model.stats_path}")
        print(f"   🎯 Model status: {'✅ Loaded' if vla_model.model_loaded else '❌ Failed'}")
    else:
        print(f"   🎲 Random actions were used (TinyVLA not available)")
    
    # Cleanup
    if display_ready:
        print("🖥️ Keeping display window open for 3 seconds...")
        time.sleep(3)
        display.close()


# ==============================================================================
# MAIN EXECUTION - Entry point when script is run
# ==============================================================================

if __name__ == "__main__":
    """
    Main execution block - runs when script is executed directly.
    
    This handles:
    - Running the demo with advanced prompt engineering
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
            
    except Exception as e:
        # Something went wrong
        print(f"\n❌ Error: {e}")
        print("💡 Make sure you have conda environment activated: conda activate tinyvla")
        print("💡 For GPU acceleration, ensure you have proper OpenGL/EGL support") 