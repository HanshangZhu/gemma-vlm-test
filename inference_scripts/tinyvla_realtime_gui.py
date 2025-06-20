#!/usr/bin/env python3
"""
TinyVLA Real-Time GUI
Interactive interface for testing TinyVLA diffusion policy with live visualization
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import cv2
from PIL import Image, ImageTk

# Add parent directory to path for unified_tinyvla
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set rendering mode for headless if needed
if os.environ.get('DISPLAY') is None:
    os.environ['MUJOCO_GL'] = 'osmesa'
else:
    os.environ['MUJOCO_GL'] = 'glfw'

import torch
from transformers import AutoTokenizer, CLIPImageProcessor
from unified_tinyvla import UnifiedTinyVLAModel

# MetaWorld imports
import metaworld
import random

class TinyVLAGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("TinyVLA Real-Time Inference GUI")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.model = None
        self.env = None
        self.is_running = False
        self.current_episode = 0
        self.step_count = 0
        self.total_reward = 0.0
        self.frame_queue = queue.Queue(maxsize=10)
        self.reward_history = []
        self.step_history = []
        
        # Model settings
        self.model_path = "VLM_weights/Llava-Pythia-400M"
        self.checkpoint_path = "checkpoints/TinyVLA-droid_diffusion_metaworld/diff_head_FIXED_epoch_40.pth"
        
        self.setup_gui()
        self.load_model()
        
    def setup_gui(self):
        """Setup the GUI layout"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Task selection
        ttk.Label(control_frame, text="Task:").pack(anchor=tk.W)
        self.task_var = tk.StringVar(value="pick-place-v2")
        task_combo = ttk.Combobox(control_frame, textvariable=self.task_var, width=20)
        task_combo['values'] = [
            'pick-place-v2', 'reach-v2', 'button-press-topdown-v2',
            'door-open-v2', 'drawer-open-v2', 'push-v2'
        ]
        task_combo.pack(fill=tk.X, pady=(0, 10))
        
        # Prompt input
        ttk.Label(control_frame, text="Prompt:").pack(anchor=tk.W)
        self.prompt_var = tk.StringVar(value="Pick up the object and place it at the target location")
        prompt_entry = ttk.Entry(control_frame, textvariable=self.prompt_var, width=30)
        prompt_entry.pack(fill=tk.X, pady=(0, 10))
        
        # Quick prompt buttons
        ttk.Label(control_frame, text="Quick Prompts:").pack(anchor=tk.W)
        quick_prompts = [
            ("Simple", "pick and place"),
            ("Detailed", "Carefully grasp the red object and precisely place it on the green target zone"),
            ("Reach", "Reach to the target position"),
            ("Button", "Press the button"),
            ("Minimal", "reach")
        ]
        
        for name, prompt in quick_prompts:
            btn = ttk.Button(control_frame, text=name, 
                           command=lambda p=prompt: self.prompt_var.set(p))
            btn.pack(fill=tk.X, pady=1)
        
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Episode settings
        ttk.Label(control_frame, text="Max Steps:").pack(anchor=tk.W)
        self.max_steps_var = tk.IntVar(value=100)
        steps_spin = ttk.Spinbox(control_frame, from_=50, to=200, textvariable=self.max_steps_var, width=10)
        steps_spin.pack(anchor=tk.W, pady=(0, 10))
        
        # Control buttons
        self.start_btn = ttk.Button(control_frame, text="Start Episode", command=self.start_episode)
        self.start_btn.pack(fill=tk.X, pady=2)
        
        self.stop_btn = ttk.Button(control_frame, text="Stop", command=self.stop_episode, state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, pady=2)
        
        self.reset_btn = ttk.Button(control_frame, text="Reset Environment", command=self.reset_environment)
        self.reset_btn.pack(fill=tk.X, pady=2)
        
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Status display
        ttk.Label(control_frame, text="Status:").pack(anchor=tk.W)
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(control_frame, textvariable=self.status_var, foreground="blue")
        status_label.pack(anchor=tk.W)
        
        ttk.Label(control_frame, text="Episode:").pack(anchor=tk.W)
        self.episode_var = tk.StringVar(value="0")
        ttk.Label(control_frame, textvariable=self.episode_var).pack(anchor=tk.W)
        
        ttk.Label(control_frame, text="Step:").pack(anchor=tk.W)
        self.step_var = tk.StringVar(value="0")
        ttk.Label(control_frame, textvariable=self.step_var).pack(anchor=tk.W)
        
        ttk.Label(control_frame, text="Reward:").pack(anchor=tk.W)
        self.reward_var = tk.StringVar(value="0.000")
        ttk.Label(control_frame, textvariable=self.reward_var).pack(anchor=tk.W)
        
        # Right panel - Visualization
        viz_frame = ttk.LabelFrame(main_frame, text="Live Visualization", padding=10)
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure for robot view
        self.fig, (self.ax_robot, self.ax_reward) = plt.subplots(2, 1, figsize=(8, 10))
        self.fig.tight_layout(pad=3.0)
        
        # Robot camera view
        self.ax_robot.set_title("Robot Camera View")
        self.ax_robot.axis('off')
        self.robot_im = self.ax_robot.imshow(np.zeros((240, 320, 3), dtype=np.uint8))
        
        # Reward plot
        self.ax_reward.set_title("Reward Over Time")
        self.ax_reward.set_xlabel("Step")
        self.ax_reward.set_ylabel("Cumulative Reward")
        self.reward_line, = self.ax_reward.plot([], [], 'b-', linewidth=2)
        self.ax_reward.grid(True, alpha=0.3)
        
        # Embed matplotlib in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Animation for live updates
        self.anim = FuncAnimation(self.fig, self.update_plots, interval=100, blit=False)
        
    def load_model(self):
        """Load the TinyVLA model"""
        try:
            self.status_var.set("Loading model...")
            self.root.update()
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = UnifiedTinyVLAModel(self.model_path, mode="action").to(device)
            
            # Load checkpoint
            if os.path.exists(self.checkpoint_path):
                checkpoint = torch.load(self.checkpoint_path, map_location=device)
                checkpoint = {k.replace('_orig_mod.', ''): v for k, v in checkpoint.items()}
                self.model.base_model.embed_out.load_state_dict(checkpoint)
                self.model.eval()
                
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.image_processor = CLIPImageProcessor.from_pretrained(self.model_path)
            
            self.status_var.set("Model loaded successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.status_var.set("Model loading failed")
    
    def setup_environment(self, task_name):
        """Setup MetaWorld environment"""
        try:
            # Map v2 to v3 tasks
            v2_to_v3_mapping = {
                'pick-place-v2': 'pick-place-v3',
                'door-open-v2': 'door-open-v3',
                'drawer-open-v2': 'drawer-open-v3',
                'button-press-topdown-v2': 'button-press-topdown-v3',
                'reach-v2': 'reach-v3',
                'push-v2': 'push-v3'
            }
            
            if task_name in v2_to_v3_mapping:
                v3_task_name = v2_to_v3_mapping[task_name]
            else:
                v3_task_name = task_name
                
            benchmark = metaworld.ML1(v3_task_name)
            self.env = benchmark.train_classes[v3_task_name]()
            task = random.choice(benchmark.train_tasks)
            self.env.set_task(task)
            
            if hasattr(self.env, 'env'):
                self.env = self.env.env
            self.env.render_mode = 'rgb_array'
            self.env.camera_name = 'corner'
            
            return True
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to setup environment: {str(e)}")
            return False
    
    def predict_action(self, rgb, prompt, robot_state=None):
        """Predict action using the model"""
        try:
            model_dtype = next(self.model.parameters()).dtype
            
            # Process image
            pil_img = Image.fromarray(rgb.astype(np.uint8))
            img_tensor = self.image_processor(pil_img, return_tensors="pt")["pixel_values"]
            img_tensor = img_tensor.to(self.model.base_model.device, dtype=model_dtype)
            
            # Process text
            tokens = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            tokens = {k: v.to(self.model.base_model.device) for k, v in tokens.items()}
            
            # Process state
            if robot_state is None:
                states = torch.zeros((1, 7), device=self.model.base_model.device, dtype=model_dtype)
            else:
                states = torch.tensor(robot_state, device=self.model.base_model.device, dtype=model_dtype).unsqueeze(0)
            
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
            print(f"Prediction error: {e}")
            return np.zeros(4, dtype=np.float32)
    
    def run_episode_thread(self):
        """Run episode in separate thread"""
        try:
            task_name = self.task_var.get()
            prompt = self.prompt_var.get()
            max_steps = self.max_steps_var.get()
            
            if not self.setup_environment(task_name):
                return
                
            obs, info = self.env.reset()
            self.total_reward = 0.0
            self.step_count = 0
            self.reward_history = []
            self.step_history = []
            
            for step in range(max_steps):
                if not self.is_running:
                    break
                    
                # Get camera image
                try:
                    rgb = self.env.render()
                    if rgb is not None:
                        # Add frame to queue for display
                        if not self.frame_queue.full():
                            self.frame_queue.put(rgb.copy())
                except:
                    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
                
                # Get robot state
                state = obs[:7] if len(obs) >= 7 else None
                
                # Predict action
                action = self.predict_action(rgb, prompt, state)
                
                # Execute action
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                self.total_reward += reward
                self.step_count = step + 1
                
                # Update history
                self.reward_history.append(self.total_reward)
                self.step_history.append(step)
                
                # Update GUI
                self.root.after(0, self.update_status)
                
                # Check for success or completion
                if done or info.get("success", False):
                    if info.get("success", False):
                        self.status_var.set(f"SUCCESS! Completed in {step+1} steps")
                    else:
                        self.status_var.set(f"Episode finished in {step+1} steps")
                    break
                    
                # Small delay for visualization
                time.sleep(0.05)
                
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Episode error: {str(e)}"))
        finally:
            self.is_running = False
            self.root.after(0, self.episode_finished)
    
    def start_episode(self):
        """Start a new episode"""
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded!")
            return
            
        self.is_running = True
        self.current_episode += 1
        self.episode_var.set(str(self.current_episode))
        
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_var.set("Running episode...")
        
        # Start episode in separate thread
        thread = threading.Thread(target=self.run_episode_thread)
        thread.daemon = True
        thread.start()
    
    def stop_episode(self):
        """Stop current episode"""
        self.is_running = False
        self.status_var.set("Stopping...")
    
    def episode_finished(self):
        """Called when episode finishes"""
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        
        if not self.status_var.get().startswith("SUCCESS"):
            self.status_var.set("Episode stopped")
    
    def reset_environment(self):
        """Reset the environment"""
        self.current_episode = 0
        self.episode_var.set("0")
        self.step_count = 0
        self.step_var.set("0")
        self.total_reward = 0.0
        self.reward_var.set("0.000")
        self.reward_history = []
        self.step_history = []
        self.status_var.set("Environment reset")
    
    def update_status(self):
        """Update status display"""
        self.step_var.set(str(self.step_count))
        self.reward_var.set(f"{self.total_reward:.3f}")
    
    def update_plots(self, frame):
        """Update the live plots"""
        # Update robot camera view
        if not self.frame_queue.empty():
            try:
                rgb_frame = self.frame_queue.get_nowait()
                self.robot_im.set_array(rgb_frame)
            except queue.Empty:
                pass
        
        # Update reward plot
        if len(self.reward_history) > 1:
            self.reward_line.set_data(self.step_history, self.reward_history)
            self.ax_reward.relim()
            self.ax_reward.autoscale_view()
        
        return [self.robot_im, self.reward_line]

def main():
    """Main function"""
    root = tk.Tk()
    app = TinyVLAGUI(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("GUI closed by user")

if __name__ == "__main__":
    main() 