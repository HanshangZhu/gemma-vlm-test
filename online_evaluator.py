#!/usr/bin/env python3
"""
Real Online VLA Evaluator:
Runs a true online evaluation of a LoRA-finetuned VLA model in a live MetaWorld environment.
This script loads a model, runs it on specified tasks, and generates a report with real performance metrics.
"""

import os
import sys
import json
import time
import argparse
import datetime
import random
from typing import Dict, List, Any

import torch
import numpy as np
import gymnasium as gym
from PIL import Image
from torchvision import transforms as T
from transformers import AutoTokenizer
from peft import PeftModel
import metaworld.envs.mujoco.env_dict as env_dict

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from unified_tinyvla import UnifiedTinyVLAModel

# --- Task Definitions ---
MT10_V2_TASK_NAMES = [
    "reach-v2-goal-observable", "push-v2-goal-observable", "pick-place-v2-goal-observable", "door-open-v2-goal-observable", 
    "drawer-open-v2-goal-observable", "drawer-close-v2-goal-observable", "button-press-topdown-v2-goal-observable", 
    "peg-insert-side-v2-goal-observable", "window-open-v2-goal-observable", "window-close-v2-goal-observable"
]
ZEROSHOT_EVAL_TASK_NAMES = MT10_V2_TASK_NAMES

class RealOnlineEvaluator:
    """Runs a real online evaluation and generates a report."""

    def __init__(self, lora_checkpoint_path: str, output_dir: str, num_episodes: int = 10):
        self.lora_checkpoint_path = lora_checkpoint_path
        self.output_dir = output_dir
        self.num_episodes = num_episodes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}

        os.makedirs(self.output_dir, exist_ok=True)
        print(f"DEVICE: {self.device}")

    def _load_model_and_tokenizer(self):
        """Loads the complete model and tokenizer using the refactored class."""
        print("Loading model and tokenizer...")
        
        # Define paths for the different model components
        base_model_path = "VLM_weights/Llava-Pythia-400M"
        
        # The refactored model class handles all the loading logic
        model = UnifiedTinyVLAModel(
            model_path=base_model_path,
            lora_checkpoint_path=self.lora_checkpoint_path
            # action_head_checkpoint_path can be added here if/when we have it
        ).to(self.device)
        
        model.eval()
        
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        
        print("✅ Model and tokenizer loaded.")
        return model, tokenizer

    def evaluate(self):
        """Main evaluation loop."""
        model, tokenizer = self._load_model_and_tokenizer()
        
        image_transform = T.Compose([
            T.ToTensor(),
            T.Resize((336, 336)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        all_task_success_rates = []

        for task_name in ZEROSHOT_EVAL_TASK_NAMES:
            print(f"--- Evaluating task: {task_name} ---")
            
            try:
                env_cls = env_dict.ALL_V2_ENVIRONMENTS[task_name]
                env = env_cls(seed=random.randint(0, 100000))
            except Exception as e:
                print(f"Failed to create env for {task_name}: {e}")
                continue

            task_successes = []
            for episode in range(self.num_episodes):
                obs = env.reset()
                done = False
                steps = 0
                episode_success = 0.0

                while not done:
                    img_obs = env.render(offscreen=True, camera_name='corner3')
                    img = Image.fromarray(np.array(img_obs, dtype=np.uint8))

                    image_tensor = image_transform(img).unsqueeze(0).to(self.device)
                    prompt = f"perform the task {task_name}"
                    tokens = tokenizer([prompt], return_tensors="pt").to(self.device)
                    
                    # Use the first 9 elements of the observation for the state
                    robot_state = obs[:9]
                    state_tensor = torch.from_numpy(robot_state).float().unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        # The model returns a dictionary containing the 'actions' tensor
                        model_output = model(
                            input_ids=tokens.input_ids,
                            attention_mask=tokens.attention_mask,
                            images=image_tensor,
                            states=state_tensor
                        )
                        action_tensor = model_output['actions']
                        
                        if action_tensor is None:
                            print("❌ Model failed to return an action. Using zero action.")
                            action = np.zeros(4) # Assuming action space is 4
                        else:
                            action = torch.tanh(action_tensor)[0, -1].cpu().numpy()

                    obs, reward, done, info = env.step(action)
                    done = done
                    steps += 1
                    
                    if info.get('success', 0.0) > 0.5:
                        episode_success = 1.0
                        break
                
                task_successes.append(episode_success)
                print(f"  Episode {episode + 1}/{self.num_episodes} | Success: {'Yes' if episode_success else 'No'}")
            
            avg_task_success = np.mean(task_successes)
            all_task_success_rates.append(avg_task_success)
            print(f"  Task: {task_name[:20]:<20} | Avg Success Rate: {avg_task_success:.2f}")
            env.close()

        overall_success_rate = np.mean(all_task_success_rates) if all_task_success_rates else 0.0
        print(f"\n📈 Overall Zero-Shot Success Rate: {overall_success_rate:.4f}")
        
        self.results['overall_success_rate'] = overall_success_rate
        self.results['task_breakdown'] = {task: rate for task, rate in zip(ZEROSHOT_EVAL_TASK_NAMES, all_task_success_rates)}
        
        self.generate_markdown_report()

    def generate_markdown_report(self):
        """Generates a markdown report from the evaluation results."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_path = os.path.join(self.output_dir, "real_evaluation_report.md")

        content = f"""# Real Online VLA Evaluation Report

**Generated:** {timestamp}  
**Checkpoint:** `{self.lora_checkpoint_path}`  
**Episodes per Task:** {self.num_episodes}

---

## Overall Performance

- **Mean Zero-Shot Success Rate:** **{self.results.get('overall_success_rate', 0.0):.4f}**

---

## Per-Task Success Rates

| Task Name | Success Rate |
|-----------|--------------|
"""
        for task, rate in self.results.get('task_breakdown', {}).items():
            content += f"| {task} | {rate:.4f} |\n"

        content += "\n---"

        with open(report_path, 'w') as f:
            f.write(content)
        
        print(f"\n📋 Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Real Online VLA Evaluator")
    parser.add_argument("--lora_checkpoint", type=str, required=True, help="Path to the LoRA checkpoint directory.")
    parser.add_argument("--output_dir", type=str, default="real_evaluation_results", help="Directory to save results and the report.")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to run for each task.")
    
    args = parser.parse_args()
    
    evaluator = RealOnlineEvaluator(
        lora_checkpoint_path=args.lora_checkpoint,
        output_dir=args.output_dir,
        num_episodes=args.num_episodes
    )
    evaluator.evaluate()

if __name__ == "__main__":
    main() 