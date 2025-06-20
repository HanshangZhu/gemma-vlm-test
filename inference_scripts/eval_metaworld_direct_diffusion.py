# eval_metaworld_direct_diffusion.py - Direct diffusion head call approach
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

# Add TinyVLA to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'TinyVLA'))

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

def setup_env(task_name: str, render_mode: str = 'rgb_array'):
    """Sets up a MetaWorld environment using ML1 benchmark."""
    print(f"[INFO] Setting up MetaWorld ML1 environment for task: {task_name}")
    
    # Map v2 task names to v3 equivalents
    v2_to_v3_mapping = {
        'pick-place-v2': 'pick-place-v3',
        'door-open-v2': 'door-open-v3',
        'drawer-open-v2': 'drawer-open-v3', 
        'button-press-topdown-v2': 'button-press-topdown-v3',
        'reach-v2': 'reach-v3',
        'push-v2': 'push-v3',
        'door-close-v2': 'door-close-v3',
        'drawer-close-v2': 'drawer-close-v3'
    }
    
    # Convert v2 task name to v3 if needed
    if task_name in v2_to_v3_mapping:
        v3_task_name = v2_to_v3_mapping[task_name]
        print(f"[INFO] Mapping {task_name} -> {v3_task_name}")
        task_name = v3_task_name
    
    try:
        # Use v3 API
        benchmark = metaworld.ML1(task_name)
        env = benchmark.train_classes[task_name]()
        task = random.choice(benchmark.train_tasks)
        env.set_task(task)

        # The new API often returns the env within a Gym wrapper
        if hasattr(env, 'env'):
            env = env.env
            
        # Set up camera for side view
        env.render_mode = render_mode
        env.camera_name = 'corner'  # Use corner camera
        
        return env
        
    except Exception as e:
        raise RuntimeError(f"Failed to create environment for {task_name}: {e}")

class DirectDiffusionPolicy:
    """Policy that directly calls the diffusion head"""
    
    def __init__(self, policy_config):
        self.load_policy(policy_config)
        
    def load_policy(self, policy_config):
        self.policy_config = policy_config
        model_base = policy_config["model_base"] if policy_config.get('enable_lora') else None
        model_name = get_model_name_from_path(policy_config['model_path'])
        model_path = policy_config["model_path"]
        
        # Temporarily modify the model loading to use float32
        original_float16 = torch.float16
        torch.float16 = torch.float32  # Hack to force float32 loading
        
        try:
            self.tokenizer, self.policy, self.image_processor, self.context_len = load_pretrained_model(
                model_path, model_base, model_name, False, False
            )
        finally:
            torch.float16 = original_float16  # Restore original
        
        # Ensure the noise scheduler uses float32
        if hasattr(self.policy, 'noise_scheduler'):
            # Recreate scheduler with float32
            from diffusers.schedulers.scheduling_ddim import DDIMScheduler
            self.policy.noise_scheduler = DDIMScheduler(
                num_train_timesteps=100,
                beta_schedule='squaredcos_cap_v2',
                clip_sample=True,
                set_alpha_to_one=True,
                steps_offset=0,
                prediction_type='epsilon'
            )
        
        # Load config
        config_path = '/'.join(model_path.split('/')[:-1])
        if os.path.exists(os.path.join(config_path, 'config.json')):
            self.config = LlavaPythiaConfig.from_pretrained(config_path, trust_remote_code=True)
        else:
            # Use default config
            self.config = type('Config', (), {
                'action_dim': 4,
                'chunk_size': 20,
                'mm_use_im_start_end': False
            })()
        
        # Set visual_concat if not set
        if not hasattr(self.policy, 'visual_concat') or self.policy.visual_concat == 'None':
            self.policy.visual_concat = 'token_cat'  # Default for dual camera
    
    def expand2square(self, pil_imgs, background_color):
        """Same as eval_real_franka.py"""
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
    
    def get_hidden_states(self, curr_image, robo_state, raw_lang):
        """Get hidden states from the model"""
        self.conv = conv_templates[self.policy_config['conv_mode']].copy()
        
        # For MetaWorld single camera, duplicate image to simulate dual camera
        if len(curr_image.shape) == 4:
            curr_image = curr_image.squeeze(0)
        
        # Duplicate image for dual camera simulation
        image = curr_image.unsqueeze(0)
        image_r = curr_image.unsqueeze(0)
        
        # Expand to square
        image = self.expand2square(image, tuple(x for x in self.image_processor.image_mean))
        image_tensor = self.image_processor.preprocess(
            image, return_tensors='pt', do_normalize=True, do_rescale=False,
            do_center_crop=False
        )['pixel_values']
        image_tensor = image_tensor.to(self.policy.device, dtype=self.policy.dtype)
        
        image_r = self.expand2square(image_r, tuple(x for x in self.image_processor.image_mean))
        image_tensor_r = self.image_processor.preprocess(
            image_r, return_tensors='pt', do_normalize=True, do_rescale=False,
            do_center_crop=False
        )['pixel_values']
        image_tensor_r = image_tensor_r.to(self.policy.device, dtype=self.policy.dtype)
        
        # Process language input
        inp = raw_lang
        if self.policy.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        
        self.conv.append_message(self.conv.roles[0], inp)
        self.conv.append_message(self.conv.roles[1], None)
        prompt = self.conv.get_prompt()
        prompt += " <|endoftext|>"
        
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        attn_mask = input_ids.ne(self.tokenizer.pad_token_id)
        states = robo_state.to(self.policy.device, dtype=self.policy.dtype)
        
        # Prepare inputs for multimodal processing
        input_ids, attention_mask, _, inputs_embeds, _ = self.policy.prepare_inputs_labels_for_multimodal(
            input_ids, attn_mask, None, None, image_tensor, 
            images_r=image_tensor_r, images_top=None, 
            visual_concat=self.policy.visual_concat, states=states
        )
        
        # Get hidden states from the model
        outputs = self.policy.get_model()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            return_dict=True
        )
        
        hidden_states = outputs[0]
        
        return hidden_states, states
    
    def generate_actions_direct(self, hidden_states, states):
        """Directly call the diffusion head to generate actions"""
        # Initialize action from Gaussian noise
        B = 1
        Tp = self.policy.num_queries
        action_dim = self.policy.action_dim
        
        noisy_action = torch.randn((B, Tp, action_dim)).cuda()
        naction = noisy_action.to(dtype=hidden_states.dtype)
        
        # Initialize scheduler
        self.policy.noise_scheduler.set_timesteps(self.policy.num_inference_timesteps)
        
        # Diffusion denoising loop
        for k in self.policy.noise_scheduler.timesteps:
            # Predict noise
            noise_pred = self.policy.embed_out(
                naction, k, 
                global_cond=hidden_states, 
                states=states
            )
            
            # Inverse diffusion step (remove noise)
            naction = self.policy.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample
        
        return naction

def eval_bc(policy, env, checkpoint_path=None, num_rollouts=1, max_timesteps=150, 
            raw_lang="Pick up the object and place it at the target.", save_video=False):
    """Evaluate using direct diffusion head calls"""
    
    set_seed(0)
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[INFO] Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
        # Remove _orig_mod prefix if present
        checkpoint = {k.replace('_orig_mod.', ''): v for k, v in checkpoint.items()}
        policy.policy.embed_out.load_state_dict(checkpoint, strict=False)
        print("[INFO] Checkpoint loaded successfully!")
    
    # Settings
    temporal_agg = True
    action_dim = getattr(policy.config, 'action_dim', 4)
    chunk_size = getattr(policy.config, 'chunk_size', 20)
    
    policy.policy.eval()
    
    # Load stats if available
    stats = None
    stats_path = os.path.join(os.path.dirname(policy.policy_config['model_path']), 'dataset_stats.pkl')
    if os.path.exists(stats_path):
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
        print(f"[INFO] Loaded stats from {stats_path}")
    
    # Post-processing function
    if stats and 'action_mean' in stats:
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    else:
        # No normalization for raw actions
        post_process = lambda a: a
    
    query_frequency = 1 if temporal_agg else chunk_size // 2
    num_queries = chunk_size
    
    all_results = []
    
    for rollout_id in range(num_rollouts):
        print(f"\n===== Episode {rollout_id + 1}/{num_rollouts} =====")
        
        obs, info = env.reset()
        
        # Initialize temporal aggregation
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps + num_queries, action_dim], 
                                         dtype=torch.float32).cuda()
        
        frames = []
        total_reward = 0.0
        
        with torch.inference_mode():
            for t in range(max_timesteps):
                # Get observation
                rgb = env.render()
                if rgb is None:
                    rgb = np.zeros((480, 480, 3), dtype=np.uint8)
                
                # Convert to tensor
                curr_image = torch.from_numpy(rgb / 255.0).float().cuda()
                curr_image = curr_image.permute(2, 0, 1)  # HWC -> CHW
                
                # Get robot state
                robot_state = torch.from_numpy(obs[:7]).float().cuda().unsqueeze(0)
                
                # Query policy at specified frequency
                if t % query_frequency == 0:
                    # Get hidden states
                    hidden_states, states = policy.get_hidden_states(curr_image, robot_state, raw_lang)
                    
                    # DIRECTLY call diffusion head
                    all_actions = policy.generate_actions_direct(hidden_states, states)
                    
                    print(f"  Generated actions shape: {all_actions.shape}, range: [{all_actions.min():.3f}, {all_actions.max():.3f}]")
                
                # Temporal aggregation
                if temporal_agg:
                    all_time_actions[[t], t:t + num_queries] = all_actions
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    
                    if len(actions_for_curr_step) > 0:
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = torch.zeros((1, action_dim)).cuda()
                else:
                    raw_action = all_actions[:, t % query_frequency]
                
                # Post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                
                # Clip actions for MetaWorld
                action = np.clip(action, -1.0, 1.0)
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                
                frames.append(rgb)
                
                if t % 25 == 0:
                    print(f"  Step {t}: reward={total_reward:.3f}, action={action}")
                
                if info.get("success", False):
                    print(f"[SUCCESS] Task completed at step {t}!")
                    break
                
                if done:
                    print(f"Episode ended at step {t}")
                    break
        
        result = {
            'rollout_id': rollout_id,
            'total_reward': total_reward,
            'success': info.get("success", False),
            'steps': t + 1,
            'frames': frames
        }
        all_results.append(result)
        
        print(f"Episode {rollout_id + 1} Results:")
        print(f"  Reward: {total_reward:.3f}")
        print(f"  Success: {result['success']}")
        print(f"  Steps: {result['steps']}")
        
        # Save video if requested
        if save_video and frames:
            save_episode_video(frames, f"metaworld_direct_episode_{rollout_id + 1}.mp4")
    
    return all_results

def save_episode_video(frames, filename, fps=10):
    """Save frames as video"""
    if not frames:
        return
    
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    out.release()
    print(f"[INFO] Video saved: {filename}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to the trained VLA model")
    parser.add_argument("--model-base", type=str, default=None,
                       help="Base model path for LoRA models")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to diffusion head checkpoint")
    parser.add_argument("--task", default="pick-place-v3")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--prompt", default="Pick up the object and place it at the target.")
    parser.add_argument("--save-video", action="store_true")
    
    args = parser.parse_args()
    
    # Suppress warnings
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    
    print("🤖 TinyVLA MetaWorld Evaluation (Direct Diffusion)")
    print("=" * 60)
    print("This version directly calls the diffusion head,")
    print("bypassing the broken routing logic in forward()")
    print("=" * 60)
    
    # Policy config
    policy_config = {
        "model_path": args.model_path,
        "model_base": args.model_base,
        "enable_lora": args.model_base is not None,
        "conv_mode": "pythia",
    }
    
    # Setup environment
    env = setup_env(args.task)
    if env is None:
        return
    
    # Create policy
    policy = DirectDiffusionPolicy(policy_config)
    
    # Run evaluation
    results = eval_bc(
        policy=policy,
        env=env,
        checkpoint_path=args.checkpoint,
        num_rollouts=args.episodes,
        max_timesteps=args.max_steps,
        raw_lang=args.prompt,
        save_video=args.save_video
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("📈 EVALUATION SUMMARY")
    print("=" * 60)
    
    avg_reward = np.mean([r['total_reward'] for r in results])
    success_rate = np.mean([r['success'] for r in results]) * 100
    avg_steps = np.mean([r['steps'] for r in results])
    
    print(f"Task: {args.task}")
    print(f"Prompt: '{args.prompt}'")
    print(f"Average Reward: {avg_reward:.3f}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Average Steps: {avg_steps:.1f}")
    
    print("\n🔍 Key Insight:")
    print("By directly calling the diffusion head, we bypass the")
    print("problematic routing logic that was preventing action generation!")

if __name__ == "__main__":
    main() 