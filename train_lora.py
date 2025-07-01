#!/usr/bin/env python3
"""
A clean, from-scratch LoRA VLA Trainer for MetaWorld Multi-Task Learning
"""

import os
import sys
import json
import pickle
import argparse
import yaml
from dataclasses import dataclass, asdict, field
from functools import partial
import math
import io
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm
import glob
from transformers.utils import logging as hf_logging
from datasets import load_dataset  # NEW: HuggingFace Datasets for MT50

# ADD THESE DEEPSPEED IMPORTS
import deepspeed
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

# ---------------------------------------------------------------------------
# Path setup BEFORE heavy imports
# ---------------------------------------------------------------------------
# 1. Add TinyVLA root so that `model` and other sub-packages resolve
tinyvla_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'TinyVLA'))
if tinyvla_root not in sys.path:
    sys.path.insert(0, tinyvla_root)

# 2. Explicitly add the `llava_pythia` package path so `import llava_pythia.*` works
llava_pkg_root = os.path.join(tinyvla_root, 'llava_pythia')  # note underscore, not hyphen
if llava_pkg_root not in sys.path:
    sys.path.insert(0, llava_pkg_root)

# 3. Add nested path (`TinyVLA/llava_pythia/llava_pythia`) to catch deep relative imports
nested_llava_pkg = os.path.join(llava_pkg_root, 'llava_pythia')
if os.path.isdir(nested_llava_pkg) and nested_llava_pkg not in sys.path:
    sys.path.insert(0, nested_llava_pkg)

# ---------------------------------------------------------------------------

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#this is where this script exists
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../TinyVLA')))

# Add project root to path for local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
# Add the correct sub-directory to the path (underscored directory name)
llava_pythia_path = os.path.join(project_root, "TinyVLA", "llava_pythia")
if llava_pythia_path not in sys.path:
    sys.path.insert(0, llava_pythia_path)

# Also add the nested package directory if present (TinyVLA/llava_pythia/llava_pythia)
nested_llava_pythia = os.path.join(llava_pythia_path, "llava_pythia")
#os.join() is used to join two paths together, means that if the path is not in the path, it will be added to the path
if nested_llava_pythia not in sys.path and os.path.isdir(nested_llava_pythia):
    sys.path.insert(0, nested_llava_pythia)

# Remove outdated/incorrect hyphenated path if it was previously inserted
incorrect_path = os.path.join(project_root, "TinyVLA", "llava-pythia")
if incorrect_path in sys.path:
    sys.path.remove(incorrect_path)

# Now we can import from TinyVLA
from model.builder import load_pretrained_model
from model.multimodal_encoder.clip_encoder import CLIPVisionTower
from constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

# 🔧 Fix 2: Apply dimension mismatch patch for ConditionalUnet1D
def apply_dimension_fixes():
    """Apply fixes for dimension mismatch issues in ConditionalUnet1D"""
    try:
        from TinyVLA.policy_heads.models.droid_unet_diffusion import ConditionalUnet1D
        
        # Store the original forward method
        original_forward = ConditionalUnet1D.forward
        
        def patched_forward(self, sample, timestep, global_cond=None, states=None):
            """Patched forward method that properly handles dimension mismatches"""
            #sample is the image, timestep is the timestep, global_cond is the global conditioning, states is the states
            #the global conditioning of the image is what is passed to the unet, so the image-text pair embedding;
            try:
                # move axis for processing
                sample = sample.moveaxis(-1, -2)
                #here we are moving the axis of the image to the last dimension, so that the image is in the shape of [batch_size, channels, height, width]
                
                
                # process global conditioning with proper error handling
                if global_cond is not None:
                    # Pool and normalize global conditioning
                    global_cond = self.global_1d_pool(global_cond.permute(0, 2, 1)).squeeze(-1)
                    global_cond = self.norm_after_pool(global_cond)  # layernorm
                    
                    # Concatenate states if provided and apply combine layer properly
                    if states is not None:
                        combined_cond = torch.cat([global_cond, states], dim=-1)
                        # Apply combine layer to project back to global_cond_dim
                        global_cond = self.combine(combined_cond)
                
                # Handle timesteps
                timesteps = timestep
                if not torch.is_tensor(timesteps):
                    timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
                elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                    timesteps = timesteps[None].to(sample.device)
                timesteps = timesteps.expand(sample.shape[0])

                # Get diffusion step embedding
                global_feature = self.diffusion_step_encoder(timesteps)

                if global_cond is not None:
                    global_feature = torch.cat([global_feature, global_cond], axis=-1)

                # Standard UNet path
                x = sample
                h = []
                for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
                    x = resnet(x, global_feature)
                    x = resnet2(x, global_feature)
                    h.append(x)
                    x = downsample(x)

                for mid_module in self.mid_modules:
                    x = mid_module(x, global_feature)

                for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
                    h_pop = h.pop()
                    x = torch.cat((x, h_pop), dim=1)
                    x = resnet(x, global_feature)
                    x = resnet2(x, global_feature)
                    x = upsample(x)

                x = self.final_conv(x)
                x = x.moveaxis(-1, -2)
                return x
                
            except Exception as e:
                print(f"💥 Error in patched diffusion forward: {e}")
                # Fall back to original method
                return original_forward(self, sample, timestep, global_cond, states)
        
        # Replace the forward method
        ConditionalUnet1D.forward = patched_forward
        print("✅ Applied dimension mismatch patch to ConditionalUnet1D")
        
    except Exception as e:
        print(f"⚠️ Could not apply dimension patch: {e}")

# Apply the fixes immediately
apply_dimension_fixes()

# Heavy imports that rely on the paths above
from llava_pythia.model.language_model.pythia.llava_pythia import LlavaPythiaForCausalLM, LlavaPythiaConfig
from policy_heads.models import ConditionalUnet1D

# Silence irrelevant weight-mismatch warnings/info from Transformers (e.g., "Some weights of the model checkpoint were not used …")
hf_logging.set_verbosity_error()

# ---------------------------------------------------------------------------
# Monkey-patch for tokenizer compatibility: some older configs expect the
# non-fast `GPTNeoXTokenizer` class name, which no longer ships with
# transformers.  We alias it to the modern `GPTNeoXTokenizerFast` so that
# `AutoTokenizer.from_pretrained` can resolve it seamlessly.
# ---------------------------------------------------------------------------
from transformers import GPTNeoXTokenizerFast
import transformers as _tf_patch
if not hasattr(_tf_patch, 'GPTNeoXTokenizer'):
    _tf_patch.GPTNeoXTokenizer = GPTNeoXTokenizerFast

@dataclass
class TrainingConfig:
    """Configuration for LoRA training, loaded from a YAML file."""
    model_path: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    diffusion_learning_rate: float
    max_steps: int
    save_steps: int
    output_dir: str
    diffusion_head_save_dir: str  # New parameter for diffusion head save directory
    data_root: str
    train_tasks: str
    
    # Memory optimization options
    use_bf16: bool
    use_gradient_checkpointing: bool
    freeze_vision_encoder: bool
    train_diffusion_head: bool
    cpu_offload: bool
    max_memory_cleanup_steps: int
    chunk_size: int
    image_size: int
    
    # Advanced memory options
    dataloader_num_workers: int
    pin_memory: bool
    persistent_workers: bool
    
    # NaN prevention options
    gradient_clip_norm: float
    warmup_steps: int
    weight_decay: float
    diffusion_warmup_steps: int = 0  # if >0 keep embed_out frozen for first N steps
    dataset_variant: str = "short"  # "short" (legacy) or "mt50"
    # Debugging option: create synthetic MT50 dataset locally without downloads
    dummy_mt50_samples: int = 0  # >0 activates 'mt50_dummy' variant implicitly
    # Prompt engineering
    prompt_style: str = "simple"  # "simple" or "detailed"
    prompt_json_path: str = "datasets/mt50_task_prompts.json"
    # Dataset validation options
    validate_samples_count: int = 0  # 0 = skip, >0 = validate that many random samples.
    # ADD THIS NEW FIELD FOR DEEPSPEED CONFIG PATH
    deepspeed_config_path: str | None = None # Path to DeepSpeed config file for optional use

class MetaWorldDataset(torch.utils.data.Dataset):
    """MetaWorld dataset for multi-task robot learning."""
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tasks = config.train_tasks.split(",")
        self.trajectories = self._load_trajectories()
        self.transform = T.Compose([
            T.Resize((config.image_size, config.image_size)), # Use configurable image size for memory savings
            T.ToTensor(),
        ])

    def _load_trajectories(self):
        all_trajectories = []
        
        # Fix path construction to find the img_only directory
        # data_root: datasets/short-MetaWorld/short-MetaWorld/r3m-processed/r3m_MT10_20
        # img_root should be: datasets/short-MetaWorld/short-MetaWorld/img_only
        dataset_root = os.path.dirname(self.config.data_root)  # r3m-processed
        parent_dir = os.path.dirname(dataset_root)             # short-MetaWorld
        img_root = os.path.join(parent_dir, "img_only")

        print(f"🔍 Debug paths:")
        print(f"  data_root: {self.config.data_root}")
        print(f"  dataset_root: {dataset_root}")
        print(f"  img_root: {img_root}")
        print(f"  img_root exists: {os.path.exists(img_root)}")

        for task in self.tasks:
            pkl_path = os.path.join(self.config.data_root, f"{task}.pkl")
            if not os.path.exists(pkl_path):
                print(f"Warning: {pkl_path} not found, skipping.")
                continue
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)

            task_img_dir = os.path.join(img_root, task)
            print(f"🔍 Looking for task images at: {task_img_dir}")
            if not os.path.exists(task_img_dir):
                print(f"Warning: {task_img_dir} not found for task {task}")
                continue

            traj_dirs = sorted(glob.glob(f"{task_img_dir}/*"), key=lambda x: int(os.path.basename(x)))
            print(f"🔍 Found {len(traj_dirs)} trajectory directories for {task}")
            for traj_idx, tdir in enumerate(traj_dirs):
                if traj_idx >= len(data['actions']):
                    continue

                img_paths = sorted(glob.glob(f"{tdir}/*.jpg"), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                
                num_steps = len(data['actions'][traj_idx])
                num_images = len(img_paths)
                
                # Use minimum length to handle mismatched data
                if num_images == 0 or num_steps == 0:
                    print(f"🔍 Skipping traj {traj_idx}: {num_images} images, {num_steps} actions - one is empty")
                    continue
                
                # Use the minimum of available images and actions
                min_steps = min(num_images, num_steps)
                if min_steps < 5:  # Skip very short trajectories
                    print(f"🔍 Skipping traj {traj_idx}: only {min_steps} valid steps")
                    continue

                for step_idx in range(min_steps):
                    all_trajectories.append({
                        'image_path': img_paths[step_idx],
                        'state': data['state'][traj_idx][step_idx][:7],
                        'action': data['actions'][traj_idx][step_idx],
                        'prompt': f"In: What action should the robot take to {task.replace('-', ' ')}? State:"
                    })
        
        print(f"🔍 Total trajectories loaded: {len(all_trajectories)}")
        return all_trajectories

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        
        image = Image.open(traj['image_path']).convert("RGB")
        image_tensor = self.transform(image)

        state = torch.tensor(traj['state'], dtype=torch.float32)
        action = torch.tensor(traj['action'], dtype=torch.float32)
        prompt = traj['prompt']
        
        return image_tensor, state, action, prompt

# =============================================================================
# NEW: MT50 DATASET SUPPORT (lerobot/metaworld_mt50)
# =============================================================================

class MetaWorldMT50Dataset(torch.utils.data.Dataset):
    """MT50 dataset fetched from HuggingFace (lerobot/metaworld_mt50)."""

    HF_NAME = "lerobot/metaworld_mt50"

    def __init__(self, config: TrainingConfig):
        self.config = config

        # Load split (non-streaming so we can random-access)
        print("📥 Loading MT50 dataset from HuggingFace … (first call will download)")
        self.ds = load_dataset(self.HF_NAME, split="train", cache_dir=os.getenv("HF_DATASETS_CACHE", None))

        # Load prompt mapping JSON if available
        self.prompt_map = {}
        if os.path.isfile(config.prompt_json_path):
            import json
            with open(config.prompt_json_path, 'r') as f:
                self.prompt_map = json.load(f)
            print(f"📖 Loaded prompt map with {len(self.prompt_map)} tasks from {config.prompt_json_path}")
        else:
            print(f"⚠️ Prompt JSON {config.prompt_json_path} not found – using fallback prompts.")

        # Image transform matching vision tower resolution
        self.transform = T.Compose([
            T.Resize((config.image_size, config.image_size)),
            T.ToTensor(),
        ])

        # Pre-computed mapping from integer task_id to human-readable Meta-World task names.
        # Index taken from official MT50 ordering.
        self._task_names = [
            "reach-v2", "push-v2", "pick-place-v2", "door-open-v2", "drawer-open-v2",
            "drawer-close-v2", "button-press-topdown-v2", "button-press-v2", "button-press-wall-v2", "button-press-topdown-wall-v2",
            "door-close-v2", "hammer-v2", "handle-press-side-v2", "handle-press-v2", "handle-pull-v2",
            "handle-pull-side-v2", "lever-pull-v2", "peg-insert-side-v2", "pick-place-wall-v2", "push-wall-v2",
            "reach-wall-v2", "shelf-place-v2", "sweep-into-v2", "sweep-v2", "window-open-v2",
            "window-close-v2", "coffee-button-v2", "coffee-pull-v2", "coffee-push-v2", "dial-turn-v2",
            "disassemble-v2", "door-lock-v2", "hand-insert-v2", "handover-v2", "lamp-turn-on-v2",
            "pour-v2", "soccer-v2", "stick-push-v2", "stick-pull-v2", "switch-v2",
            "basketball-v2", "faucet-open-v2", "faucet-close-v2", "ladle-pick-v2", "safepick-v2",
            "safepush-v2", "scale-v2", "spoon-pick-v2", "tray-pick-v2", "utensil-pick-v2"
        ]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]

        # HuggingFace image feature → PIL.Image
        img = row["observation.image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        image_tensor = self.transform(img)

        # We no longer use environment states for MT50 – provide zeros (7-D) as placeholder.
        state = torch.zeros(7, dtype=torch.float32)
        action = torch.tensor(row["action"], dtype=torch.float32)

        # Prompt engineering from task_id → task name
        tid = int(row.get("task_id", -1))
        if 0 <= tid < len(self._task_names):
            task_name = self._task_names[tid]
        else:
            task_name = f"task-{tid}"

        # ---------------------------------------------------------------
        # Prompt style selection
        # ---------------------------------------------------------------
        # Always sample style per item using 40/30/30 distribution
        r = random.random()
        if r < 0.40:
            chosen_style = "simple"
        elif r < 0.70:
            chosen_style = "detailed"
        else:
            chosen_style = "very_detailed"

        # Retrieve template or fallbacks
        prompt_template = self.prompt_map.get(task_name, {}).get(chosen_style)
        if prompt_template is None:
            # Derive from less-detailed versions or a generic template
            prompt_template = self.prompt_map.get(task_name, {}).get("detailed") or self.prompt_map.get(task_name, {}).get("simple")
            if prompt_template is None:
                prompt_template = f"Perform the task: {task_name.replace('-', ' ')}"

        # Auto-augment for very_detailed when custom field absent
        if chosen_style == "very_detailed" and self.prompt_map.get(task_name, {}).get("very_detailed") is None:
            prompt_template = (
                prompt_template +
                " Provide fine-grained control: approach with the end-effector aligned, adjust orientation as needed, "
                "close the gripper gently, verify stable grasp/activation, and hold the pose for 0.5 s to ensure success."
            )

        prompt = f"<image>\n{prompt_template}"

        return image_tensor, state, action, prompt

# -----------------------------------------------------------------------------
# DUMMY variant – generates small synthetic set for local debugging
# -----------------------------------------------------------------------------

class MetaWorldMT50DummyDataset(torch.utils.data.Dataset):
    """Small synthetic dataset with MT50-like shapes to debug the pipeline."""

    def __init__(self, config: TrainingConfig, n_samples: int = 100):
        self.n = max(1, n_samples)
        self.config = config

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Random RGB image tensor in [0,1]
        image_tensor = torch.rand(3, self.config.image_size, self.config.image_size)

        # Random state (7-D) and action (4-D) in reasonable ranges
        state = torch.randn(7).clamp(-1, 1)
        action = torch.randn(4).clamp(-1, 1)

        prompt = "In: What action should the robot take? State:"
        return image_tensor, state, action, prompt

def get_dataset_and_stats(config: TrainingConfig):
    """Factory that returns the chosen dataset and (mean/std) stats."""

    variant = config.dataset_variant.lower()

    # Allow quick local debugging without heavy downloads
    if variant == "mt50_dummy" or (variant == "mt50" and config.dummy_mt50_samples > 0):
        dummy_n = config.dummy_mt50_samples or 100
        dataset = MetaWorldMT50DummyDataset(config, n_samples=dummy_n)
        # Gather stats over full dummy set – tiny, so fine
        states = torch.stack([dataset[i][1] for i in range(len(dataset))])
        actions = torch.stack([dataset[i][2] for i in range(len(dataset))])

    elif variant == "mt50":
        dataset = MetaWorldMT50Dataset(config)

        # Compute running mean/std on a sample subset (to save RAM)
        max_samples = min(20000, len(dataset))
        print(f"📊 Computing stats on {max_samples} random MT50 samples …")

        rng = torch.Generator().manual_seed(42)
        indices = torch.randint(high=len(dataset), size=(max_samples,), generator=rng)

        # online algorithm
        state_acc = []
        action_acc = []
        for idx in indices:
            _, state, action, _ = dataset[int(idx)]
            state_acc.append(state)
            action_acc.append(action)

        states = torch.stack(state_acc)
        actions = torch.stack(action_acc)

    else:
        # Legacy short-MetaWorld dataset
        dataset = MetaWorldDataset(config)
        states_list = [item[1] for item in dataset]
        actions_list = [item[2] for item in dataset]
        states = torch.stack(states_list)
        actions = torch.stack(actions_list)

    stats = {
        'state_mean': states.mean(dim=0),
        'state_std': states.std(dim=0) + 1e-6,
        'action_mean': actions.mean(dim=0),
        'action_std': actions.std(dim=0) + 1e-6,
    }

    # Debug
    print("🔍 Action statistics:")
    print(f"  - Action mean: {stats['action_mean']}")
    print(f"  - Action std: {stats['action_std']}")
    print(f"  - Action raw min/max: [{actions.min():.4f}, {actions.max():.4f}]")

    return dataset, stats

class CollateFn:
    """A callable class for collating batches, making it pickle-able for multiprocessing."""
    def __init__(self, stats, tokenizer, config):
        self.stats = stats
        self.tokenizer = tokenizer
        self.config = config
        # The device should be handled by the main training loop, not the collate_fn
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, batch):
        # Our dataset returns tuples: (image, state, action, language)
        images, states, actions, languages = zip(*batch)
        
        # Use configurable chunk_size for memory optimization
        chunk_size = self.config.chunk_size
        action_sequences = []
        for action in actions:
            # Convert to tensor and repeat to create sequence
            if isinstance(action, torch.Tensor):
                action_tensor = action.float()
            else:
                action_tensor = torch.from_numpy(action).float()
            # Repeat the action chunk_size times: [action_dim] -> [chunk_size, action_dim]
            action_sequence = action_tensor.unsqueeze(0).repeat(chunk_size, 1)
            action_sequences.append(action_sequence)
        
        # Determine dtype based on config
        dtype = torch.bfloat16 if self.config.use_bf16 else torch.float32
        
        # Stack into batches with memory-efficient dtype, but keep on CPU
        images = torch.stack([torch.from_numpy(img).float() if not isinstance(img, torch.Tensor) else img.float() for img in images])
        states = torch.stack([torch.from_numpy(state).float() if not isinstance(state, torch.Tensor) else state.float() for state in states])
        actions = torch.stack(action_sequences)  # [batch_size, chunk_size, action_dim]
        
        # Cast to correct dtype
        images = images.to(dtype=dtype)
        states = states.to(dtype=dtype)
        actions = actions.to(dtype=dtype)

        # --- Robust normalisation --------------------------------------------------
        state_std = self.stats['state_std'].to(dtype=dtype)
        action_std_base = self.stats['action_std'].to(dtype=dtype)

        # Clamp tiny stds to avoid blow-ups (if a dimension is near-constant)
        state_std  = torch.clamp(state_std,  min=1e-2)
        action_std = torch.clamp(action_std_base, min=1e-2)

        states  = (states  - self.stats['state_mean'].to(dtype=dtype))  / state_std
        # ---------------------------------------------------------------------------
        
        # Debug: Check state dimensions to ensure they don't cause the 519-512=7 dimension mismatch
        if states.shape[0] > 0: # Ensure batch is not empty
             print(f"🔍 State debugging: shape={states.shape}, dtype={states.dtype}, min/max=[{states.min():.4f}, {states.max():.4f}]")
        
        # 🔧 Better action normalization to prevent NaN
        # First normalize actions to prevent extreme values
        actions_normalized = (actions - self.stats['action_mean'].to(dtype=dtype)) / action_std
        # Then clamp normalized actions to reasonable bounds
        actions_normalized = torch.clamp(actions_normalized, min=-3.0, max=3.0)
        
        # Debug: Per-dimension statistics for the first batch processed
        if not hasattr(self, '_printed_stats'):
            self._printed_stats = True
            print("🔎 Collate debug -- state_std (clamped):", state_std)
            print("🔎 Collate debug -- action_std (clamped):", action_std)
            for d in range(actions_normalized.shape[-1]):
                print(f"    action dim {d}: min {actions_normalized[...,d].min():.3f} max {actions_normalized[...,d].max():.3f}")
        
        # Debug: Check action values after normalization and clamping
        if torch.isnan(actions_normalized).any() or torch.isinf(actions_normalized).any():
            print(f"⚠️ NaN/Inf in actions after normalization! min/max: [{actions_normalized.min():.4f}, {actions_normalized.max():.4f}]")
            # Replace NaN/inf with zeros
            actions_normalized = torch.where(torch.isnan(actions_normalized) | torch.isinf(actions_normalized), torch.zeros_like(actions_normalized), actions_normalized)
        
        # Validate that no NaN or inf values exist in the data
        if torch.isnan(images).any() or torch.isinf(images).any():
            print("Warning: NaN/Inf detected in images")
        if torch.isnan(states).any() or torch.isinf(states).any():
            print("Warning: NaN/Inf detected in states")
        if torch.isnan(actions_normalized).any() or torch.isinf(actions_normalized).any():
            print("Warning: NaN/Inf detected in actions after normalization")
        
        # Create language inputs (keep as long/int, not bf16)
        input_ids = []
        labels = []
        for language in languages:
            # Create instruction text
            instruction = f"<image>\nPerform the task: {language}"
            
            # Tokenize
            tokens = self.tokenizer.encode(instruction)
            out_token_ids = tokens + [self.tokenizer.eos_token_id]
            
            input_ids.append(torch.tensor(tokens))
            labels.append(torch.tensor(out_token_ids))
        
        # Pad sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        return {
            "images": images,
            "states": states,
            "actions": actions_normalized,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "is_pad": torch.zeros(actions_normalized.size(0), actions_normalized.size(1), dtype=torch.bool)  # No padding since we repeat actions
        }

class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        # DeepSpeed will set LOCAL_RANK env var, otherwise default to 0 for single-GPU
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device("cpu") # Fallback to CPU if no CUDA
        print(f"✅ Using device: {self.device}")

        # Initialize components to None
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.data_loader = None
        self.stats = None
        self.best_loss = float('inf')

        # DeepSpeed engine (will be set if config.deepspeed_config_path is provided)
        self.model_engine = None

    def _setup(self):
        """Initializes all components for training."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        self._setup_dataset_and_stats()
        self._setup_model_and_tokenizer()
        self._setup_lora()

        # Enable gradient checkpointing once, after LoRA has wrapped the model
        if self.config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Conditionally setup DeepSpeed or regular optimizer/scheduler
        self._setup_deepspeed_or_optimizer()
        self._initialize_parameters()
        self._setup_dataloader()

    def _setup_dataset_and_stats(self):
        print("🔄 Creating dataset...")
        self.dataset, self.stats = get_dataset_and_stats(self.config)
        print(f"✅ Dataset created with {len(self.dataset)} samples.")

        # ------------------------------------------------------------------
        # Optional quick validation – can be skipped to speed up debug runs
        # ------------------------------------------------------------------
        n_validate_raw = getattr(self.config, "validate_samples_count", 0)
        # Accept int or numeric string
        try:
            n_validate = int(n_validate_raw)
        except (TypeError, ValueError):
            n_validate = 0

        if n_validate != 0:
            validate_n = min(abs(n_validate), len(self.dataset))
            if n_validate < 0:
                # Negative means validate entire dataset (legacy behaviour)
                validate_indices = range(len(self.dataset))
            else:
                # Random subset of n samples
                validate_indices = torch.randperm(len(self.dataset))[:validate_n]

            from tqdm import tqdm as _tqdm
            nan_img = nan_state = nan_action = 0
            for enum_idx, ds_idx in enumerate(_tqdm(validate_indices, desc="Validating dataset")):
                img, state, action, prompt = self.dataset[int(ds_idx)]
                if torch.isnan(img).any() or torch.isinf(img).any():
                    nan_img += 1
                if torch.isnan(state).any() or torch.isinf(state).any():
                    nan_state += 1
                if torch.isnan(action).any() or torch.isinf(action).any():
                    nan_action += 1
                if enum_idx < 3:
                    print(f"🔍 Sample {enum_idx}: state range [{state.min():.3f},{state.max():.3f}] | action range [{action.min():.3f},{action.max():.3f}] | img min/max [{img.min():.3f},{img.max():.3f}]")

            print(f"✅ Dataset validation completed on {validate_n} samples: NaN images:{nan_img}, NaN states:{nan_state}, NaN actions:{nan_action}")
        else:
            print("⏭️  Dataset validation skipped (validate_samples_count=0)")

        with open(os.path.join(self.config.output_dir, "stats.pkl"), "wb") as f:
            pickle.dump(self.stats, f)

    def _setup_model_and_tokenizer(self):
        print("🔄 Loading model and tokenizer...")

        # ------------------------------------------------------------------
        # Construct a patched config FIRST so that custom VLA fields exist
        # even for backbones that ship without them (e.g. Lesjie/Llava-Pythia-700M).
        # ------------------------------------------------------------------

        patched_cfg = LlavaPythiaConfig.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
        )

        # Inject/overwrite VLA-specific attributes (harmless if they already exist)
        patched_cfg.action_head_type = 'droid_diffusion'
        patched_cfg.action_dim = 4
        patched_cfg.state_dim = 7
        patched_cfg.chunk_size = self.config.chunk_size
        patched_cfg.concat = 'token_cat'
        patched_cfg.mm_use_im_start_end = True

        # Now load the model with the patched config
        self.model = LlavaPythiaForCausalLM.from_pretrained(
            self.config.model_path,
            config=patched_cfg,
            torch_dtype=torch.float32 if not self.config.use_bf16 else torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path, use_fast=False, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.unk_token
        
        # 🔥 CRITICAL FIX: Properly resize vision tower for target image size
        vision_tower = self.model.get_vision_tower()
        vision_tower.to(device=self.device, dtype=torch.float32)
        
        vision_config = vision_tower.vision_model.config
        original_image_size = vision_config.image_size
        target_image_size = self.config.image_size

        print(f"🔍 Vision tower current image size: {original_image_size}")
        print(f"🔍 Target image size: {target_image_size}")

        # Always resize if sizes don't match
        if original_image_size != target_image_size:
            print(f"🔄 Resizing vision tower from {original_image_size} to {target_image_size}")
            
            # Calculate positions for both original and target sizes
            original_patch_size = vision_config.patch_size
            original_num_patches = (original_image_size // original_patch_size) ** 2
            original_num_positions = original_num_patches + 1  # +1 for CLS token
            
            target_num_patches = (target_image_size // original_patch_size) ** 2
            target_num_positions = target_num_patches + 1
            
            print(f"🔍 Original: {original_num_patches} patches, {original_num_positions} positions")
            print(f"🔍 Target: {target_num_patches} patches, {target_num_positions} positions")
            
            # Update vision config
            vision_config.image_size = target_image_size
            
            # Save original position embeddings
            original_pos_embedding = vision_tower.vision_model.embeddings.position_embedding
            original_weights = original_pos_embedding.weight.data.clone()
            
            # Update embedding dimensions
            vision_tower.vision_model.embeddings.num_patches = target_num_patches
            vision_tower.vision_model.embeddings.num_positions = target_num_positions
            
            # Create new position embedding layer
            new_pos_embedding = nn.Embedding(
                target_num_positions,
                vision_config.hidden_size
            ).to(self.device, dtype=torch.float32)
            
            # Handle interpolation
            if target_num_positions != original_num_positions:
                print("🔄 Interpolating position embeddings...")
                
                # Extract CLS token (first position)
                cls_token = original_weights[0:1, :]  # [1, hidden_size]
                
                # Extract patch embeddings (skip CLS token)
                patch_embeddings = original_weights[1:, :]  # [original_num_patches, hidden_size]
                
                # Reshape for interpolation
                original_grid_size = int(math.sqrt(original_num_patches))
                target_grid_size = int(math.sqrt(target_num_patches))
                
                # Reshape to 2D grid: [hidden_size, grid_size, grid_size]
                patch_embeddings_2d = patch_embeddings.T.reshape(
                    vision_config.hidden_size, original_grid_size, original_grid_size
                ).unsqueeze(0)  # [1, hidden_size, grid_size, grid_size]
                
                # Interpolate to target size
                interpolated_embeddings = F.interpolate(
                    patch_embeddings_2d,
                    size=(target_grid_size, target_grid_size),
                    mode='bicubic',
                    align_corners=False
                )
                
                # Reshape back: [target_num_patches, hidden_size]
                interpolated_patch_embeddings = interpolated_embeddings.squeeze(0).reshape(
                    vision_config.hidden_size, -1
                ).T
                
                # Set new position embeddings
                new_pos_embedding.weight.data[0:1, :] = cls_token
                new_pos_embedding.weight.data[1:, :] = interpolated_patch_embeddings
                
            else:
                # Same size, just copy
                new_pos_embedding.weight.data.copy_(original_weights)
            
            # Replace the position embedding
            vision_tower.vision_model.embeddings.position_embedding = new_pos_embedding
            
            # Update position_ids
            vision_tower.vision_model.embeddings.position_ids = torch.arange(
                target_num_positions
            ).expand((1, -1)).to(self.device)
            
            print(f"✅ Vision tower resized to {target_image_size}x{target_image_size}")
            print(f"✅ Position embeddings: {target_num_positions} positions")
            
        else:
            print(f"✅ Vision tower already matches target size: {target_image_size}")
        
        # 🔥 ADDITIONAL FIX: Ensure the model config reflects the change
        if hasattr(self.model.config, 'vision_config'):
            if isinstance(self.model.config.vision_config, dict):
                self.model.config.vision_config['image_size'] = target_image_size
            else:
                self.model.config.vision_config.image_size = target_image_size
        
        # Verify the change took effect
        final_image_size = vision_tower.vision_model.config.image_size
        if final_image_size != target_image_size:
            print(f"⚠️ Warning: Vision tower size mismatch! Expected {target_image_size}, got {final_image_size}")
        else:
            print(f"✅ Verified: Vision tower now expects {final_image_size}x{final_image_size} images")

    def _setup_lora(self):
        print("🔄 Setting up LoRA...")
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'dense', 'query_key_value'],
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        model_dtype = torch.bfloat16 if self.config.use_bf16 else torch.float32
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = param.data.to(model_dtype)

        # ------------------------------------------------------------
        # Attach NaN/Inf debugging hooks to the diffusion UNet
        # ------------------------------------------------------------
        from policy_heads.models import ConditionalUnet1D

        def _add_nan_hooks(unet: ConditionalUnet1D):
            """Register forward hooks that yell when any tensor goes NaN/Inf."""
            def make_hook(layer_name: str):
                def hook(module, inputs, output):
                    with torch.no_grad():
                        t = None
                        if isinstance(output, torch.Tensor):
                            t = output
                        elif isinstance(output, (list, tuple)) and len(output):
                            t = output[0]
                        if t is not None and (torch.isnan(t).any() or torch.isinf(t).any()):
                            print(f"🚨 NaN/Inf detected inside ConditionalUnet1D → layer {layer_name} | range [" \
                                  f"{t.min().item():.4f}, {t.max().item():.4f}]")
                            # Once triggered, remove hook to avoid flooding
                            return
                return hook
            for n, m in unet.named_modules():
                m.register_forward_hook(make_hook(n))

        # Apply only if the model actually has the diffusion head
        try:
            if hasattr(self.model, 'module'):
                _maybe_model = self.model.module  # de-parallelise
            else:
                _maybe_model = self.model
            if hasattr(_maybe_model, 'base_model'):
                _maybe_model = _maybe_model.base_model  # unwrap PEFT
            if hasattr(_maybe_model, 'embed_out') and isinstance(_maybe_model.embed_out, ConditionalUnet1D):
                _add_nan_hooks(_maybe_model.embed_out)
                print("🔬 NaN/Inf hooks attached to ConditionalUnet1D")
        except Exception as e:
            print("⚠️ Failed to attach NaN hooks:", e)

        # ------------------------------------------------------------------
        # Keep the diffusion UNet (embed_out) in full-precision FP32 even when
        # the rest of the model runs in BF16 – this avoids BF16 over/underflow
        # during the very large dynamic ranges we have observed.
        # ------------------------------------------------------------------
        if self.config.use_bf16:
            moved = 0
            for mod_name, mod in self.model.named_modules():
                if 'embed_out' in mod_name:
                    mod.to(dtype=torch.float32)
                    moved += 1
            if moved:
                print(f"🔧 Cast {moved} embed_out sub-modules to FP32 for stability")

    def _add_nan_hooks(self, unet: ConditionalUnet1D):
        # ... existing _add_nan_hooks code ...
        pass

    def _initialize_parameters(self):
        """Initializes trainable parameters."""
        # This function is now mostly for non-DeepSpeed paths or if specific init is needed after DeepSpeed.
        # DeepSpeed's initialize function can handle parameter initialization based on its config.
        print("🔄 Initializing trainable parameters...")
        
        if not self.model_engine: # Only apply if not DeepSpeed
            # For non-DeepSpeed: apply custom initializations if needed
            # Ensure new tokens are initialized (handled in _setup_model_and_tokenizer)
            pass # Existing code might be here if you have specific initialization

    def _setup_optimizer(self):
        """Sets up the optimizer and (optional) learning rate scheduler."""
        print("🔄 Setting up optimizer and scheduler...")
        # Parameters that require gradient
        params = [p for p in self.model.parameters() if p.requires_grad]

        # Filter out diffusion head parameters if not training it
        if not self.config.train_diffusion_head:
            params = [p for p in params if 'embed_out' not in p._name]

        self.optimizer = torch.optim.AdamW(params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        print(f"✅ Optimizer: AdamW, LR: {self.config.learning_rate}, Weight decay: {self.config.weight_decay}")

        # GradScaler for mixed precision (non-DeepSpeed only)
        if self.config.use_bf16 and not self.model_engine:
            self.scaler = torch.cuda.amp.GradScaler()
            print("✅ Using GradScaler for BF16 mixed precision.")

    def _setup_deepspeed_or_optimizer(self):
        """Conditionally sets up DeepSpeed or falls back to standard optimizer/scheduler."""
        if self.config.deepspeed_config_path:
            print(f"🔧 DeepSpeed config path found: {self.config.deepspeed_config_path}")
            # DeepSpeed requires its own arguments object
            ds_args = argparse.Namespace()
            ds_args.deepspeed = self.config.deepspeed_config_path
            # These are typically set by the deepspeed launcher, but define them for `deepspeed.initialize`
            ds_args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            ds_args.world_size = int(os.environ.get("WORLD_SIZE", 1))
            ds_args.rank = int(os.environ.get("RANK", 0))

            print("🔧 Initializing DeepSpeed engine...")
            # DeepSpeed handles optimizer, scheduler, and model wrapping internally
            self.model_engine, self.optimizer, _, self.scheduler = deepspeed.initialize(
                args=ds_args,             # Pass the DeepSpeed arguments
                model=self.model,
                model_parameters=[p for p in self.model.parameters() if p.requires_grad], # Only trainable params
                config_params=None,       # Config loaded via args.deepspeed
            )
            # Ensure self.device is updated to the DeepSpeed assigned local rank
            self.device = self.model_engine.local_rank
            print(f"✅ DeepSpeed initialized with ZeRO stage {self.model_engine.zero_optimization_stage()} on device {self.device}")
        else:
            print("⚠️ No DeepSpeed config path provided. Falling back to standard optimizer setup.")
            self._setup_optimizer() # Call your existing _setup_optimizer method

    def _setup_dataloader(self):
        print("🔄 Creating DataLoader...")
        # Reuse dataset built earlier
        dataset = getattr(self, 'dataset', None)
        if dataset is None:
            dataset, _ = get_dataset_and_stats(self.config)
            self.dataset = dataset
        collate_fn = CollateFn(self.stats, self.tokenizer, self.config)
        self.data_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers
        )
        print(f"✅ DataLoader created with {len(self.data_loader)} batches")

    def save_checkpoint(self, step, loss_value, is_best=False):
        checkpoint_dir = os.path.join(self.config.output_dir, f"step_{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # --- Begin DeepSpeed-aware saving logic ---
        if self.model_engine: # If DeepSpeed is initialized
            # DeepSpeed handles saving sharded model, optimizer, and scheduler states.
            # It saves to a step-specific directory within checkpoint_dir.
            self.model_engine.save_checkpoint(checkpoint_dir, tag=f"step_{step}")

            # Only rank 0 saves the consolidated FP32 model for easier loading without DeepSpeed
            if self.model_engine.global_rank == 0: # Ensure only one process saves this
                print("Consolidating DeepSpeed checkpoint to FP32 model...")
                consolidated_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
                # `get_fp32_state_dict_from_zero_checkpoint` needs the path to the sharded checkpoint dir
                # DeepSpeed saves into `checkpoint_dir/tag/` (e.g., `step_X/step_X/`) or just `checkpoint_dir/tag/`
                # Check for the correct DeepSpeed internal path structure
                deepspeed_ckpt_path = os.path.join(checkpoint_dir, f"step_{step}")
                if os.path.exists(deepspeed_ckpt_path):
                    fp32_state_dict = get_fp32_state_dict_from_zero_checkpoint(deepspeed_ckpt_path)
                    torch.save(fp32_state_dict, consolidated_path)
                    print(f"💾 Saved consolidated FP32 model to {consolidated_path}")
                else:
                    print(f"⚠️ DeepSpeed checkpoint path not found for consolidation: {deepspeed_ckpt_path}")

            print(f"💾 Saved DeepSpeed checkpoint at step {step} to {checkpoint_dir}")
            
        else: # Fallback to standard PyTorch saving (original logic)
            # Save LoRA adapter weights (if applicable, assuming self.model is a PeftModel)
            if hasattr(self.model, 'save_pretrained'):
                self.model.save_pretrained(checkpoint_dir)
            else:
                # For non-PEFT models, save the full state dict
                torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, "pytorch_model.bin"))

            # Save diffusion head weights separately when training enabled (original logic)
            if self.config.train_diffusion_head:
                diffusion_state_dict = {}
                for name, param in self.model.named_parameters():
                    # Ensure it's the trainable diffusion head
                    if 'embed_out' in name and param.requires_grad:
                        # Remove any PEFT prefixes if present
                        clean_name = name.replace('base_model.model.', '')
                        diffusion_state_dict[clean_name] = param.data.clone()
                
                if diffusion_state_dict:
                    diffusion_path = os.path.join(checkpoint_dir, "diffusion_head.bin")
                    torch.save(diffusion_state_dict, diffusion_path)
                    print(f"💾 Saved {len(diffusion_state_dict)} diffusion head parameters to {diffusion_path}")
                    
                    # Also save to configurable diffusion head directory for latest version
                    vla_diff_dir = self.config.diffusion_head_save_dir
                    os.makedirs(vla_diff_dir, exist_ok=True) # Ensure dir exists
                    torch.save(diffusion_state_dict, os.path.join(vla_diff_dir, f"diffusion_head_step_{step}.bin"))
                    torch.save(diffusion_state_dict, os.path.join(vla_diff_dir, "diffusion_head_latest.bin"))
                    print(f"💾 Updated latest diffusion head: {os.path.join(vla_diff_dir, 'diffusion_head_latest.bin')}")
                else:
                    print("⚠️ No trainable diffusion head parameters found to save!")

            # Save training state (optimizer, scaler, config) for non-DeepSpeed runs
            checkpoint_data = {
                'step': step,
                'loss': loss_value,
                'config': asdict(self.config),
            }
            if self.optimizer: # Only save if optimizer exists (not handled by DeepSpeed)
                checkpoint_data['optimizer__state_dict'] = self.optimizer.state_dict()
            if self.scaler: # Only save if scaler exists (not handled by DeepSpeed)
                checkpoint_data['scaler_state_dict'] = self.scaler.state_dict()
            
            torch.save(checkpoint_data, os.path.join(checkpoint_dir, "training_state.pt"))
            print(f"💾 Saved standard checkpoint at step {step} to {checkpoint_dir}")

    def _adjust_lr(self, step):
        """Manually adjust learning rate based on warmup schedule."""
        if step < self.config.warmup_steps:
            lr_scale = float(step) / float(max(1, self.config.warmup_steps))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config.learning_rate * lr_scale
        # No decay, just warmup then constant LR

    def train(self):
        self._setup()
        # Set training mode based on whether DeepSpeed is active
        if self.model_engine:
            self.model_engine.train()
        else:
            self.model.train()
        global_step = 0
        consecutive_nan_count = 0
        max_consecutive_nans = 3
        
        # Calculate batches per epoch for epoch/batch display
        batches_per_epoch = len(self.data_loader)
        total_epochs = (self.config.max_steps + batches_per_epoch - 1) // batches_per_epoch  # Ceiling division
        
        # 🔥 MEMORY MANAGEMENT: Set up memory optimization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Set memory fraction to prevent OOM (very conservative)
            # DeepSpeed manages memory, avoid setting per_process_memory_fraction if DS is used
            if not self.model_engine:
                torch.cuda.set_per_process_memory_fraction(0.6)
                print(f"🔧 Set CUDA memory fraction to 0.6")
        
        print(f"🚀 Starting training: {batches_per_epoch} batches/epoch, ~{total_epochs} epochs, {self.config.max_steps} total steps")
        
        with tqdm(total=self.config.max_steps, desc="Training") as pbar:
            # --------------------------------------------------------------
            # One-batch forward/backward sanity test before full loop
            # --------------------------------------------------------------
            try:
                test_batch = next(iter(self.data_loader))
                for k, v in test_batch.items():
                    if torch.is_tensor(v):
                        test_batch[k] = v.to(self.device)

                with torch.cuda.amp.autocast(enabled=self.config.use_bf16):
                    out = self.model(**test_batch)

                    if hasattr(out, 'logits') and out.logits is not None:
                        # Language modeling path
                        logits = out.logits  # (B,T,V)
                        tgt = test_batch.get("labels")
                        if tgt is None:
                            raise ValueError("labels tensor missing from batch during sanity-test")
                        tgt = tgt.to(logits.device)

                        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                        test_loss = loss_fct(logits.view(-1, logits.size(-1)), tgt.view(-1))
                        
                    elif hasattr(out, 'loss') and out.loss is not None:
                        # Action prediction path
                        test_loss = out.loss
                        
                    else:
                        raise ValueError("Model outputs don't contain expected loss or logits during sanity-test")

                print(f"🔬 Sanity-test loss: {test_loss.item():.4f} | requires_grad={test_loss.requires_grad}")

                if test_loss.requires_grad:
                    # Backward pass handled by DeepSpeed if active
                    if self.model_engine:
                        self.model_engine.backward(test_loss)
                    else:
                        test_loss.backward()
                        # Clear gradients for next step (DeepSpeed handles this automatically)
                        self.optimizer.zero_grad(set_to_none=True)
                    print("✅ Sanity backward pass succeeded")
                else:
                    print("⚠️  Sanity loss does not require grad; check trainable params list")
                    
                # 🔥 MEMORY CLEANUP after sanity test
                del test_loss, out, test_batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"⚠️  Sanity-test skipped due to error: {e}")

            while global_step < self.config.max_steps:
                for batch_idx, batch in enumerate(self.data_loader):
                    if global_step >= self.config.max_steps:
                        break

                    # Calculate epoch and batch within epoch
                    current_epoch = global_step // batches_per_epoch
                    batch_in_epoch = global_step % batches_per_epoch

                    for key, value in batch.items():
                        if torch.is_tensor(value):
                            batch[key] = value.to(self.device)
                    
                    # 🔥 MEMORY MANAGEMENT: Regular cleanup every N steps
                    if global_step > 0 and self.config.max_memory_cleanup_steps > 0 and global_step % self.config.max_memory_cleanup_steps == 0:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            # Get memory stats
                            allocated = torch.cuda.memory_allocated() / 1024**3
                            reserved = torch.cuda.memory_reserved() / 1024**3
                            print(f"🧹 Memory cleanup at epoch {current_epoch}, batch {batch_in_epoch}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
                    
                    with torch.cuda.amp.autocast(enabled=self.config.use_bf16):
                        outputs = self.model(**batch)

                        # Check if outputs are valid
                        if outputs is None:
                            print(f"💥 Model outputs are None at epoch {current_epoch}, batch {batch_in_epoch}, skipping batch.")
                            consecutive_nan_count += 1
                            if consecutive_nan_count >= max_consecutive_nans:
                                print("🛑 Stopping training due to consecutive None outputs.")
                                return
                            continue

                        # Handle different model types (VLA vs pure language models)
                        if hasattr(outputs, 'logits') and outputs.logits is not None:
                            # Standard language modeling path
                            logits = outputs.logits
                            
                            # Check if logits contain NaN/Inf
                            if torch.isnan(logits).any() or torch.isinf(logits).any():
                                print(f"💥 NaN/Inf detected in logits at epoch {current_epoch}, batch {batch_in_epoch}, skipping batch.")
                                consecutive_nan_count += 1
                                if consecutive_nan_count >= max_consecutive_nans:
                                    print("🛑 Stopping training due to consecutive NaN logits.")
                                    return  
                                continue
                                
                            tgt = batch["labels"].to(logits.device)
                            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                            loss = loss_fct(logits.view(-1, logits.size(-1)), tgt.view(-1))
                            
                        elif hasattr(outputs, 'loss') and outputs.loss is not None:
                            # Action prediction path
                            loss = outputs.loss
                            
                        else:
                            raise ValueError("Model outputs don't contain expected loss or logits.")

                    # Backward pass and optimizer step
                    if self.model_engine: # DeepSpeed handles backward, gradient accumulation, and optimizer step
                        self.model_engine.backward(loss)
                        self.model_engine.step() # Includes optimizer.step() and zero_grad()
                    else: # Regular training fallback
                        if self.scaler:
                            self.scaler.scale(loss).backward()
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            loss.backward()
                            self.optimizer.step()
                        
                        self.optimizer.zero_grad() # Clear gradients for next step
                    
                    # Logging
                    if global_step % 100 == 0:
                        print(f"Step {global_step}: Loss = {loss.item():.4f}")
                    
                    # Save checkpoint
                    if global_step % self.config.save_steps == 0 and global_step > 0:
                        self.save_checkpoint(global_step, loss.item()) # Call the unified save_checkpoint
                    
                    global_step += 1

            # After an epoch (if not yet max_steps), empty cache
            if global_step < self.config.max_steps and torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"\n🎉 Training finished after {current_epoch + 1} epochs.")
        self.save_checkpoint(global_step, loss.item())
        
        # 🔥 FINAL MEMORY CLEANUP
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

def parse_args():
    """Parses command line arguments (simplified for YAML config)."""
    parser = argparse.ArgumentParser(description="TinyVLA LoRA Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional: override output directory from config.")
    return parser.parse_args()

def main():
    """Modified main function with DeepSpeed support controlled by config."""
    args = parse_args() # Use the simplified parse_args()
    
    # Load config from YAML
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Override output_dir if provided via command line (important for consistency)
    if args.output_dir:
        config_dict['output_dir'] = args.output_dir
    
    config = TrainingConfig(**config_dict)
    
    # Initialize distributed training if DeepSpeed config path is provided
    if config.deepspeed_config_path:
        # deepspeed.init_distributed() will initialize the process group
        # It reads LOCAL_RANK, WORLD_SIZE, RANK from environment variables
        # which are set by the `deepspeed` launcher.
        deepspeed.init_distributed()
        print("✅ DeepSpeed distributed environment initialized.")
    
    # Create trainer with just the config object (args are now handled internally)
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main() 