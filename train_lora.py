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
from dataclasses import dataclass, asdict
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    def _setup(self):
        """Initializes all components for training."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        self._setup_dataset_and_stats()
        self._setup_model_and_tokenizer()
        self._setup_lora()

        # Enable gradient checkpointing once, after LoRA has wrapped the model
        if self.config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self._initialize_parameters()
        self._setup_optimizer()
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

    def _initialize_parameters(self):
        print("🔧 Initializing diffusion head parameters with conservative scaling...")
        for name, param in self.model.named_parameters():
            if 'embed_out' in name and param.requires_grad:
                if 'weight' in name:
                    if len(param.shape) >= 2:
                        torch.nn.init.normal_(param.data, mean=0.0, std=0.001)
                    else:
                        torch.nn.init.normal_(param.data, mean=0.0, std=0.0001)
                elif 'bias' in name:
                    torch.nn.init.constant_(param.data, 0.0)
            elif 'lora' in name and param.requires_grad:
                if 'lora_A' in name:
                    torch.nn.init.normal_(param.data, mean=0.0, std=0.001)
                elif 'lora_B' in name:
                    torch.nn.init.zeros_(param.data)
        
        if self.config.train_diffusion_head:
            if self.config.diffusion_warmup_steps > 0:
                print(f"🕒 Diffusion head will stay FROZEN for the first {self.config.diffusion_warmup_steps} steps (warm-up).")
                for name, param in self.model.named_parameters():
                    if 'embed_out' in name:
                        param.requires_grad = False
                # We will unfreeze later in the train() loop.
                self._diffusion_unfrozen = False
            else:
                self._diffusion_unfrozen = True

            print("🔧 Applying GroupNorm epsilon fix...")
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.GroupNorm) and 'embed_out' in name:
                    module.eps = 1e-4
                elif isinstance(module, torch.nn.LayerNorm) and 'embed_out' in name:
                    module.eps = 1e-4
        else:
            self._diffusion_unfrozen = False

        # ------------------------------------------------------------------
        # Final safety sweep: replace any NaN / Inf weights or biases **anywhere**.
        # ------------------------------------------------------------------
        bad_param_count = 0
        for name, p in self.model.named_parameters():
            if torch.isnan(p.data).any() or torch.isinf(p.data).any():
                bad_param_count += 1
                print(f"🚑 Found NaN/Inf in parameter {name}; re-initialising.")
                if p.data.ndim >= 2:
                    torch.nn.init.xavier_uniform_(p.data)
                else:
                    torch.nn.init.zeros_(p.data)

        # ------------------------------------------------------------------
        # Additional sweep: extremely large finite weights (>1e3) in the
        # diffusion head can also destabilise forward passes while escaping
        # the NaN/Inf check.  Clamp by re-initialising those elements.
        # ------------------------------------------------------------------
        huge_param_count = 0
        for name, p in self.model.named_parameters():
            if 'embed_out' in name:
                big_mask = torch.abs(p.data) > 1e3
                if big_mask.any():
                    huge_param_count += 1
                    num_big = big_mask.sum().item()
                    print(f"🚑 Re-initialising {num_big} huge weights in {name}")
                    with torch.no_grad():
                        p.data[big_mask] = torch.normal(mean=0.0, std=0.001, size=(num_big,), device=p.data.device).view(-1)

        if bad_param_count == 0:
            print("✅ Parameter sweep: no NaN/Inf weights found after init.")
        else:
            print(f"✅ Parameter sweep: fixed {bad_param_count} parameters containing NaN/Inf.")

        if huge_param_count == 0:
            print("✅ Parameter sweep: no oversized weights found after init.")
        else:
            print(f"✅ Parameter sweep: rescaled {huge_param_count} parameters containing oversized weights.")

        # ------------------------------------------------------------------
        # Freeze diffusion head parameters completely if training is disabled
        # ------------------------------------------------------------------
        if not self.config.train_diffusion_head:
            frozen = 0
            for name, p in self.model.named_parameters():
                if 'embed_out' in name:
                    p.requires_grad = False
                    frozen += 1
            print(f"🧊 Frozen {frozen} embed_out parameters (train_diffusion_head=False)")

    def _setup_optimizer(self):
        print("🔄 Setting up optimizers...")
        
        # Separate diffusion head for a lower LR
        embed_out_params = []
        other_params = []
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                if 'embed_out' in n:
                    embed_out_params.append(p)
                else:
                    other_params.append(p)

        optimizer_params = []
        if other_params:
            optimizer_params.append({'params': other_params, 'lr': self.config.learning_rate, 'weight_decay': self.config.weight_decay})
        if embed_out_params:
            optimizer_params.append({'params': embed_out_params, 'lr': self.config.diffusion_learning_rate, 'weight_decay': self.config.weight_decay})
        
        self.optimizer = torch.optim.AdamW(optimizer_params, eps=1e-8)
        
        # We will handle learning rate scheduling manually
        self.scheduler = None 
        
        if self.config.use_bf16:
            self.scaler = torch.cuda.amp.GradScaler()

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
        
        # Save LoRA adapter weights
        self.model.save_pretrained(checkpoint_dir)
        
        # 🔥 CRITICAL FIX: Save diffusion head weights separately when training enabled
        if self.config.train_diffusion_head:
            diffusion_state_dict = {}
            for name, param in self.model.named_parameters():
                if 'embed_out' in name and param.requires_grad:
                    # Remove any PEFT prefixes to get the actual parameter name
                    clean_name = name.replace('base_model.model.', '')
                    diffusion_state_dict[clean_name] = param.data.clone()
            
            if diffusion_state_dict:
                # Save to step directory
                diffusion_path = os.path.join(checkpoint_dir, "diffusion_head.bin")
                torch.save(diffusion_state_dict, diffusion_path)
                print(f"💾 Saved {len(diffusion_state_dict)} diffusion head parameters to {diffusion_path}")
                
                # 🎯 Save to configurable diffusion head directory
                vla_diff_dir = self.config.diffusion_head_save_dir  # Use config parameter instead of hardcoded path
                os.makedirs(vla_diff_dir, exist_ok=True)
                
                # Save latest checkpoint
                latest_diff_path = os.path.join(vla_diff_dir, f"diffusion_head_step_{step}.bin")
                torch.save(diffusion_state_dict, latest_diff_path)
                
                # Save as "latest" for easy loading
                latest_symlink = os.path.join(vla_diff_dir, "diffusion_head_latest.bin")
                torch.save(diffusion_state_dict, latest_symlink)
                
                print(f"💾 Also saved diffusion head to {latest_diff_path}")
                print(f"💾 Updated latest diffusion head: {latest_symlink}")
            else:
                print("⚠️ No diffusion head parameters found to save!")
        
        checkpoint_data = {
            'step': step,
            'loss': loss_value,
            'optimizer__state_dict': self.optimizer.state_dict(),
            'config': asdict(self.config),
        }
        if self.scaler:
            checkpoint_data['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint_data, os.path.join(checkpoint_dir, "training_state.pt"))
        print(f"💾 Saved checkpoint at step {step} to {checkpoint_dir}")

    def _adjust_lr(self, step):
        """Manually adjust learning rate based on warmup schedule."""
        if step < self.config.warmup_steps:
            lr_scale = float(step) / float(max(1, self.config.warmup_steps))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config.learning_rate * lr_scale
        # No decay, just warmup then constant LR

    def train(self):
        self._setup()
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
                    test_loss.backward()
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
                            # Action prediction model path (VLA models)
                            loss = outputs.loss
                            print(f"🎯 Using action prediction loss at epoch {current_epoch}, batch {batch_in_epoch}: {loss.item():.4f}")
                            
                        else:
                            print(f"💥 Model outputs don't contain expected loss or logits at epoch {current_epoch}, batch {batch_in_epoch}, skipping batch.")
                            consecutive_nan_count += 1
                            if consecutive_nan_count >= max_consecutive_nans:
                                print("🛑 Stopping training due to consecutive invalid outputs.")
                                return
                            continue

                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"💥 NaN/Inf detected in loss at epoch {current_epoch}, batch {batch_in_epoch}, skipping.")
                        consecutive_nan_count += 1
                        if consecutive_nan_count >= max_consecutive_nans:
                            print("🛑 Stopping training due to consecutive NaN losses.")
                            return
                        continue
                    
                    consecutive_nan_count = 0
                    loss = loss / self.config.gradient_accumulation_steps
                    
                    if self.scaler:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    # 🔥 MEMORY CLEANUP: Delete intermediate tensors
                    del outputs
                    if torch.cuda.is_available():
                        # Only clear cache occasionally to avoid overhead (fix division by zero)
                        cleanup_interval = max(1, self.config.max_memory_cleanup_steps // 2) if self.config.max_memory_cleanup_steps > 0 else 50
                        if global_step % cleanup_interval == 0:
                            torch.cuda.empty_cache()

                    # ------------------------------------------------------
                    # Late-unfreeze of diffusion head after warm-up
                    # ------------------------------------------------------
                    if (not self._diffusion_unfrozen and 
                        global_step >= self.config.diffusion_warmup_steps):
                        print(f"🟢 Unfreezing diffusion head at step {global_step} (warm-up finished)")
                        for n,p in self.model.named_parameters():
                            if 'embed_out' in n:
                                p.requires_grad = True
                        self._diffusion_unfrozen = True
                        # No need to add new param groups; they already exist in the optimiser

                    if (global_step + 1) % self.config.gradient_accumulation_steps == 0:
                        # Manual LR adjustment
                        self._adjust_lr(global_step)

                        if self.scaler:
                            # Unscale first so that grad norms are in FP32
                            self.scaler.unscale_(self.optimizer)

                        # ------------------------------------------------------------------
                        # Gradient-norm monitoring.  If the total grad-norm is NaN / Inf or
                        # explodes beyond a huge threshold we skip the optimiser step to
                        # avoid corrupting the weights.
                        # ------------------------------------------------------------------
                        total_norm = torch.nn.utils.clip_grad_norm_(
                            filter(lambda p: p.requires_grad, self.model.parameters()),
                            self.config.gradient_clip_norm
                        )
                        if torch.isnan(total_norm) or torch.isinf(total_norm) or total_norm > 1e4:
                            print(f"🚨 Abnormal grad-norm {total_norm:.2e} – skipping optimiser step at epoch {current_epoch}, batch {batch_in_epoch}")
                            self.optimizer.zero_grad(set_to_none=True)
                            # Skip weight update and move on to next batch
                            continue
                        # ------------------------------------------------------------------

                        if self.scaler:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()
                        
                        # Prepare for next accumulation step
                        self.optimizer.zero_grad()
                    
                    pbar.update(1)
                    # Update progress bar with epoch/batch info instead of just step
                    if torch.is_tensor(loss):
                        try:
                            pbar.set_description(f"Epoch {current_epoch}, Batch {batch_in_epoch}/{batches_per_epoch-1}, Loss: {loss.item():.4f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}")
                        except (ValueError, TypeError):
                            pbar.set_description(f"Epoch {current_epoch}, Batch {batch_in_epoch}/{batches_per_epoch-1}, Loss: nan, LR: {self.optimizer.param_groups[0]['lr']:.6f}")
                    
                    if global_step > 0 and global_step % self.config.save_steps == 0:
                        print(f"💾 Saving checkpoint at epoch {current_epoch}, batch {batch_in_epoch} (step {global_step})")
                        self.save_checkpoint(global_step, loss.item())
                        
                        # 🔥 MEMORY CLEANUP after checkpoint save
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            print(f"🧹 Memory cleanup after checkpoint save")

                    global_step += 1

        print(f"\n🎉 Training finished after {current_epoch + 1} epochs.")
        self.save_checkpoint(global_step, loss.item())
        
        # 🔥 FINAL MEMORY CLEANUP
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

# 🔧 Safety patch: robustify ConditionalResidualBlock1D to zero-out NaNs early
from TinyVLA.policy_heads.models.droid_unet_diffusion import ConditionalResidualBlock1D as _CRB1D_Old

def _crb1d_forward_safe(self, x, cond):
    """A safer forward for ConditionalResidualBlock1D that eliminates NaN/Inf early."""
    out = self.blocks[0](x)
    # Encoding FiLM parameters
    embed = self.cond_encoder(cond)
    embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
    scale, bias = embed[:, 0, ...], embed[:, 1, ...]

    # Replace NaN/Inf in FiLM parameters (before clamp)
    if torch.isnan(scale).any() or torch.isinf(scale).any() or torch.isnan(bias).any() or torch.isinf(bias).any():
        print("🚑 Sanitising NaN/Inf in FiLM scale/bias – replaced with zeros")
        scale = torch.nan_to_num(scale, nan=0.0, posinf=5.0, neginf=-5.0)
        bias  = torch.nan_to_num(bias,  nan=0.0, posinf=5.0, neginf=-5.0)

    # Clamp FiLM parameters
    scale = torch.clamp(scale, -5.0, 5.0)
    bias  = torch.clamp(bias,  -5.0, 5.0)

    out = scale * out + bias
    # Clamp and sanitise
    out = torch.clamp(out, -6.0, 6.0)
    if torch.isnan(out).any() or torch.isinf(out).any():
        print("🚑 Sanitising NaN/Inf right after FiLM")
        out = torch.nan_to_num(out, nan=0.0, posinf=6.0, neginf=-6.0)

    out = self.blocks[1](out)
    # Final safety pass
    if torch.isnan(out).any() or torch.isinf(out).any():
        print("🚑 Sanitising NaN/Inf after second conv in ConditionalResidualBlock1D")
        out = torch.nan_to_num(out, nan=0.0, posinf=6.0, neginf=-6.0)

    return out + self.residual_conv(x)

# Monkey-patch
import types as _types
_CRB1D_Old.forward = _types.MethodType(_crb1d_forward_safe, _CRB1D_Old)
print("✅ Patched ConditionalResidualBlock1D.forward with NaN-safety")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="LoRA VLA Trainer for MetaWorld")
    parser.add_argument("--config", type=str, default="configs/train_lora.yaml",
                        help="Path to the YAML configuration file.")
    args = parser.parse_args()
    
    # Load configuration from YAML file
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Ensure LR fields are numeric floats (YAML may leave them as strings like '1e-5')
    for lr_key in ["learning_rate", "diffusion_learning_rate", "gradient_clip_norm", "weight_decay"]:
        if isinstance(config_dict.get(lr_key, 0), str):
            try:
                config_dict[lr_key] = float(config_dict[lr_key])
            except ValueError:
                raise ValueError(f"Config value {lr_key}={config_dict[lr_key]} is not a valid float")

    # ----------------------------------------------------------------------
    # Post-processing tweaks
    # 1. Auto-upgrade default 400M path if still present in config
    # 2. Resolve/append timestamp to output_dir if it already exists or
    #    contains literal shell placeholder $(date ...)
    # ----------------------------------------------------------------------

    # Upgrade backbone automatically if old 400M path was left in config
    if config_dict.get("model_path", "").endswith("Llava-Pythia-400M"):
        print("🔄 Detected 400M backbone in config – switching to 700M by default")
        config_dict["model_path"] = "lesjie/Llava-Pythia-700M"

    # Resolve output directory
    out_dir_raw = config_dict.get("output_dir", "./outputs/lora_run")
    # If bash substitution left literally in YAML, remove it and generate timestamp here
    if "$(" in out_dir_raw or os.path.exists(out_dir_raw):
        timestamp =  __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
        # Strip potential shell command part
        base_out = out_dir_raw.split("$(")[0].rstrip("/_") or "./outputs/lora_run"
        new_out_dir = f"{base_out}_{timestamp}"
        print(f"📁 Output directory resolved to {new_out_dir}")
        config_dict["output_dir"] = new_out_dir

    # Ensure a diffusion_head_save_dir is present
    if "diffusion_head_save_dir" not in config_dict or not config_dict["diffusion_head_save_dir"]:
        config_dict["diffusion_head_save_dir"] = os.path.join(config_dict["output_dir"], "diff_head")
        print(f"📁 diffusion_head_save_dir set to {config_dict['diffusion_head_save_dir']}")

    config = TrainingConfig(**config_dict)
    
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    # Set multiprocessing start method for CUDA safety
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
        print("✅ Set multiprocessing start method to 'spawn'")
    except RuntimeError:
        print("⚠️ Multiprocessing context already set.")
        pass
    main() 