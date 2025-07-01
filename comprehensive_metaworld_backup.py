#!/usr/bin/env python3
"""
Comprehensive backup script for short-MetaWorld dataset to Hugging Face Hub
Preserves folder structure, includes task prompts, and provides loading code.

Dataset structure:
- Images: datasets/short-MetaWorld/short-MetaWorld/img_only/
- R3M processed: datasets/short-MetaWorld/short-MetaWorld/r3m-processed/
- Task prompts: datasets/mt50_task_prompts.json
"""

import os
import shutil
import json
import pickle
from pathlib import Path
from huggingface_hub import HfApi, create_repo
import subprocess

def check_hf_login():
    """Check if user is logged into Hugging Face"""
    try:
        result = subprocess.run(['huggingface-cli', 'whoami'], 
                               capture_output=True, text=True)
        if "Not logged in" in result.stdout:
            print("❌ You need to log into Hugging Face first!")
            print("Run: huggingface-cli login")
            return False
        else:
            print(f"✅ Logged in to Hugging Face: {result.stdout.strip()}")
            return True
    except Exception as e:
        print(f"❌ Error checking HF login: {e}")
        return False

def create_dataset_loader():
    """Create the dataset loader code"""
    loader_code = '''#!/usr/bin/env python3
"""
Short-MetaWorld Dataset Loader
Loads the dataset with proper structure preservation and task prompts.
"""

import os
import pickle
import json
import glob
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

class ShortMetaWorldDataset(Dataset):
    """
    Short-MetaWorld dataset loader that preserves the original structure.
    
    Args:
        data_root (str): Path to the dataset root directory
        task_list (list): List of tasks to load (default: all available tasks)
        image_size (int): Target image size for transforms (default: 224)
        transform (callable): Optional custom transform
        load_prompts (bool): Whether to load task prompts (default: True)
    """
    
    def __init__(self, data_root, task_list=None, image_size=224, transform=None, load_prompts=True):
        self.data_root = Path(data_root)
        self.image_size = image_size
        
        # Load task prompts
        self.prompts = {}
        if load_prompts:
            prompt_file = self.data_root / "mt50_task_prompts.json"
            if prompt_file.exists():
                with open(prompt_file, 'r') as f:
                    self.prompts = json.load(f)
                print(f"📖 Loaded {len(self.prompts)} task prompts")
            else:
                print("⚠️ Task prompts not found, using fallback prompts")
        
        # Set up paths
        self.img_root = self.data_root / "short-MetaWorld" / "short-MetaWorld" / "img_only"
        self.data_pkl_root = self.data_root / "short-MetaWorld" / "r3m-processed" / "r3m_MT10_20"
        
        # Discover available tasks
        available_tasks = []
        if self.data_pkl_root.exists():
            for pkl_file in self.data_pkl_root.glob("*.pkl"):
                task_name = pkl_file.stem
                if (self.img_root / task_name).exists():
                    available_tasks.append(task_name)
        
        # Filter tasks if task_list provided
        if task_list is not None:
            self.tasks = [task for task in task_list if task in available_tasks]
        else:
            self.tasks = available_tasks
        
        print(f"📊 Loading {len(self.tasks)} tasks: {self.tasks}")
        
        # Default transform
        if transform is None:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
            ])
        else:
            self.transform = transform
        
        # Load all trajectories
        self.trajectories = self._load_trajectories()
        print(f"✅ Loaded {len(self.trajectories)} trajectory steps")
    
    def _load_trajectories(self):
        """Load all trajectory data"""
        all_trajectories = []
        
        for task in self.tasks:
            # Load pickle data
            pkl_path = self.data_pkl_root / f"{task}.pkl"
            if not pkl_path.exists():
                print(f"⚠️ Pickle file not found: {pkl_path}")
                continue
                
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            
            # Get image directories
            task_img_dir = self.img_root / task
            if not task_img_dir.exists():
                print(f"⚠️ Image directory not found: {task_img_dir}")
                continue
            
            # Process each trajectory
            traj_dirs = sorted(task_img_dir.glob("*"), key=lambda x: int(x.name))
            
            for traj_idx, traj_dir in enumerate(traj_dirs):
                if traj_idx >= len(data['actions']):
                    continue
                
                # Get image paths
                img_paths = sorted(traj_dir.glob("*.jpg"), 
                                 key=lambda x: int(x.stem))
                
                num_steps = len(data['actions'][traj_idx])
                num_images = len(img_paths)
                
                # Use minimum length to handle mismatched data
                min_steps = min(num_images, num_steps)
                if min_steps < 1:
                    continue
                
                # Create trajectory entries
                for step_idx in range(min_steps):
                    trajectory_entry = {
                        'task_name': task,
                        'trajectory_id': traj_idx,
                        'step_id': step_idx,
                        'image_path': str(img_paths[step_idx]),
                        'state': data['state'][traj_idx][step_idx][:7],  # First 7 dims
                        'action': data['actions'][traj_idx][step_idx],
                        'prompt': self._get_prompt(task)
                    }
                    all_trajectories.append(trajectory_entry)
        
        return all_trajectories
    
    def _get_prompt(self, task_name):
        """Get prompt for a task"""
        if task_name in self.prompts:
            # Use simple prompt by default
            return self.prompts[task_name].get('simple', f"Perform the task: {task_name.replace('-', ' ')}")
        else:
            return f"Perform the task: {task_name.replace('-', ' ')}"
    
    def get_task_info(self, task_name):
        """Get comprehensive task information"""
        if task_name in self.prompts:
            return self.prompts[task_name]
        return {"simple": f"Perform the task: {task_name.replace('-', ' ')}"}
    
    def get_available_tasks(self):
        """Get list of available tasks"""
        return self.tasks.copy()
    
    def get_dataset_stats(self):
        """Get dataset statistics"""
        task_counts = {}
        for traj in self.trajectories:
            task = traj['task_name']
            task_counts[task] = task_counts.get(task, 0) + 1
        
        return {
            'total_steps': len(self.trajectories),
            'num_tasks': len(self.tasks),
            'task_step_counts': task_counts,
            'tasks': self.tasks
        }
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        """Get a single trajectory step"""
        traj = self.trajectories[idx]
        
        # Load and transform image
        image = Image.open(traj['image_path']).convert("RGB")
        image_tensor = self.transform(image)
        
        # Convert to tensors
        state = torch.tensor(traj['state'], dtype=torch.float32)
        action = torch.tensor(traj['action'], dtype=torch.float32)
        
        return {
            'image': image_tensor,
            'state': state,
            'action': action,
            'prompt': traj['prompt'],
            'task_name': traj['task_name'],
            'trajectory_id': traj['trajectory_id'],
            'step_id': traj['step_id']
        }

# Example usage functions
def load_short_metaworld(data_root, tasks=None, image_size=224):
    """
    Convenience function to load the dataset.
    
    Args:
        data_root (str): Path to dataset root
        tasks (list): List of tasks to load (None for all)
        image_size (int): Image size for transforms
    
    Returns:
        ShortMetaWorldDataset: The loaded dataset
    """
    return ShortMetaWorldDataset(
        data_root=data_root,
        task_list=tasks,
        image_size=image_size
    )

def get_mt10_tasks():
    """Get the MT10 task list"""
    return [
        "reach-v2", "push-v2", "pick-place-v2", "door-open-v2", "drawer-open-v2",
        "drawer-close-v2", "button-press-topdown-v2", "button-press-v2", 
        "button-press-wall-v2", "button-press-topdown-wall-v2"
    ]

def demo_usage():
    """Demonstrate how to use the dataset"""
    print("📖 Short-MetaWorld Dataset Usage Example")
    print("=" * 50)
    
    # Load dataset
    dataset = load_short_metaworld("./", tasks=get_mt10_tasks())
    
    # Print stats
    stats = dataset.get_dataset_stats()
    print(f"📊 Dataset Statistics:")
    print(f"   Total steps: {stats['total_steps']}")
    print(f"   Number of tasks: {stats['num_tasks']}")
    print(f"   Available tasks: {stats['tasks']}")
    
    # Get a sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\\n📋 Sample data:")
        print(f"   Image shape: {sample['image'].shape}")
        print(f"   State shape: {sample['state'].shape}")
        print(f"   Action shape: {sample['action'].shape}")
        print(f"   Task: {sample['task_name']}")
        print(f"   Prompt: {sample['prompt']}")

if __name__ == "__main__":
    demo_usage()
'''
    return loader_code

def create_requirements():
    """Create requirements.txt for the dataset"""
    requirements = """# Short-MetaWorld Dataset Requirements
torch>=1.9.0
torchvision>=0.10.0
Pillow>=8.0.0
numpy>=1.20.0
"""
    return requirements

def create_dataset_card():
    """Create comprehensive dataset card"""
    card_content = """---
license: mit
task_categories:
- robotics
- reinforcement-learning
tags:
- metaworld
- robotics
- manipulation
- multi-task
- r3m
- vision-language
- imitation
size_categories:
- 1K<n<10K
language:
- en
pretty_name: Short-MetaWorld Dataset
dataset_info:
  features:
  - name: image
    dtype: image
  - name: state
    dtype: 
      sequence: float32
  - name: action
    dtype:
      sequence: float32
  - name: prompt
    dtype: string
  - name: task_name
    dtype: string
  splits:
  - name: train
    num_bytes: 1900000000
    num_examples: 40000
  download_size: 1900000000
  dataset_size: 1900000000
---

# Short-MetaWorld Dataset

## Overview

Short-MetaWorld is a curated dataset from Meta-World containing **Multi-Task 10 (MT10)** and **Meta-Learning 10 (ML10)** tasks with **100 successful trajectories per task** and **20 steps per trajectory**. This dataset is specifically designed for multi-task robot learning, imitation learning, and vision-language robotics research.

## 🚀 Quick Start

```python
from short_metaworld_loader import load_short_metaworld
from torch.utils.data import DataLoader

# Load the dataset
dataset = load_short_metaworld("./", image_size=224)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Get a sample
sample = dataset[0]
print(f"Image shape: {sample['image'].shape}")
print(f"State: {sample['state']}")
print(f"Action: {sample['action']}")
print(f"Task: {sample['task_name']}")
print(f"Prompt: {sample['prompt']}")
```

## 📁 Dataset Structure

```
short-MetaWorld/
├── README.txt                     # Original dataset documentation
├── short-MetaWorld/
│   ├── img_only/                    # 224x224 RGB images
│   │   ├── button-press-topdown-v2/
│   │   │   ├── 0/                   # Trajectory 0
│   │   │   │   ├── 0.jpg           # Step 0
│   │   │   │   ├── 1.jpg           # Step 1
│   │   │   │   └── ...
│   │   │   ├── 1/                   # Trajectory 1
│   │   │   └── ...
│   │   ├── door-open-v2/
│   │   └── ...
│   └── r3m-processed/              # R3M processed features
│       └── r3m_MT10_20/
│           ├── button-press-topdown-v2.pkl
│           ├── door-open-v2.pkl
│           └── ...
└── r3m-processed/                  # Additional R3M data
    └── r3m_MT10_20/
├── mt50_task_prompts.json          # Task descriptions & prompts
├── short_metaworld_loader.py       # Dataset loader
└── requirements.txt
```

## 🎯 Tasks Included

### Multi-Task 10 (MT10)
- `button-press-topdown-v2` - Press button from above
- `door-open-v2` - Open door by pulling handle  
- `drawer-close-v2` - Close drawer
- `drawer-open-v2` - Open drawer
- `peg-insert-side-v2` - Insert peg into hole
- `pick-place-v2` - Pick up object and place on target

### Meta-Learning 10 (ML10)
Additional tasks for meta-learning evaluation.

## 📊 Data Format

- **Images**: 224×224 RGB images in JPEG format
- **States**: 7-dimensional robot state vectors (joint positions)
- **Actions**: 4-dimensional continuous control actions
- **Prompts**: Natural language task descriptions in 3 styles:
  - `simple`: Brief task description
  - `detailed`: Comprehensive task explanation  
  - `task_specific`: Context-specific variations
- **R3M Features**: Pre-processed visual representations using R3M model

## 💾 Loading the Dataset

The dataset comes with a comprehensive loader (`short_metaworld_loader.py`):

```python
# Load specific tasks
mt10_tasks = [
    "reach-v2", "push-v2", "pick-place-v2", "door-open-v2", 
    "drawer-open-v2", "drawer-close-v2", "button-press-topdown-v2",
    "button-press-v2", "button-press-wall-v2", "button-press-topdown-wall-v2"
]
dataset = load_short_metaworld("./", tasks=mt10_tasks)

# Load all available tasks
dataset = load_short_metaworld("./")

# Get dataset statistics
stats = dataset.get_dataset_stats()
print(f"Total steps: {stats['total_steps']}")
print(f"Tasks: {stats['tasks']}")

# Get task-specific prompts
task_info = dataset.get_task_info("pick-place-v2")
print(task_info['detailed'])  # Detailed task description
```

## 🔬 Research Applications

This dataset is designed for:

- **Multi-task Reinforcement Learning**: Train policies across multiple manipulation tasks
- **Imitation Learning**: Learn from demonstration trajectories
- **Vision-Language Robotics**: Connect visual observations with natural language instructions
- **Meta-Learning**: Adapt quickly to new manipulation tasks
- **Robot Policy Training**: End-to-end visuomotor control

## 📈 Dataset Statistics

- **Total trajectories**: 2,000 (100 per task × 20 tasks)
- **Total steps**: ~40,000 (20 steps per trajectory)
- **Image resolution**: 224×224 RGB
- **State dimension**: 7 (robot joint positions)
- **Action dimension**: 4 (continuous control)
- **Dataset size**: ~1.9GB

## 🛠️ Installation

```bash
pip install torch torchvision Pillow numpy
```

## 📖 Citation

If you use this dataset, please cite:

```bibtex
@inproceedings{yu2020meta,
  title={Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning},
  author={Yu, Tianhe and Quillen, Deirdre and He, Zhanpeng and Julian, Ryan and Hausman, Karol and Finn, Chelsea and Levine, Sergey},
  booktitle={Conference on robot learning},
  pages={1094--1100},
  year={2020},
  organization={PMLR}
}

@inproceedings{nair2022r3m,
  title={R3M: A Universal Visual Representation for Robot Manipulation},
  author={Nair, Suraj and Rajeswaran, Aravind and Kumar, Vikash and Finn, Chelsea and Gupta, Abhinav},
  booktitle={Conference on Robot Learning},
  pages={892--902},
  year={2023},
  organization={PMLR}
}
```

## 📧 Contact

- Original dataset: liangzx@connect.hku.hk
- Questions about this upload: Open an issue in the dataset repository

## ⚖️ License

MIT License - See LICENSE file for details.
"""
    return card_content

def backup_comprehensive_dataset(repo_name: str, private: bool = False):
    """
    Create comprehensive backup with proper structure and loading code
    
    Args:
        repo_name: Name for the HF repository
        private: Whether to make repository private
    """
    
    # Check authentication
    if not check_hf_login():
        return False
    
    # Check paths
    dataset_path = Path("datasets/short-MetaWorld")
    prompts_path = Path("datasets/mt50_task_prompts.json")
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found at {dataset_path}")
        return False
    
    if not prompts_path.exists():
        print(f"❌ Task prompts not found at {prompts_path}")
        return False
    
    print(f"📊 Dataset found at {dataset_path}")
    print(f"📖 Task prompts found at {prompts_path}")
    
    try:
        # Initialize HF API
        api = HfApi()
        
        # Get current user to construct proper repo_id
        user_info = api.whoami()
        username = user_info['name']
        
        # Construct full repository ID
        if '/' not in repo_name:
            full_repo_id = f"{username}/{repo_name}"
        else:
            full_repo_id = repo_name
            
        print(f"🏗️ Creating repository: {full_repo_id}")
        
        # Create repository with better error handling
        try:
            create_repo(
                repo_id=full_repo_id,
                token=api.token,
                repo_type="dataset",
                private=private,
                exist_ok=True  # Don't fail if repo already exists
            )
            print(f"✅ Repository created/verified: {full_repo_id}")
        except Exception as repo_error:
            print(f"❌ Failed to create repository: {repo_error}")
            return False
        
        # Create all necessary files
        print("📝 Creating dataset files...")
        
        # 1. Dataset card
        with open("README.md", "w") as f:
            f.write(create_dataset_card())
        
        # 2. Dataset loader
        with open("short_metaworld_loader.py", "w") as f:
            f.write(create_dataset_loader())
        
        # 3. Requirements
        with open("requirements.txt", "w") as f:
            f.write(create_requirements())
        
        # Upload files with better error handling
        print("📤 Uploading dataset components...")
        
        # Upload README
        try:
            api.upload_file(
                path_or_fileobj="README.md",
                path_in_repo="README.md",
                repo_id=full_repo_id,
                repo_type="dataset",
                commit_message="Add comprehensive dataset card"
            )
            print("✅ README uploaded")
        except Exception as e:
            print(f"❌ Failed to upload README: {e}")
            return False
        
        # Upload loader
        try:
            api.upload_file(
                path_or_fileobj="short_metaworld_loader.py",
                path_in_repo="short_metaworld_loader.py",
                repo_id=full_repo_id,
                repo_type="dataset",
                commit_message="Add dataset loader code"
            )
            print("✅ Loader uploaded")
        except Exception as e:
            print(f"❌ Failed to upload loader: {e}")
            return False
        
        # Upload requirements
        try:
            api.upload_file(
                path_or_fileobj="requirements.txt",
                path_in_repo="requirements.txt",
                repo_id=full_repo_id,
                repo_type="dataset",
                commit_message="Add requirements"
            )
            print("✅ Requirements uploaded")
        except Exception as e:
            print(f"❌ Failed to upload requirements: {e}")
            return False
        
        # Upload task prompts
        try:
            api.upload_file(
                path_or_fileobj=str(prompts_path),
                path_in_repo="mt50_task_prompts.json",
                repo_id=full_repo_id,
                repo_type="dataset",
                commit_message="Add task prompts"
            )
            print("✅ Task prompts uploaded")
        except Exception as e:
            print(f"❌ Failed to upload task prompts: {e}")
            return False
        
        # Upload entire dataset folder with structure preserved
        print("📤 Uploading dataset files (this may take a while for 1.9GB)...")
        try:
            api.upload_folder(
                folder_path=str(dataset_path),
                repo_id=full_repo_id,
                repo_type="dataset",
                commit_message="Upload complete short-MetaWorld dataset with preserved structure",
                ignore_patterns=["*.pyc", "__pycache__/", ".DS_Store"]
            )
            print("✅ Dataset files uploaded")
        except Exception as e:
            print(f"❌ Failed to upload dataset files: {e}")
            return False
        
        print(f"✅ Successfully uploaded comprehensive dataset!")
        print(f"📍 Available at: https://huggingface.co/datasets/{full_repo_id}")
        print(f"🔗 Loader: https://huggingface.co/datasets/{full_repo_id}/blob/main/short_metaworld_loader.py")
        
        # Clean up temporary files
        for temp_file in ["README.md", "short_metaworld_loader.py", "requirements.txt"]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        return True
        
    except Exception as e:
        print(f"❌ Error uploading dataset: {e}")
        print(f"❌ Error type: {type(e).__name__}")
        import traceback
        print(f"❌ Full traceback: {traceback.format_exc()}")
        # Clean up on error
        for temp_file in ["README.md", "short_metaworld_loader.py", "requirements.txt"]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        return False

def main():
    """Main function"""
    print("🚀 Comprehensive Short-MetaWorld Dataset Backup")
    print("=" * 60)
    print("This script will upload:")
    print("  ✅ Complete folder structure (images + R3M data)")
    print("  ✅ Task prompts with descriptions")
    print("  ✅ Python loader code")
    print("  ✅ Comprehensive documentation")
    print("  ✅ Usage examples")
    print()
    
    # Get repository name
    default_repo = "short-metaworld-complete"
    print(f"📝 Repository name format: 'repo-name' (will become 'username/repo-name')")
    repo_name = input(f"Repository name (default: {default_repo}): ").strip()
    if not repo_name:
        repo_name = default_repo
    
    # Privacy setting
    private_input = input("Make repository private? (y/N): ").strip().lower()
    private = private_input in ['y', 'yes']
    
    print(f"\n📋 Configuration:")
    print(f"   Repository: {repo_name} (will be created as username/{repo_name})")
    print(f"   Private: {private}")
    print(f"   Size: ~1.9GB + loader code")
    
    confirm = input("\nProceed with comprehensive upload? (y/N): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("❌ Upload cancelled")
        return
    
    # Start backup
    success = backup_comprehensive_dataset(repo_name, private)
    
    if success:
        print("\n🎉 Comprehensive backup completed!")
        print(f"📍 Dataset: https://huggingface.co/datasets/username/{repo_name}")
        print("\n📖 Usage:")
        print(f"   from datasets import load_dataset")
        print(f"   # Download the loader")
        print(f"   # Then use: from short_metaworld_loader import load_short_metaworld")
        print(f"   # dataset = load_short_metaworld('./')")
    else:
        print("\n❌ Backup failed!")

if __name__ == "__main__":
    main() 