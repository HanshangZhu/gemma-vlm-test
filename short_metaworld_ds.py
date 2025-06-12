# short_metaworld_ds.py
import os, glob, pickle
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms

class ShortMWDataset(Dataset):
    """Yield (image_tensor, prompt_str, action_tensor) using images from img_only and actions from r3m-processed .pkl files."""
    def __init__(self, root, task_list, split='train'):
        self.root = root
        self.items = []
        
        # Transform to resize and convert to tensor
        self.transform = transforms.Compose([
            transforms.Resize((336, 336)),  # Resize to match CLIP's expected size
            transforms.ToTensor()  # Convert to tensor in [0,1] range
        ])

        # Load prompt dict
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "tdesc", os.path.join(root, "task_description.py"))
        tdesc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tdesc)
        prompt_dict = tdesc.PROMPT_DICT  # task → str

        img_root = os.path.join(root, "short-MetaWorld", "img_only")
        act_root = os.path.join(root, "short-MetaWorld", "r3m-processed", "r3m_MT10_20")

        for task in task_list:
            # Load actions from R3M processed .pkl file
            pkl_path = os.path.join(act_root, f"{task}.pkl")
            if not os.path.exists(pkl_path):
                print(f"Warning: {pkl_path} not found")
                continue
            data = pickle.load(open(pkl_path, "rb"))
            actions = data['actions']  # (100, 20, 4)
            # Keep full trajectories instead of just first action
            # actions = actions[:, 0]  # (100, 4) - REMOVED

            # Get image paths
            task_img_dir = os.path.join(img_root, task)
            if not os.path.exists(task_img_dir):
                print(f"Warning: {task_img_dir} not found")
                continue

            traj_dirs = sorted(glob.glob(f"{task_img_dir}/*"))
            for ti, tdir in enumerate(traj_dirs):
                img_paths = sorted(glob.glob(f"{tdir}/*.jpg"))
                if img_paths and ti < len(actions):
                    # Use the first image from each trajectory
                    ip = img_paths[0]
                    self.items.append(
                        (ip,
                         prompt_dict[task],
                         torch.tensor(actions[ti], dtype=torch.float32))  # Now passing full trajectory
                    )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ipath, prompt, act = self.items[idx]
        image = Image.open(ipath).convert("RGB")
        # Resize and convert to tensor
        image = self.transform(image)
        return image, prompt, act

class ShortMWR3MDataset(Dataset):
    """Yield (image_tensor, prompt_str, action_tensor) from r3m processed pkl."""
    def __init__(self, root, task_list):
        self.items = []
        self.prompts = self._load_prompts(root)
        
        # Update path to point to the nested short-MetaWorld directory
        r3m_root = os.path.join(root, "short-MetaWorld", "r3m-processed", "r3m_MT10_20")
        
        for task in task_list:
            # Load actions from R3M processed data
            pkl_path = os.path.join(r3m_root, f"{task}.pkl")
            if not os.path.exists(pkl_path):
                print(f"Warning: {pkl_path} not found")
                continue
                
            data = pickle.load(open(pkl_path, "rb"))
            print(f"\nLoading {task}:")
            print(f"Features shape: {data['features'].shape if isinstance(data['features'], np.ndarray) else 'unknown'}")
            print(f"Actions shape: {data['actions'].shape if isinstance(data['actions'], np.ndarray) else 'unknown'}")
            
            # Get features and actions from R3M data
            features = data['features']    # (100, 20, 2048)
            actions = data['actions']      # (100, 20, 4)
            
            # Use first timestep of each trajectory
            features = features[:, 0]      # (100, 2048)
            actions = actions[:, 0]        # (100, 4)
            
            # Add to items list
            for i in range(len(features)):
                self.items.append((features[i], self.prompts[task], actions[i]))
                
        print(f"Loaded {len(self.items)} samples from {len(task_list)} tasks")

    def _load_prompts(self, root):
        from importlib.util import spec_from_file_location, module_from_spec
        spec = spec_from_file_location("tdesc", os.path.join(root, "task_description.py"))
        tdesc = module_from_spec(spec); spec.loader.exec_module(tdesc)
        return tdesc.PROMPT_DICT

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        features, prompt, action = self.items[idx]
        
        # Convert to tensors if needed
        if isinstance(features, np.ndarray):
            features = torch.tensor(features, dtype=torch.float32)
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float32)
            
        return features, prompt, action
