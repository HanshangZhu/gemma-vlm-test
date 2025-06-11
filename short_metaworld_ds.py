import os, glob, pickle
from PIL import Image
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

class ShortMWDataset(Dataset):
    """Yield (image_tensor, prompt_str, action_tensor)."""
    def __init__(self, root, task_list, split='train'):
        self.root = root
        self.tf = T.Compose([T.Resize((224, 224)), T.ToTensor()])
        self.items = []

        # ↳  load prompt dict -------------------------------------------------
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "tdesc", os.path.join(root, "task_description.py"))
        tdesc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tdesc)
        prompt_dict = tdesc.PROMPT_DICT       # task → str

        img_root = os.path.join(root, "img_only")
        act_root = os.path.join(root, "unprocessed")

        for task in task_list:
            pkl = glob.glob(f"{act_root}/**/{task}.pkl", recursive=True)[0]
            actions = pickle.load(open(pkl, "rb"))           # (100,20,10)

            traj_dirs = sorted(glob.glob(f"{img_root}/{task}/*"))
            for ti, tdir in enumerate(traj_dirs):
                img_paths = sorted(glob.glob(f"{tdir}/*.jpg"))
                for si, ip in enumerate(img_paths):
                    self.items.append(
                        (ip,
                         prompt_dict[task],
                         torch.tensor(actions[ti, si], dtype=torch.float32))
                    )

    def __len__(self):  return len(self.items)

    def __getitem__(self, idx):
        ipath, prompt, act = self.items[idx]
        img = self.tf(Image.open(ipath).convert("RGB"))
        return img, prompt, act
