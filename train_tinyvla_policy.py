import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse, time, os, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from unified_tinyvla import UnifiedTinyVLAModel
#from datasets.short-MetaWorld.short-MetaWo import ShortMWDataset
from short_metaworld_ds import ShortMWDataset

def main(cfg):
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ⚙️ dataset -------------------------------------------------------------
    tasks_train = cfg.tasks.split(",")
    ds = ShortMWDataset(cfg.data_root, tasks_train)
    dl = DataLoader(ds, batch_size=cfg.bs, shuffle=True,
                    num_workers=4, pin_memory=True)
    print("Dataset size:", len(ds))
    
    img, prompt, act = ds[0]
    print(f"Sample image: {img.shape}, prompt: {prompt}, action: {act.shape}")
    

    # ⚙️ model --------------------------------------------------------------
    model = UnifiedTinyVLAModel(cfg.model_path, mode="action").to(device)
    # Get the diffusion head from the base model for compilation and optimization
    diffusion_head = model.base_model.embed_out  # This is the action head in the base model
    diffusion_head = torch.compile(diffusion_head)      # speed-up
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    opt = torch.optim.AdamW(diffusion_head.parameters(), lr=cfg.lr)

    # 🏃 training -----------------------------------------------------------
    start = time.time()
    for epoch in range(cfg.epochs):
        tot_loss = 0
        for img, prompt, act in dl:
            img, act = img.to(device), act.to(device)
            tok = tokenizer(list(prompt), return_tensors="pt",
                            padding=True, truncation=True).to(device)

            # Create dummy states for the diffusion head - batch size x state_dim
            batch_size = img.shape[0]
            dummy_states = torch.zeros(batch_size, 7, device=device, dtype=torch.float32)

            outputs = model(
                input_ids=tok.input_ids,
                attention_mask=tok.attention_mask,
                images=img,
                states=dummy_states,  # Provide dummy states instead of None
                actions=act           # Pass target actions for training
            )
            
            print(f"DEBUG: outputs type: {type(outputs)}")
            print(f"DEBUG: outputs keys: {outputs.keys() if isinstance(outputs, dict) else 'not a dict'}")
            
            # During training, the diffusion head computes loss internally
            # Check if loss is available in outputs
            if isinstance(outputs, dict) and 'loss' in outputs and outputs['loss'] is not None:
                loss = outputs['loss']
            elif hasattr(outputs, 'loss') and outputs.loss is not None:
                loss = outputs.loss
            else:
                # Fallback: compute loss from predicted actions
                pred = outputs['actions'] if isinstance(outputs, dict) else outputs.actions
                if pred is not None:
                    loss = F.mse_loss(pred, act)
                else:
                    print("Warning: No loss or actions returned from model")
                    continue

            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item()

        print(f"epoch {epoch:02d} | loss {tot_loss/len(dl):.4f} "
              f"| elapsed {time.time()-start:.1f}s")
    os.makedirs(cfg.out_dir, exist_ok=True)
    torch.save(diffusion_head.state_dict(),
               f"{cfg.out_dir}/diff_head_ft.pth")
    print("✔ saved diffusion head to", cfg.out_dir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="VLM_weights/Llava-Pythia-400M")
    p.add_argument("--data_root",  default="./datasets/short-MetaWorld")
    p.add_argument("--tasks",      default="pick-place-v2,door-v2,drawer-open-v2")
    p.add_argument("--epochs",     type=int, default=10)
    p.add_argument("--bs",         type=int, default=16)   # Reduced from 32 to 16 to prevent OOM
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--out_dir",    default="checkpoints")
    cfg = p.parse_args()
    main(cfg)

    '''
    python train_tinyvla_policy.py \
  --data_root ./datasets/short-metaworld \
  --model_path VLM_weights/Llava-Pythia-400M \
  --tasks reach-v1,pick-place-v1,door-open-v1 \
  --epochs 10 \
  --bs 128 \
  --out_dir checkpoints

    '''
