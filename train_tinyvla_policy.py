import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import argparse, time, os, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import gc

from unified_tinyvla import UnifiedTinyVLAModel
from short_metaworld_ds import ShortMWDataset

def compute_loss(outputs, act, batch_idx, debug=False):
    if not isinstance(outputs, dict):
        if debug: print(f"\nWarning: Unexpected output type: {type(outputs)}")
        return None

    if 'loss' in outputs and outputs['loss'] is not None:
        if debug: print(f"Using diffusion head loss: {outputs['loss'].item():.4f}")
        return outputs['loss']

    if 'actions' in outputs and outputs['actions'] is not None:
        pred_actions = outputs['actions']
        if debug:
            print(f"[DEBUG] Pred mean: {pred_actions.mean():.4f}, std: {pred_actions.std():.4f}")
            print(f"\nComputing MSE loss from predicted actions:")
            print(f"Pred shape: {pred_actions.shape}, Target shape: {act.shape}")
            print(f"Pred stats: mean={pred_actions.mean():.4f}, std={pred_actions.std():.4f}")
            print(f"Target stats: mean={act.mean():.4f}, std={act.std():.4f}")
        return F.mse_loss(pred_actions, act)

    if debug:
        print(f"\nWarning: No valid outputs for loss computation")
        print(f"Available keys: {outputs.keys()}")
    return None

def main(cfg):
    debug = getattr(cfg, "debug", False)

    torch.cuda.empty_cache()
    gc.collect()
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if debug:
        print(f"\nUsing device: {device}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        print(f"CUDA memory reserved: {torch.cuda.memory_reserved()/1024**2:.2f} MB")

    tasks_train = cfg.tasks.split(",")
    ds = ShortMWDataset(cfg.data_root, tasks_train)
    dl = DataLoader(ds, batch_size=cfg.bs, shuffle=True, num_workers=4, pin_memory=True)
    if debug:
        print(f"\nDataset Info:\nTotal samples: {len(ds)}\nBatch size: {cfg.bs}\nBatches/epoch: {len(dl)}\nTasks: {', '.join(tasks_train)}")

    img, prompt, act = ds[0]
    if debug:
        print(f"Sample data shapes:\nImage: {img.shape}\nPrompt: {prompt}\nAction: {act.shape}")

    if debug:
        print("\n2. Computing action statistics...")
    all_actions = torch.cat([a for _, _, a in dl], dim=0)
    action_mean = all_actions.mean(dim=0, keepdim=True)
    action_std = all_actions.std(dim=0, keepdim=True) + 1e-6
    if debug:
        print(f"Action stats: mean={action_mean.mean():.4f}, std={action_std.mean():.4f}")

    os.makedirs(cfg.model_path, exist_ok=True)
    with open(os.path.join(cfg.model_path, "norm_stats.pkl"), "wb") as f:
        import pickle
        pickle.dump({"action_mean": action_mean.cpu().numpy(), "action_std": action_std.cpu().numpy()}, f)

    if debug:
        print("\n3. Initializing model...")
    model = UnifiedTinyVLAModel(cfg.model_path, mode="action")
    if hasattr(model.base_model, 'gradient_checkpointing_enable'):
        model.base_model.gradient_checkpointing_enable()
        if debug: print("Enabled gradient checkpointing")

    model = model.to(device)
    model.train()

    diffusion_head = model.base_model.embed_out
    for p in diffusion_head.parameters():
        p.requires_grad = True

    if debug:
        print(f"Model Info: Trainable params: {sum(p.numel() for p in diffusion_head.parameters() if p.requires_grad):,}")
        print(f"CUDA memory after model load: {torch.cuda.memory_allocated()/1024**2:.2f} MB")

    opt = torch.optim.AdamW(diffusion_head.parameters(), lr=cfg.lr, weight_decay=0.01)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)

    if debug:
        print("\n4. Starting training...")
    start = time.time()
    best_loss = float('inf')
    patience = 0
    max_patience = 5

    for epoch in range(cfg.epochs):
        tot_loss, num_batches = 0, 0
        model.train()

        for batch_idx, (img, prompt, act) in enumerate(dl):
            try:
                img, act = img.to(device), act.to(device)
                act = (act - action_mean.to(device)) / action_std.to(device)
                tok = tokenizer(list(prompt), return_tensors="pt", padding=True, truncation=True).to(device)
                dummy_states = torch.zeros(img.shape[0], 7, device=device)
                outputs = model(
                    input_ids=tok.input_ids,
                    attention_mask=tok.attention_mask,
                    images=img,
                    states=dummy_states,
                    actions=act
                )

                if debug:
                    for k, v in outputs.items():
                        if torch.is_tensor(v):
                            print(k, v.min().item(), v.max().item(), torch.isfinite(v).all().item())

                pred = outputs.get("actions", None)
                if debug:
                    print(f"[DEBUG] Pred mean: {pred.mean():.4f}, std: {pred.std():.4f}" if pred is not None else "[DEBUG] No predicted actions")

                loss = compute_loss(outputs, act, batch_idx, debug)
                if loss is None or torch.isnan(loss) or torch.isinf(loss):
                    if debug: print(f"Skipping batch {batch_idx}, invalid loss")
                    continue
                if loss.item() > 1000:
                    if debug: print(f"Large loss {loss.item():.4f}, scaling down")
                    loss = loss * 0.1

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(diffusion_head.parameters(), max_norm=1.0)
                opt.step()

                tot_loss += loss.item()
                num_batches += 1

                if debug and batch_idx % 5 == 0:
                    print(f"Batch {batch_idx}/{len(dl)} | Loss: {loss.item():.4f} | Mem: {torch.cuda.memory_allocated()/1024**2:.2f} MB")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    if debug: print(f"OOM at batch {batch_idx}")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

        avg_loss = tot_loss / num_batches if num_batches else float('inf')
        print(f"Epoch {epoch:02d} | Avg Loss: {avg_loss:.4f} | Time: {time.time()-start:.1f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience = 0
            torch.save(diffusion_head.state_dict(), f"{cfg.out_dir}/diff_head_ft_best.pth")
            print(f"New best saved @ epoch {epoch:02d} | Loss: {best_loss:.4f}")
        else:
            patience += 1
            if patience >= max_patience:
                print(f"Early stopping triggered at epoch {epoch:02d}")
                break

        if (epoch + 1) % 10 == 0:
            torch.save(diffusion_head.state_dict(), f"{cfg.out_dir}/diff_head_ft_epoch_{epoch+1}.pth")

    torch.save(diffusion_head.state_dict(), f"{cfg.out_dir}/diff_head_ft_final.pth")
    print(f"Training done | Final Loss: {best_loss:.4f} | Model saved to {cfg.out_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="VLM_weights/Llava-Pythia-400M")
    p.add_argument("--data_root",  default="datasets/short-MetaWorld/short-MetaWorld/r3m-processed/r3m_MT10_20")
    p.add_argument("--tasks", default="pick-place-v2,peg-insert-side-v2,drawer-open-v2,drawer-close-v2,door-open-v2,button-press-topdown-v2")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--out_dir", default="checkpoints")
    p.add_argument("--debug", action="store_true")
    cfg = p.parse_args()
    main(cfg)
