import os
import sys

# --- NEW: Add project root to PYTHONPATH ---
# This ensures the unified_tinyvla module can be found
# '..' is the parent directory of the training_scripts directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- END NEW ---

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import argparse, time, os, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig
import gc
import math

from unified_tinyvla import UnifiedTinyVLAModel
from short_metaworld_ds import ShortMWDataset

def proper_weight_init(module):
    """Proper weight initialization to prevent scale explosion"""
    if isinstance(module, torch.nn.Linear):
        # Xavier initialization for linear layers
        torch.nn.init.xavier_uniform_(module.weight, gain=0.5)  # Smaller gain to prevent explosion
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, torch.nn.Conv1d):
        # Kaiming initialization for conv layers
        torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu', a=0.1)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, (torch.nn.GroupNorm, torch.nn.LayerNorm, torch.nn.BatchNorm1d)):
        # Proper normalization layer initialization
        if hasattr(module, 'weight') and module.weight is not None:
            torch.nn.init.ones_(module.weight)
        if hasattr(module, 'bias') and module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, torch.nn.Embedding):
        # Smaller embedding initialization
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

def compute_loss(outputs, act, batch_idx, debug=False):
    """Compute loss without clipping - let's see the real values"""
    if not isinstance(outputs, dict):
        if debug: print(f"\nWarning: Unexpected output type: {type(outputs)}")
        return None

    if 'loss' in outputs and outputs['loss'] is not None:
        loss = outputs['loss']
        if debug: 
            print(f"Raw diffusion loss: {loss.item():.4f}")
            if torch.isnan(loss) or torch.isinf(loss):
                print("⚠️  NaN/Inf loss detected!")
                return None
        return loss

    if 'actions' in outputs and outputs['actions'] is not None:
        pred_actions = outputs['actions']
        if debug:
            print(f"Pred stats: mean={pred_actions.mean():.4f}, std={pred_actions.std():.4f}")
            print(f"Target stats: mean={act.mean():.4f}, std={act.std():.4f}")
        return F.mse_loss(pred_actions, act)

    if debug:
        print(f"No valid outputs for loss computation")
        print(f"Available keys: {outputs.keys()}")
    return None

def main(cfg):
    debug = getattr(cfg, "debug", False)

    torch.cuda.empty_cache()
    gc.collect()
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if debug:
        print(f"🚀 FIXED TRAINING SCRIPT")
        print(f"Using device: {device}")
        print(f"CUDA memory: {torch.cuda.memory_allocated()/1024**2:.2f} MB")

    # Load dataset
    tasks_train = cfg.tasks.split(",")
    ds = ShortMWDataset(cfg.data_root, tasks_train)
    dl = DataLoader(ds, batch_size=cfg.bs, shuffle=True, num_workers=2, pin_memory=True)
    
    if debug:
        print(f"\n📊 Dataset: {len(ds)} samples, {len(dl)} batches/epoch")
        print(f"Tasks: {', '.join(tasks_train)}")

    # Compute action normalization stats
    if debug: print("\n📈 Computing action statistics...")
    all_actions = []
    for i, (_, _, a) in enumerate(dl):
        all_actions.append(a)
        if i >= 10:  # Sample first 10 batches for stats
            break
    
    all_actions = torch.cat(all_actions, dim=0)
    action_mean = all_actions.mean(dim=0, keepdim=True)
    action_std = all_actions.std(dim=0, keepdim=True) + 1e-6
    
    if debug:
        print(f"Action normalization: mean={action_mean.mean():.4f}, std={action_std.mean():.4f}")

    # Save normalization stats
    os.makedirs(cfg.model_path, exist_ok=True)
    with open(os.path.join(cfg.model_path, "norm_stats.pkl"), "wb") as f:
        import pickle
        pickle.dump({"action_mean": action_mean.cpu().numpy(), "action_std": action_std.cpu().numpy()}, f)

    # Initialize model with proper config
    if debug: print("\n🏗️  Initializing model...")
    
    config = AutoConfig.from_pretrained(cfg.model_path)
    config.action_head_type = 'droid_diffusion'
    config.action_dim = 4
    config.state_dim = 7
    config.chunk_size = 20
    
    model = UnifiedTinyVLAModel(cfg.model_path, mode="action")
    
    # CRITICAL: Apply proper weight initialization to diffusion head
    if debug: print("🔧 Applying proper weight initialization...")
    diffusion_head = model.base_model.embed_out
    diffusion_head.apply(proper_weight_init)
    
    # Test the initialization worked
    if debug:
        print("🧪 Testing diffusion head after proper initialization...")
        with torch.no_grad():
            B, T, D = 2, 20, 4
            test_sample = torch.randn(B, T, D)
            test_timestep = torch.randint(0, 100, (B,))
            test_global_cond = torch.randn(B, 512)
            test_states = torch.randn(B, 7)
            
            test_output = diffusion_head(test_sample, test_timestep, 
                                       global_cond=test_global_cond, states=test_states)
            print(f"  Output stats: mean={test_output.mean():.4f}, std={test_output.std():.4f}")
            print(f"  Output range: [{test_output.min():.4f}, {test_output.max():.4f}]")
            
            if test_output.std() > 5.0:
                print("⚠️  Still large outputs after init - this might be a problem")
            else:
                print("✅ Output scale looks reasonable!")

    model = model.to(device)
    model.train()

    # Enable gradient checkpointing
    if hasattr(model.base_model, 'gradient_checkpointing_enable'):
        model.base_model.gradient_checkpointing_enable()
        if debug: print("✅ Gradient checkpointing enabled")

    # Setup training parameters
    trainable_params = list(diffusion_head.parameters())
    for p in trainable_params:
        p.requires_grad = True

    if debug:
        total_params = sum(p.numel() for p in trainable_params)
        print(f"🎯 Trainable parameters: {total_params:,}")

    # BETTER OPTIMIZER SETTINGS
    optimizer = torch.optim.AdamW(
        trainable_params, 
        lr=cfg.lr,
        weight_decay=0.01,
        betas=(0.9, 0.95),  # Better betas for stability
        eps=1e-8
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.1
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)

    if debug: print(f"\n🚀 Starting training with lr={cfg.lr}")
    
    start_time = time.time()
    best_loss = float('inf')
    patience = 0
    max_patience = 10

    for epoch in range(cfg.epochs):
        epoch_losses = []
        model.train()

        for batch_idx, (img, prompt, act) in enumerate(dl):
            try:
                img, act = img.to(device), act.to(device)
                
                # Normalize actions
                act_norm = (act - action_mean.to(device)) / action_std.to(device)
                
                # Tokenize prompts
                tok = tokenizer(list(prompt), return_tensors="pt", padding=True, truncation=True).to(device)
                dummy_states = torch.zeros(img.shape[0], 7, device=device)
                is_pad = torch.zeros(act_norm.shape[:-1], dtype=torch.bool, device=device)

                # Forward pass
                outputs = model(
                    input_ids=tok.input_ids,
                    attention_mask=tok.attention_mask,
                    images=img,
                    states=dummy_states,
                    actions=act_norm,
                    is_pad=is_pad
                )

                # Compute loss (NO CLIPPING!)
                loss = compute_loss(outputs, act_norm, batch_idx, debug and batch_idx % 20 == 0)
                
                if loss is None or torch.isnan(loss) or torch.isinf(loss):
                    if debug: print(f"⚠️  Skipping batch {batch_idx} - invalid loss")
                    continue

                # Backward pass with gradient clipping
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping (but not loss clipping!)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                
                optimizer.step()
                
                epoch_losses.append(loss.item())

                if debug and batch_idx % 10 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"Batch {batch_idx:3d}/{len(dl)} | Loss: {loss.item():8.4f} | "
                          f"Grad: {grad_norm:.4f} | LR: {current_lr:.2e} | "
                          f"Mem: {torch.cuda.memory_allocated()/1024**2:.0f}MB")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    if debug: print(f"💥 OOM at batch {batch_idx}, clearing cache")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

        # End of epoch
        if len(epoch_losses) == 0:
            print(f"❌ Epoch {epoch:02d}: No valid batches!")
            continue
            
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        scheduler.step()
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch:02d} | Loss: {avg_loss:8.4f} | Time: {elapsed:.1f}s | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # --- NEW: PERIODIC CHECKPOINTING ---
        if (epoch + 1) % 10 == 0:
            checkpoint_dir = os.path.join(cfg.checkpoint_dir, "TinyVLA-droid_diffusion_metaworld")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"diff_head_FIXED_epoch_{epoch+1}.pth")
            torch.save(model.base_model.embed_out.state_dict(), checkpoint_path)
            print(f"✅ Checkpoint saved: {checkpoint_path}")
        # --- END NEW ---

        # Early stopping logic
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                if debug:
                    print(f"⏹️  Early stopping at epoch {epoch}")
                break

    print(f"\n🎉 Training completed!")
    print(f"⏱️  Total time: {time.time() - start_time:.1f}s")
    print(f"🏆 Best loss: {best_loss:.4f}")

    # Final save
    final_checkpoint_dir = os.path.join(cfg.checkpoint_dir, "TinyVLA-droid_diffusion_metaworld")
    os.makedirs(final_checkpoint_dir, exist_ok=True)
    final_checkpoint_path = os.path.join(final_checkpoint_dir, f"diff_head_FIXED_epoch_{cfg.epochs}_final.pth")
    torch.save(model.base_model.embed_out.state_dict(), final_checkpoint_path)
    print(f"✅ Final model saved: {final_checkpoint_path}")

    if best_loss > 10:
        print(f"⚠️  Loss still high - may need more epochs or different hyperparameters")
    elif best_loss < 2:
        print(f"✅ Excellent! Loss in normal range for diffusion models")
    else:
        print(f"👍 Good progress! Loss is reasonable")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--bs", default=8, type=int)
    parser.add_argument("--data_root", default="datasets/short-MetaWorld/short-MetaWorld/r3m-processed/r3m_MT10_20")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--model_path", default="VLM_weights/Llava-Pythia-400M")
    parser.add_argument("--tasks", default="pick-place-v2,peg-insert-side-v2,drawer-open-v2,drawer-close-v2,door-open-v2,button-press-topdown-v2")
    parser.add_argument("--debug", action='store_true')
    return parser.parse_args()

if __name__ == "__main__":
    cfg = get_args()
    main(cfg) 