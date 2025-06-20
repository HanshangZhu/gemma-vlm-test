import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig
from unified_tinyvla import UnifiedTinyVLAModel
from short_metaworld_ds import ShortMWDataset
from torch.utils.data import DataLoader
import numpy as np

def check_loss_magnitudes():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("\n1. Loading data...")
    ds = ShortMWDataset("datasets/short-MetaWorld/short-MetaWorld/r3m-processed/r3m_MT10_20", ["pick-place-v2"])
    dl = DataLoader(ds, batch_size=4, shuffle=False)
    img, prompt, act = next(iter(dl))
    print(f"Data shapes: img={img.shape}, act={act.shape}")
    print(f"Raw action stats: mean={act.mean():.4f}, std={act.std():.4f}, min={act.min():.4f}, max={act.max():.4f}")
    
    # Normalize actions (like in training)
    action_mean = act.mean(dim=(0, 1), keepdim=True) 
    action_std = act.std(dim=(0, 1), keepdim=True) + 1e-6
    act_normalized = (act - action_mean) / action_std
    print(f"Normalized action stats: mean={act_normalized.mean():.4f}, std={act_normalized.std():.4f}")
    
    # Setup model
    print("\n2. Setting up model...")
    model_path = "VLM_weights/Llava-Pythia-400M"
    
    config = AutoConfig.from_pretrained(model_path)
    config.action_head_type = 'droid_diffusion'
    config.action_dim = 4
    config.state_dim = 7
    config.chunk_size = 20
    
    model = UnifiedTinyVLAModel(model_path, mode="action")
    model = model.to(device)
    
    # Load trained weights
    checkpoint_path = "checkpoints/diff_head_ft_best.pth"
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.base_model.embed_out.load_state_dict(checkpoint)
        print(f"✓ Loaded trained diffusion head from {checkpoint_path}")
    except Exception as e:
        print(f"Could not load checkpoint: {e}")
        print("Using randomly initialized weights...")
    
    model.eval()
    
    # Test with real data
    print("\n3. Testing loss computation...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    with torch.no_grad():
        img, act = img.to(device), act_normalized.to(device)
        tok = tokenizer(list(prompt), return_tensors="pt", padding=True, truncation=True).to(device)
        dummy_states = torch.zeros(img.shape[0], 7, device=device)
        is_pad = torch.zeros(act.shape[:-1], dtype=torch.bool, device=device)
        
        outputs = model(
            input_ids=tok.input_ids,
            attention_mask=tok.attention_mask,
            images=img,
            states=dummy_states,
            actions=act,
            is_pad=is_pad
        )
        
        print(f"Model outputs:")
        for k, v in outputs.items():
            if torch.is_tensor(v):
                print(f"  {k}: shape={v.shape}, finite={torch.isfinite(v).all().item()}")
                if k == 'loss':
                    print(f"       loss value: {v.item():.6f}")
        
        # Analyze why diffusion loss is large
        print(f"\n4. Understanding diffusion loss magnitude:")
        
        # The diffusion loss is MSE between predicted noise and actual noise
        # Let's understand the scale
        if 'loss' in outputs and outputs['loss'] is not None:
            loss = outputs['loss']
            print(f"Current diffusion loss: {loss.item():.6f}")
        
        # Compare with simple MSE on actions
        if 'actions' in outputs and outputs['actions'] is not None:
            pred_actions = outputs['actions']
            simple_mse = F.mse_loss(pred_actions, act)
            print(f"Simple action MSE: {simple_mse.item():.6f}")
        
        # Test noise magnitudes
        print(f"\n5. Noise magnitude analysis:")
        noise_sample = torch.randn_like(act)
        noise_mse = F.mse_loss(noise_sample, torch.zeros_like(act))
        print(f"Random noise MSE vs zeros: {noise_mse.item():.6f}")
        
        # Different noise levels
        for scale in [0.1, 0.5, 1.0, 2.0]:
            scaled_noise = noise_sample * scale
            scaled_mse = F.mse_loss(scaled_noise, torch.zeros_like(act))
            print(f"Noise (scale={scale}) MSE: {scaled_mse.item():.6f}")

def analyze_diffusion_process():
    """Explain why diffusion losses are typically large"""
    print(f"\n" + "="*50)
    print("WHY DIFFUSION LOSSES ARE LARGE:")
    print("="*50)
    
    print("\n1. DIFFUSION LOSS IS NOISE PREDICTION ERROR:")
    print("   - Loss = MSE(predicted_noise, actual_noise)")
    print("   - Noise is sampled from N(0,1) - unit Gaussian")
    print("   - MSE of unit Gaussian vs zeros ≈ 1.0")
    
    print("\n2. NOISE SCHEDULING:")
    print("   - Training uses noise levels from 0 to T timesteps")
    print("   - At high noise levels, signal is completely corrupted")
    print("   - Model must predict pure noise → high loss")
    
    print("\n3. TYPICAL LOSS RANGES:")
    print("   - Untrained: 1.0-2.0 (random noise prediction)")
    print("   - Well-trained: 0.1-0.5 (good noise prediction)")
    print("   - Perfect: 0.0 (impossible in practice)")
    
    print("\n4. LOSS COMPARISON:")
    print("   - Action MSE loss: ~0.01-0.1 (actions are small)")
    print("   - Diffusion loss: ~0.1-1.0 (noise has unit variance)")
    print("   - 10x difference is normal!")
    
    print("\n5. WHAT MATTERS:")
    print("   - Relative improvement, not absolute values")
    print("   - Loss decreasing over training")
    print("   - Model generating coherent actions")

if __name__ == "__main__":
    check_loss_magnitudes()
    analyze_diffusion_process() 