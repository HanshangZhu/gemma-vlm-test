import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig
from unified_tinyvla import UnifiedTinyVLAModel
from short_metaworld_ds import ShortMWDataset
from torch.utils.data import DataLoader
import numpy as np

def diagnose_loss_explosion():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    ds = ShortMWDataset("datasets/short-MetaWorld/short-MetaWorld/r3m-processed/r3m_MT10_20", ["pick-place-v2"])
    dl = DataLoader(ds, batch_size=2, shuffle=False)
    img, prompt, act = next(iter(dl))
    
    # Setup model (freshly initialized)
    model_path = "VLM_weights/Llava-Pythia-400M"
    config = AutoConfig.from_pretrained(model_path)
    config.action_head_type = 'droid_diffusion'
    config.action_dim = 4
    config.state_dim = 7
    config.chunk_size = 20
    
    model = UnifiedTinyVLAModel(model_path, mode="action")
    model = model.to(device)
    
    # Test diffusion head directly to see output magnitudes
    print("\n1. Testing diffusion head output scale...")
    diffusion_head = model.base_model.embed_out
    
    B, T, D = 2, 20, 4
    sample = torch.randn(B, T, D, device=device)  # Input actions
    timestep = torch.randint(0, 100, (B,), device=device)
    global_cond = torch.randn(B, 512, device=device)
    states = torch.randn(B, 7, device=device)
    
    with torch.no_grad():
        predicted_noise = diffusion_head(sample, timestep, global_cond=global_cond, states=states)
        
    print(f"Input sample stats:")
    print(f"  Mean: {sample.mean().item():.6f}, Std: {sample.std().item():.6f}")
    print(f"  Min: {sample.min().item():.6f}, Max: {sample.max().item():.6f}")
    
    print(f"\nPredicted noise stats:")
    print(f"  Mean: {predicted_noise.mean().item():.6f}, Std: {predicted_noise.std().item():.6f}")
    print(f"  Min: {predicted_noise.min().item():.6f}, Max: {predicted_noise.max().item():.6f}")
    
    # Create target noise (what it should predict)
    target_noise = torch.randn_like(sample)
    print(f"\nTarget noise stats:")
    print(f"  Mean: {target_noise.mean().item():.6f}, Std: {target_noise.std().item():.6f}")
    print(f"  Min: {target_noise.min().item():.6f}, Max: {target_noise.max().item():.6f}")
    
    # Compute MSE and understand why it's large
    mse = F.mse_loss(predicted_noise, target_noise)
    print(f"\nMSE between predicted and target noise: {mse.item():.6f}")
    
    # Break down the MSE
    diff = predicted_noise - target_noise
    print(f"\nDifference stats:")
    print(f"  Mean: {diff.mean().item():.6f}, Std: {diff.std().item():.6f}")
    print(f"  Min: {diff.min().item():.6f}, Max: {diff.max().item():.6f}")
    
    # Individual component analysis
    squared_diff = diff ** 2
    print(f"\nSquared difference stats:")
    print(f"  Mean: {squared_diff.mean().item():.6f} (this IS the MSE)")
    print(f"  Min: {squared_diff.min().item():.6f}, Max: {squared_diff.max().item():.6f}")
    
    # Test what normal MSE should be
    normal_noise1 = torch.randn_like(sample)
    normal_noise2 = torch.randn_like(sample)
    normal_mse = F.mse_loss(normal_noise1, normal_noise2)
    print(f"\nBaseline MSE (two random unit gaussians): {normal_mse.item():.6f}")
    
    # Test different scales
    print(f"\n2. MSE vs Output Scale Analysis:")
    for scale in [0.1, 1.0, 5.0, 10.0, 20.0, 30.0]:
        scaled_pred = target_noise * scale  # Simulate model outputting scaled values
        scaled_mse = F.mse_loss(scaled_pred, target_noise)
        print(f"  Scale {scale:4.1f}: MSE = {scaled_mse.item():8.2f}")
    
    # Real model test with actual forward pass
    print(f"\n3. Testing real model forward pass...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    img, act = img.to(device), act.to(device)
    # Normalize actions
    act = (act - act.mean()) / (act.std() + 1e-6)
    
    tok = tokenizer(list(prompt), return_tensors="pt", padding=True, truncation=True).to(device)
    dummy_states = torch.zeros(img.shape[0], 7, device=device)
    is_pad = torch.zeros(act.shape[:-1], dtype=torch.bool, device=device)
    
    # Forward pass to see actual loss
    outputs = model(
        input_ids=tok.input_ids,
        attention_mask=tok.attention_mask,
        images=img,
        states=dummy_states,
        actions=act,
        is_pad=is_pad
    )
    
    if 'loss' in outputs and outputs['loss'] is not None:
        actual_loss = outputs['loss']
        print(f"Actual model loss: {actual_loss.item():.6f}")
        
        if actual_loss.item() > 100:
            print(f"🚨 DIAGNOSIS: Model is outputting noise with std >> 1.0")
            print(f"   Likely causes:")
            print(f"   1. Poor weight initialization")
            print(f"   2. Learning rate too high") 
            print(f"   3. Missing proper normalization")
            print(f"   4. Gradient explosion")

def explain_mse_math():
    print(f"\n" + "="*60)
    print("MSE EXPLOSION EXPLAINED:")
    print("="*60)
    
    print(f"\nMSE Formula: MSE = E[(predicted - target)²]")
    print(f"\nIf model predicts noise with std=σ and target has std=1:")
    print(f"  MSE ≈ σ² + 1²  (when means are both ~0)")
    print(f"\nExamples:")
    print(f"  σ=1    → MSE ≈ 2     ✅ Normal")
    print(f"  σ=10   → MSE ≈ 101   ❌ Bad")  
    print(f"  σ=30   → MSE ≈ 901   ❌ Very bad")
    print(f"  σ=40   → MSE ≈ 1601  ❌ Your case!")
    
    print(f"\n🎯 SOLUTION: Fix model initialization/training to output std≈1")

if __name__ == "__main__":
    diagnose_loss_explosion()
    explain_mse_math() 