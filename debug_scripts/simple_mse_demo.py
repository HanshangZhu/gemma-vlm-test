import torch
import torch.nn.functional as F
import numpy as np

def demonstrate_mse_explosion():
    print("=" * 60)
    print("HOW MSE CAN REACH >1000: MATHEMATICAL PROOF")
    print("=" * 60)
    
    # Create sample data
    batch_size, timesteps, action_dim = 4, 20, 4
    
    print(f"\n1. BASELINE: Normal MSE between two unit Gaussians")
    target_noise = torch.randn(batch_size, timesteps, action_dim)  # N(0,1)
    random_pred = torch.randn(batch_size, timesteps, action_dim)   # N(0,1)
    
    baseline_mse = F.mse_loss(random_pred, target_noise)
    print(f"   Target stats: mean={target_noise.mean():.3f}, std={target_noise.std():.3f}")
    print(f"   Pred stats:   mean={random_pred.mean():.3f}, std={random_pred.std():.3f}")
    print(f"   Baseline MSE: {baseline_mse.item():.3f} ✅ Normal")
    
    print(f"\n2. PROBLEM: Model outputs large-scale noise")
    scales = [1, 5, 10, 20, 30, 40]
    
    for scale in scales:
        # Simulate model outputting scaled noise
        large_pred = torch.randn(batch_size, timesteps, action_dim) * scale
        mse = F.mse_loss(large_pred, target_noise)
        
        status = "✅" if mse < 10 else "❌" if mse < 100 else "🚨"
        print(f"   Scale {scale:2d}: pred_std={large_pred.std():.1f} → MSE={mse.item():7.1f} {status}")
    
    print(f"\n3. THE MATH:")
    print(f"   MSE = E[(predicted - target)²]")
    print(f"   If predicted ~ N(0, σ²) and target ~ N(0, 1):")
    print(f"   MSE ≈ σ² + 1²  (approximately, when independent)")
    print(f"")
    print(f"   Examples:")
    for sigma in [1, 10, 20, 30, 40]:
        approx_mse = sigma**2 + 1
        print(f"   σ={sigma:2d} → MSE ≈ {approx_mse:4d}")
    
    print(f"\n4. WHY YOUR MODEL PRODUCES SCALE ~40:")
    print(f"   🔧 Poor weight initialization → Large output values")
    print(f"   🔧 No proper normalization → Unbounded outputs") 
    print(f"   🔧 Gradient explosion → Weights grow too large")
    print(f"   🔧 Wrong learning rate → Can't converge to proper scale")
    
    print(f"\n5. DEMONSTRATION WITH ACTUAL NUMBERS:")
    
    # Show the exact calculation
    target = torch.tensor([0.5, -0.3, 1.2, -0.8])  # Some target noise
    bad_pred = torch.tensor([20.1, -15.7, 35.2, -25.3])  # Bad model output
    
    diff = bad_pred - target
    squared_diff = diff ** 2
    mse_manual = squared_diff.mean()
    mse_builtin = F.mse_loss(bad_pred, target)
    
    print(f"   Target:      {target.tolist()}")
    print(f"   Bad pred:    {bad_pred.tolist()}")
    print(f"   Difference:  {diff.tolist()}")
    print(f"   Squared:     {squared_diff.tolist()}")
    print(f"   MSE manual:  {mse_manual.item():.1f}")
    print(f"   MSE builtin: {mse_builtin.item():.1f}")
    
    print(f"\n6. CONCLUSION:")
    print(f"   ✅ MSE > 1000 is MATHEMATICALLY POSSIBLE")
    print(f"   ❌ But indicates SERIOUS training problems")
    print(f"   🎯 Fix: Proper weight init + normalization + learning rate")
    
    print(f"\n" + "=" * 60)

if __name__ == "__main__":
    demonstrate_mse_explosion() 