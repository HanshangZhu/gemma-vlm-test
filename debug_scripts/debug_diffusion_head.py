import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig
from unified_tinyvla import UnifiedTinyVLAModel
from short_metaworld_ds import ShortMWDataset
from torch.utils.data import DataLoader

def debug_diffusion_head():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("\n1. Loading data...")
    ds = ShortMWDataset("datasets/short-MetaWorld/short-MetaWorld/r3m-processed/r3m_MT10_20", ["pick-place-v2"])
    dl = DataLoader(ds, batch_size=2, shuffle=False)
    img, prompt, act = next(iter(dl))
    print(f"Data shapes: img={img.shape}, act={act.shape}")
    print(f"Action stats: mean={act.mean():.4f}, std={act.std():.4f}")
    
    # Setup model
    print("\n2. Setting up model...")
    model_path = "VLM_weights/Llava-Pythia-400M"
    
    # Load config and modify it
    config = AutoConfig.from_pretrained(model_path)
    config.action_head_type = 'droid_diffusion'
    config.action_dim = 4
    config.state_dim = 7
    config.chunk_size = 20
    config.save_pretrained(model_path)
    
    model = UnifiedTinyVLAModel(model_path, mode="action")
    model = model.to(device)
    model.train()
    
    # Check model setup
    print(f"Model head_type: {model.base_model.head_type}")
    print(f"Model embed_out type: {type(model.base_model.embed_out)}")
    
    # Check which parameters have gradients
    print("\n3. Checking gradient flow...")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"TRAINABLE: {name} - {param.shape}")
        else:
            print(f"FROZEN: {name} - {param.shape}")
    
    # Test forward pass
    print("\n4. Testing forward pass...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    with torch.no_grad():
        img, act = img.to(device), act.to(device)
        tok = tokenizer(list(prompt), return_tensors="pt", padding=True, truncation=True).to(device)
        dummy_states = torch.zeros(img.shape[0], 7, device=device)
        is_pad = torch.zeros(act.shape[:-1], dtype=torch.bool, device=device)
        
        print(f"Input shapes:")
        print(f"  images: {img.shape}")
        print(f"  input_ids: {tok.input_ids.shape}")
        print(f"  states: {dummy_states.shape}")
        print(f"  actions: {act.shape}")
        print(f"  is_pad: {is_pad.shape}")
    
    # Enable gradients for forward pass
    model.train()
    img.requires_grad_(True)
    
    try:
        outputs = model(
            input_ids=tok.input_ids,
            attention_mask=tok.attention_mask,
            images=img,
            states=dummy_states,
            actions=act,
            is_pad=is_pad
        )
        
        print(f"\n5. Forward pass outputs:")
        for k, v in outputs.items():
            if torch.is_tensor(v):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                print(f"       finite={torch.isfinite(v).all().item()}")
                print(f"       min={v.min().item():.6f}, max={v.max().item():.6f}")
                print(f"       requires_grad={v.requires_grad}")
            else:
                print(f"  {k}: {type(v)}")
        
        # Test loss computation
        if 'loss' in outputs and outputs['loss'] is not None:
            loss = outputs['loss']
            print(f"\n6. Loss analysis:")
            print(f"  Loss: {loss.item():.6f}")
            print(f"  Loss requires_grad: {loss.requires_grad}")
            print(f"  Loss is finite: {torch.isfinite(loss).item()}")
            
            if loss.requires_grad:
                print("  Attempting backward pass...")
                try:
                    loss.backward()
                    print("  ✓ Backward pass successful")
                    
                    # Check gradients
                    grad_count = 0
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad_count += 1
                            if grad_count <= 3:  # Show first 3
                                print(f"    {name}: grad_norm={param.grad.norm().item():.6f}")
                    print(f"  Total parameters with gradients: {grad_count}")
                    
                except Exception as e:
                    print(f"  ✗ Backward pass failed: {e}")
            else:
                print("  ✗ Loss doesn't require gradients")
        
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

    # Test diffusion head directly
    print("\n7. Testing diffusion head directly...")
    try:
        diffusion_head = model.base_model.embed_out
        
        # Create dummy inputs
        B, T, D = 2, 20, 4
        sample = torch.randn(B, T, D, device=device)
        timestep = torch.randint(0, 100, (B,), device=device)
        global_cond = torch.randn(B, 512, device=device)  # Hidden state size
        states = torch.randn(B, 7, device=device)
        
        print(f"Direct diffusion head test:")
        print(f"  sample: {sample.shape}")
        print(f"  timestep: {timestep.shape}")
        print(f"  global_cond: {global_cond.shape}")
        print(f"  states: {states.shape}")
        
        output = diffusion_head(sample, timestep, global_cond=global_cond, states=states)
        print(f"  output: {output.shape}")
        print(f"  output finite: {torch.isfinite(output).all().item()}")
        print(f"  output stats: min={output.min().item():.6f}, max={output.max().item():.6f}")
        
    except Exception as e:
        print(f"Direct diffusion head test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_diffusion_head()