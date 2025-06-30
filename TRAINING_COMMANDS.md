# 🚀 TinyVLA Training & Testing Commands

## 🔥 TRAINING COMMANDS

### Quick Test Training (50 steps)
```bash
python train_lora.py --config configs/train_bs1_final.yaml
```

### Full Training (longer)
```bash
# Edit max_steps in configs/train_bs1_final.yaml to desired value (e.g., 2000)
python train_lora.py --config configs/train_bs1_final.yaml
```

### Alternative Training Configs
```bash
# Standard image size (224x224) - less memory optimization
python train_lora.py --config configs/train_full_bs1_standard.yaml

# Ultra memory optimized (168x168 images)
python train_lora.py --config configs/train_full_bs1_ultra_fixed.yaml
```

### Training with Multiple Tasks
```bash
# Edit train_tasks in config: "pick-place-v2,door-open-v2,drawer-close-v2"
python train_lora.py --config configs/train_bs1_final.yaml
```

## 🔍 TESTING/VALIDATION COMMANDS

### Check Diffusion Head Weights
```bash
python check_weights.py
```

### Test Model Loading (Quick)
```bash
python test_model_loading.py
```

### Test Model Inference (No NaN/Inf)
```bash
python simple_debug.py
```

### Load and Test Trained Model (Full Demo)
```bash
python minimal_pickplace_demo.py
```

## 📁 KEY DIRECTORIES

- **Base Model**: `VLA_weights/Llava-Pythia-400M/`
- **Training Output**: `VLA_weights/full_training_adapter/` (✅ HAS CHECKPOINTS)
- **Diffusion Weights**: `VLA/diff_head/` (🔥 CRITICAL - loads these!)
- **Available Checkpoints**: 
  - `step_10000` (latest)
  - `step_9500`, `step_9000`, etc.
  - `step_500` (early training)

## 🎯 IMPORTANT NOTES

1. **Diffusion Head Weights**: Always in `VLA/diff_head/diffusion_head_latest.bin`
2. **LoRA Weights**: In `VLA_weights/full_training_adapter/step_XXXX/`
3. **Memory**: Uses only ~17MB GPU memory (vs 3.66GB before fix)
4. **Batch Size**: Keep at 1 for stability
5. **Save Frequency**: Every 50 steps to prevent memory buildup

## 🚨 CRITICAL FIXES APPLIED

1. ✅ **Diffusion Head Training**: Now properly saves/loads 72.8M parameters
2. ✅ **Memory Management**: Reduced from 3.66GB to 17MB GPU usage  
3. ✅ **Model Loading**: Fixed to load both LoRA + diffusion weights
4. ✅ **NaN Prevention**: Multiple stability fixes applied 