# Project Folder Organization

## Overview
This document describes the organized structure of the TinyVLA diffusion policy training project.

## Folder Structure

```
vla-vlm-test/
├── 📁 analysis/                    # Analysis and research documents
│   └── DIFFUSION_POLICY_LOSS_ANALYSIS.md
├── 📁 training_scripts/            # Training-related scripts
│   ├── train_tinyvla_policy.py     # Original (broken) training script
│   └── train_tinyvla_policy_FIXED.py # Fixed training script
├── 📁 debug_scripts/               # Debugging and diagnostic tools
│   ├── simple_mse_demo.py
│   ├── diagnose_loss_explosion.py
│   ├── check_loss_values.py
│   └── debug_diffusion_head.py
├── 📁 inference_scripts/           # Model inference and evaluation
│   ├── eval_metaworld_rgb.py
│   └── tinyvla_inference.py
├── 📁 docs/                        # Documentation
│   ├── KEY_DIFFERENCES.md          # Code comparison between broken/fixed
│   ├── WHAT_FIXED_THE_MODEL.md     # Technical analysis of fixes
│   └── FOLDER_ORGANIZATION.md      # This file
├── 📁 checkpoints/                 # Trained model checkpoints
│   ├── diff_head_FIXED_best.pth    # Best model from fixed training
│   └── TinyVLA-droid_diffusion_metaworld/
├── 📁 datasets/                    # Training datasets
├── 📁 VLM_weights/                 # Pre-trained model weights
├── 📁 test_imgs/                   # Test images for inference
├── 📁 TinyVLA/                     # Original TinyVLA codebase
├── 📁 metaworld_v1/                # MetaWorld environment
├── 📁 Intro/                       # Introduction materials
├── 📁 flagged/                     # Gradio flagged outputs
├── 📁 .vscode/                     # VS Code settings
├── 📁 __pycache__/                 # Python cache
├── 📁 .git/                        # Git repository
├── 📄 unified_tinyvla.py           # Main model wrapper
├── 📄 short_metaworld_ds.py        # Dataset loader
├── 📄 README.md                    # Main project documentation
├── 📄 requirements.txt             # Python dependencies
├── 📄 model_card.md                # Model information
├── 📄 .gitignore                   # Git ignore rules
├── 📄 MUJOCO_LOG.TXT              # MuJoCo logs
├── 📄 test_rendering.py            # Rendering tests
├── 📄 upload_model.py              # Model upload utilities
├── 📄 run_inference.sh             # Inference shell script
└── 📄 simple_metaworld_demo.py     # MetaWorld demo
```

## Key Files by Purpose

### 🚀 **Getting Started**
- `README.md` - Start here for project overview
- `requirements.txt` - Install dependencies
- `training_scripts/train_tinyvla_policy_FIXED.py` - Train the model

### 🔧 **Training & Development**
- `training_scripts/` - All training-related scripts
- `unified_tinyvla.py` - Main model architecture
- `short_metaworld_ds.py` - Dataset loading logic

### 🐛 **Debugging & Analysis**
- `debug_scripts/` - Tools for diagnosing training issues
- `analysis/DIFFUSION_POLICY_LOSS_ANALYSIS.md` - Loss interpretation guide
- `docs/WHAT_FIXED_THE_MODEL.md` - Technical deep-dive

### 🎯 **Inference & Evaluation**
- `inference_scripts/` - Model testing and evaluation
- `checkpoints/` - Trained model weights
- `test_imgs/` - Sample images for testing

### 📚 **Documentation**
- `docs/` - All documentation files
- `analysis/` - Research and analysis documents

## Usage Patterns

### For New Users
1. Read `README.md`
2. Install from `requirements.txt`
3. Run `training_scripts/train_tinyvla_policy_FIXED.py`
4. Check `analysis/DIFFUSION_POLICY_LOSS_ANALYSIS.md` for loss interpretation

### For Debugging Training Issues
1. Use scripts in `debug_scripts/`
2. Reference `docs/WHAT_FIXED_THE_MODEL.md`
3. Compare with `docs/KEY_DIFFERENCES.md`

### For Model Inference
1. Use scripts in `inference_scripts/`
2. Load checkpoints from `checkpoints/`
3. Test with images from `test_imgs/`

## File Naming Conventions

- **Scripts**: `verb_noun_descriptor.py` (e.g., `train_tinyvla_policy.py`)
- **Documentation**: `TOPIC_NAME.md` (e.g., `LOSS_ANALYSIS.md`)
- **Checkpoints**: `model_version_epoch.pth` (e.g., `diff_head_FIXED_best.pth`)
- **Configs**: `config_purpose.yaml` (if any)

## Maintenance Notes

- Keep `analysis/` updated with new findings
- Archive old experiments in dated subfolders
- Update `README.md` when adding major features
- Clean `__pycache__/` periodically
- Backup `checkpoints/` before major changes 