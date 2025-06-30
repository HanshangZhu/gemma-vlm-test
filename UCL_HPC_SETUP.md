# UCL HPC Setup Guide for TinyVLA Training

## 🚀 Quick Start (3 Steps)

### 1. Sync Your Code to UCL
```bash
# From your local machine
rsync -avz --exclude='.git' \
      /home/hz/vla-vlm-test/ \
      YOUR_UCL_USERNAME@login05.external.legion.ucl.ac.uk:~/Scratch/tinyvla/
```

### 2. Setup Environment on UCL
```bash
# SSH into UCL HPC
ssh YOUR_UCL_USERNAME@login05.external.legion.ucl.ac.uk

# Navigate to project
cd ~/Scratch/tinyvla

# Create conda environment from requirements
module load python/miniconda3/4.10.3
conda create --name tinyvla --file requirements.txt -y

# Test environment
conda activate tinyvla
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

### 3. Submit Training Job
```bash
# Make script executable
chmod +x tinyvla_train_ucl.sh

# Edit the script to set your email
sed -i 's/your.email@ucl.ac.uk/YOUR_ACTUAL_EMAIL@ucl.ac.uk/' tinyvla_train_ucl.sh

# Submit job
qsub tinyvla_train_ucl.sh

# Monitor job
qstat
```

## 📋 Pre-Submission Checklist

- [ ] **Code synced**: Project files copied to `~/Scratch/tinyvla/`
- [ ] **Environment ready**: `conda env list` shows `tinyvla`
- [ ] **Config exists**: Check `configs/train_hpc_700m_mt50.yaml`
- [ ] **Email set**: Updated notification email in script
- [ ] **Paths correct**: Verify `$HOME/Scratch/tinyvla` exists

## 🎛️ UCL-Specific Settings

### Resource Requests (adjust based on your needs)
```bash
#$ -l h_rt=48:0:0      # 48 hours max
#$ -l mem=128G         # 128GB RAM
#$ -l gpu=2            # 2 GPUs
#$ -l tmpfs=50G        # 50GB temp storage
```

### Common UCL Module Commands
```bash
# See available modules
module avail cuda
module avail python

# Load specific versions (adjust as needed)
module load cuda/11.8/gnu/9.2.0
module load python/miniconda3/4.10.3
```

## 📊 Monitoring Your Job

```bash
# Check job status
qstat -u $USER

# View job details
qstat -j JOB_ID

# Check GPU usage (while job running)
qrsh -l gpu=1
nvidia-smi

# View logs
tail -f tinyvla_*.log
```

## 🔧 Troubleshooting

### Common Issues:

1. **Module not found**: Check `module avail` and adjust versions
2. **GPU allocation failed**: Reduce `gpu=2` to `gpu=1` or check queue
3. **Memory error**: Reduce `mem=128G` to `mem=64G`
4. **Environment issues**: Recreate conda env or load different python module

### Debug Commands:
```bash
# Test GPU access
qrsh -l gpu=1
nvidia-smi

# Test conda environment
conda activate tinyvla
python -c "import torch; print(torch.cuda.is_available())"

# Check disk space
df -h ~/Scratch
```

## 📁 UCL Directory Structure
```
~/Scratch/tinyvla/                 # Main project
├── tinyvla_train_ucl.sh          # Job script
├── train_lora.py                 # Training script
├── configs/                      # Training configs
├── requirements.txt              # Environment spec
├── ucl_results/                  # Results (created by job)
└── ucl_checkpoints/              # Model checkpoints
```

## 📝 Expected Job Flow

1. **Setup** (5 min): Load modules, activate conda, stage data
2. **Training** (24-48h): Run training with automatic checkpointing  
3. **Cleanup** (2 min): Sync results back to Scratch, generate summary
4. **Notification**: Email sent on completion/failure

That's it! Your TinyVLA training should now run smoothly on UCL HPC. 🎉 