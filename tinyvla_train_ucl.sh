#!/bin/bash -l
###############################################################################
# tinyvla_train_ucl.sh – TinyVLA 700M Training for UCL HPC (SGE/PBS) + DeepSpeed
###############################################################################

# SGE/PBS job directives (adjust based on UCL's specific system)
#$ -cwd                              # Execute in the directory qsub is run from
#$ -N tinyvla-700m-mt50-ds           # Job name (added -ds for DeepSpeed)
#$ -pe smp 16                        # 16 CPU cores (8 per GPU process)
#$ -l gpu=2                          # 2 GPUs (A100 queue preferred)
#$ -l h_rt=48:00:00                  # 48 h wall-clock
#$ -l mem=8G                         # per-core memory (8G × 16 cores ≈ 128G total)
#$ -l tmpfs=70G                      # Local NVMe scratch, 50G dataset + 20G for training checkpoints
# ---- Optional specific queue (uncomment if confirmed online) ----
# #$ -q Daenerys
#-----------------------------------------------------------------
#$ -j y                              # Merge stdout/stderr
#$ -o tinyvla_ds_$JOB_ID.log        # Consolidated log (added _ds)

# Email notifications (PLACEHOLDER - replace with your UCL email)
#$ -M zcabhz4@ucl.ac.uk
#$ -m bes

###############################################################################
# 1. ENVIRONMENT SETUP
###############################################################################
date                                       # Log start time
echo "🚀 TinyVLA Training Job Starting (DeepSpeed)"
echo "📍 Job ID: $JOB_ID"                  # SGE job ID
echo "🖥️  Hostname: $(hostname)"
echo "📁 Working directory: $(pwd)"

# UCL HPC module system
module purge                               # Clear existing modules
module load gcc-libs/11.2.0               # Newer GCC toolchain
module load cuda/12.0/gnu/11.2.0          # CUDA 12 toolkit for A100s
module load miniconda/23.3.1              # Up-to-date Miniconda module

# Display loaded modules
echo "📦 Loaded modules:"
module list

###############################################################################
# 2. CONDA ENVIRONMENT SETUP + DEEPSPEED
###############################################################################
echo "🐍 Setting up conda environment..."

# Source conda init script (fixed absolute path on Myriad)
source /shared/ucl/apps/miniconda/23.3.1/etc/profile.d/conda.sh

# Activate or create the tinyvla environment
if conda env list | grep -q "tinyvla"; then
    echo "✅ Activating existing tinyvla environment..."
    conda activate tinyvla
else
    echo "🔧 Creating tinyvla environment from requirements.txt..."
    conda create --name tinyvla python=3.10 -y
    conda activate tinyvla
    pip install -r requirements.txt
fi

# Install DeepSpeed if not already present
echo "🔧 Ensuring DeepSpeed is installed..."
pip install deepspeed==0.12.6 --no-deps || echo "DeepSpeed already installed"

# Verify environment
echo "✅ Python: $(which python)"
echo "✅ Python version: $(python --version)"
echo "✅ PyTorch: $(python -c 'import torch; print(f"v{torch.__version__}, CUDA: {torch.cuda.is_available()}")')"
echo "✅ DeepSpeed: $(python -c 'import deepspeed; print(f"v{deepspeed.__version__}")')"

###############################################################################
# 3. PROJECT SETUP AND DATA STAGING
###############################################################################
echo "📁 Setting up project workspace..."

# Use UCL's temporary storage for faster I/O
WORK_DIR="$TMPDIR/tinyvla_work"            # UCL provides $TMPDIR on compute nodes
PROJECT_SOURCE="$HOME/Scratch/tinyvla"     # Your project in Scratch space
mkdir -p "$WORK_DIR"

echo "🔄 Copying project to local storage for faster training..."
rsync -av --exclude='.git' --exclude='*.pyc' \
      "$PROJECT_SOURCE/" "$WORK_DIR/"

# Change to working directory
cd "$WORK_DIR" || { echo "❌ Failed to access work directory"; exit 1; }

echo "📂 Project contents:"
ls -la

###############################################################################
# 4. DEEPSPEED CONFIGURATION SETUP
###############################################################################
echo "⚙️ Creating DeepSpeed configuration..."

# Create DeepSpeed config for ZeRO Stage 2 (optimized for 2x40GB A100)
cat > ds_config_stage2.json << 'EOF'
{
  "train_batch_size": 96,
  "train_micro_batch_size_per_gpu": 24,
  "gradient_accumulation_steps": 2,

  "bf16": {
    "enabled": true,
    "auto_cast": false,
    "loss_scale": 0,
    "initial_scale_power": 16,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },

  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "reduce_bucket_size": 5e8,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "cpu_offload": false
  },

  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 3e-6,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },

  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 3e-6,
      "warmup_num_steps": 2000
    }
  },

  "gradient_clipping": 0.3,
  "steps_per_print": 100,
  "wall_clock_breakdown": false
}
EOF

echo "✅ DeepSpeed config created: ds_config_stage2.json"

###############################################################################
# 5. GPU AND PERFORMANCE SETUP
###############################################################################
echo "⚙️ Configuring GPU settings for DeepSpeed..."

# DeepSpeed will handle GPU assignment automatically
export CUDA_VISIBLE_DEVICES="0,1"

# Network settings for multi-GPU communication
export NCCL_SOCKET_IFNAME=eth0  # Adjust if UCL uses different interface
export NCCL_IB_DISABLE=1        # Disable InfiniBand if not available
export NCCL_DEBUG=INFO          # Enable debugging for first run

# Performance optimizations
export OMP_NUM_THREADS=8                   # OpenMP threads per process
export MKL_NUM_THREADS=8                   # Intel MKL threads

# Offline mode for tools that need internet
export WANDB_MODE=offline

# Create local cache directories
mkdir -p "$WORK_DIR/.cache/huggingface"
mkdir -p "$WORK_DIR/.cache/torch"
export HF_HOME="$WORK_DIR/.cache/huggingface"
export TORCH_HOME="$WORK_DIR/.cache/torch"

# Display GPU information
echo "🎮 GPU Information:"
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

###############################################################################
# 6. TRAINING CONFIGURATION
###############################################################################
echo "🎓 Preparing training configuration..."

# Choose config based on available resources
CONFIG_FILE="configs/train_hpc_700m_mt50.yaml"
echo "📝 Using HPC config for DeepSpeed multi-GPU training"

# Verify config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    echo "📋 Available configs:"
    ls -la configs/
    exit 1
fi

# Create timestamped output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/ucl_ds_run_${JOB_ID}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"
echo "📁 Output directory: $OUTPUT_DIR"

###############################################################################
# 7. LAUNCH DEEPSPEED TRAINING
###############################################################################
echo "🚀 Starting TinyVLA training with DeepSpeed..."
echo "📝 Config: $CONFIG_FILE"
echo "🔧 DeepSpeed config: ds_config_stage2.json"
echo "📊 Logging to: $OUTPUT_DIR/training.log"

# Launch with DeepSpeed
deepspeed --num_gpus=2 \
          --master_port=29500 \
          train_lora.py \
          --config "$CONFIG_FILE" \
          --deepspeed ds_config_stage2.json \
          --output_dir "$OUTPUT_DIR" \
          2>&1 | tee "$OUTPUT_DIR/training.log"

# Capture exit status
TRAIN_STATUS=$?

###############################################################################
# 8. POST-TRAINING SYNC AND CLEANUP
###############################################################################
echo "🧹 Post-training cleanup..."

if [ $TRAIN_STATUS -eq 0 ]; then
    echo "✅ Training completed successfully!"
    
    # Create results summary
    cat > "$OUTPUT_DIR/job_summary.txt" << EOF
UCL HPC DeepSpeed Training Job Summary
======================================
Job ID: $JOB_ID
Hostname: $(hostname)
Start Time: $(date)
Config Used: $CONFIG_FILE
DeepSpeed Config: ds_config_stage2.json
Output Directory: $OUTPUT_DIR
Training Status: SUCCESS
GPUs Used: 2x A100-40GB
DeepSpeed Stage: ZeRO-2
EOF

    # Sync results back to Scratch
    echo "💾 Syncing results to Scratch space..."
    rsync -av "$OUTPUT_DIR/" "$PROJECT_SOURCE/ucl_ds_results/"
    
    # Optional: Sync checkpoints
    if [ -d "checkpoints" ]; then
        rsync -av checkpoints/ "$PROJECT_SOURCE/ucl_ds_checkpoints/"
    fi
    
else
    echo "❌ Training failed with exit code: $TRAIN_STATUS"
    
    # Still save logs for debugging
    rsync -av "$OUTPUT_DIR/" "$PROJECT_SOURCE/ucl_ds_debug/"
fi

###############################################################################
# 9. RESOURCE USAGE SUMMARY
###############################################################################
echo "📈 Final resource usage:"
echo "GPU Memory:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

echo "Disk Usage:"
du -sh "$WORK_DIR"

# Simple wall-clock usage summary
echo "Wallclock used: $((SECONDS/3600))h $(((SECONDS%3600)/60))m"

echo "Job completed at: $(date)"
echo "Results saved to: $PROJECT_SOURCE/ucl_ds_results/"

# Exit with training status
exit $TRAIN_STATUS 

# Clean-up trap — remove staged data even if job is killed
WORK_DIR=""; trap '[[ -n $WORK_DIR && -d $WORK_DIR ]] && rm -rf "$WORK_DIR"' EXIT 