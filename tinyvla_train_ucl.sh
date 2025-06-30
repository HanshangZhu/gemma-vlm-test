#!/bin/bash -l
###############################################################################
# tinyvla_train_ucl.sh – TinyVLA 700M Training for UCL HPC (SGE/PBS)
###############################################################################

# SGE/PBS job directives (adjust based on UCL's specific system)
#$ -cwd                              # Execute in the directory qsub is run from
#$ -N tinyvla-700m-mt50              # Job name
#$ -pe smp 16                        # 16 CPU cores
#$ -l gpu=2,gpumem=80G               # 2×80 GB A100s
#$ -l h_rt=48:00:00                  # 48 h wall-clock
#$ -l mem=125G                       # Total host memory (~8 GB per core + headroom)
#$ -l tmpfs=50G                      # Local NVMe scratch
# ---- Optional specific queue (uncomment if confirmed online) ----
# #$ -q Daenerys
#-----------------------------------------------------------------
#$ -j y                              # Merge stdout/stderr
#$ -o tinyvla_$JOB_ID.log           # Consolidated log

# Email notifications (optional)
# #$ -M your.email@ucl.ac.uk
# #$ -m bes

###############################################################################
# 1. ENVIRONMENT SETUP
###############################################################################
date                                       # Log start time
echo "🚀 TinyVLA Training Job Starting"
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
# 2. CONDA ENVIRONMENT SETUP
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
    conda create --name tinyvla --file requirements.txt -y
    conda activate tinyvla
fi

# Verify environment
echo "✅ Python: $(which python)"
echo "✅ Python version: $(python --version)"
echo "✅ PyTorch: $(python -c 'import torch; print(f"v{torch.__version__}, CUDA: {torch.cuda.is_available()}")')"

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
# 4. GPU AND PERFORMANCE SETUP
###############################################################################
echo "⚙️ Configuring GPU settings..."

# Set GPU visibility (UCL typically assigns GPUs automatically)
#export CUDA_VISIBLE_DEVICES="0,1"         # Use assigned GPUs
#export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# Performance optimizations
export OMP_NUM_THREADS=8                   # OpenMP threads
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
# 5. TRAINING CONFIGURATION
###############################################################################
echo "🎓 Preparing training configuration..."

# Choose config based on available resources
CONFIG_FILE="configs/train_hpc_700m_mt50.yaml"  # For high-memory GPUs (24GB+)
echo "📝 Using HPC config for high-memory GPUs"

# Verify config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    echo "📋 Available configs:"
    ls -la configs/
    exit 1
fi

# Create timestamped output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/ucl_run_${JOB_ID}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"
echo "📁 Output directory: $OUTPUT_DIR"

###############################################################################
# 6. LAUNCH TRAINING
###############################################################################
echo "🚀 Starting TinyVLA training..."
echo "📝 Config: $CONFIG_FILE"
echo "📊 Logging to: $OUTPUT_DIR/training.log"

# Run training with comprehensive logging
python train_lora.py \
    --config "$CONFIG_FILE" \
    2>&1 | tee "$OUTPUT_DIR/training.log"

# Capture exit status
TRAIN_STATUS=$?

###############################################################################
# 7. POST-TRAINING SYNC AND CLEANUP
###############################################################################
echo "🧹 Post-training cleanup..."

if [ $TRAIN_STATUS -eq 0 ]; then
    echo "✅ Training completed successfully!"
    
    # Create results summary
    cat > "$OUTPUT_DIR/job_summary.txt" << EOF
UCL HPC Training Job Summary
============================
Job ID: $JOB_ID
Hostname: $(hostname)
Start Time: $(date)
Config Used: $CONFIG_FILE
Output Directory: $OUTPUT_DIR
Training Status: SUCCESS
EOF

    # Sync results back to Scratch
    echo "💾 Syncing results to Scratch space..."
    rsync -av "$OUTPUT_DIR/" "$PROJECT_SOURCE/ucl_results/"
    
    # Optional: Sync checkpoints
    if [ -d "checkpoints" ]; then
        rsync -av checkpoints/ "$PROJECT_SOURCE/ucl_checkpoints/"
    fi
    
else
    echo "❌ Training failed with exit code: $TRAIN_STATUS"
    
    # Still save logs for debugging
    rsync -av "$OUTPUT_DIR/" "$PROJECT_SOURCE/ucl_debug/"
fi

###############################################################################
# 8. RESOURCE USAGE SUMMARY
###############################################################################
echo "📈 Final resource usage:"
echo "GPU Memory:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

echo "Disk Usage:"
du -sh "$WORK_DIR"

# Simple wall-clock usage summary
echo "Wallclock used: $((SECONDS/3600))h $(((SECONDS%3600)/60))m"

echo "Job completed at: $(date)"
echo "Results saved to: $PROJECT_SOURCE/ucl_results/"

# Exit with training status
exit $TRAIN_STATUS 

# Clean-up trap — remove staged data even if job is killed
WORK_DIR=""; trap '[[ -n $WORK_DIR && -d $WORK_DIR ]] && rm -rf "$WORK_DIR"' EXIT 