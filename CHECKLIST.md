# ✅ TinyVLA HPC Setup Checklist

## Pre-Setup (Before connecting to HPC)
- [ ] Have HPC account and login credentials ready
- [ ] Know your HPC's module names (python, cuda, gcc)
- [ ] Have dataset ready for upload
- [ ] Repository access confirmed

## Phase 1: Initial Setup
- [ ] Connect to HPC: `ssh username@hpc-cluster.edu`
- [ ] Navigate to working directory: `cd $HOME` or `cd /scratch/username`
- [ ] Clone repository: `git clone <repo_url> vla-vlm-test`
- [ ] Enter directory: `cd vla-vlm-test`
- [ ] Edit `setup_hpc.sh` with correct module names
- [ ] Make executable: `chmod +x setup_hpc.sh`
- [ ] Run setup: `bash setup_hpc.sh`
- [ ] Check output for any errors
- [ ] Verify environment: `conda activate tinyvla`

## Phase 2: Data Preparation  
- [ ] Upload dataset to `datasets/` directory
- [ ] Verify dataset structure matches expected format
- [ ] Download base model weights if needed
- [ ] Check data paths in config files

## Phase 3: Testing
- [ ] Edit `test_hpc.sh` with correct SLURM parameters
- [ ] Make executable: `chmod +x test_hpc.sh`
- [ ] Submit test job: `sbatch test_hpc.sh`
- [ ] Check job status: `squeue -u $USER`
- [ ] Verify test output: `cat logs/test_*.out`
- [ ] Confirm all imports work and GPU is available

## Phase 4: Training Configuration
- [ ] Edit `configs/train_full_diffusion.yaml`:
  - [ ] Verify `batch_size: 4` (not 1!)
  - [ ] Check `data_root` path is correct
  - [ ] Confirm `train_tasks` match your dataset
  - [ ] Set appropriate `max_steps` and `save_steps`
- [ ] Edit `train_hpc.sh` with correct SLURM parameters:
  - [ ] Partition name (`--partition=gpu`)
  - [ ] Time limit (`--time=12:00:00`) 
  - [ ] Memory (`--mem=32G`)
  - [ ] Module names match your HPC

## Phase 5: Training Execution
- [ ] Make executable: `chmod +x train_hpc.sh`
- [ ] Submit training job: `sbatch train_hpc.sh`
- [ ] Record job ID for monitoring
- [ ] Monitor progress: `tail -f logs/train_*.out`
- [ ] Check for errors: `cat logs/train_*.err`
- [ ] Verify GPU utilization if possible: `nvidia-smi`

## Phase 6: Training Monitoring
- [ ] Check job status regularly: `squeue -u $USER`
- [ ] Monitor loss decreasing in logs
- [ ] Verify checkpoints being saved: `ls VLA_weights/full_training_adapter/step_*/`
- [ ] No NaN/Inf errors appearing
- [ ] Training progressing smoothly

## Phase 7: Post-Training
- [ ] Training completed successfully
- [ ] Final checkpoint exists: `ls VLA_weights/full_training_adapter/step_*/`
- [ ] Run evaluation: `python tinyvla_test.py`
- [ ] Test model loading and inference
- [ ] Document final results

## Emergency Troubleshooting
If something goes wrong:
- [ ] Check error logs: `cat logs/train_*.err`
- [ ] Verify environment: `conda list`
- [ ] Test imports manually: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Check module loading: `module list`
- [ ] Verify data paths exist
- [ ] Reduce batch size if memory issues
- [ ] Check SLURM parameters match HPC requirements

## Key Success Indicators
- [ ] ✅ Environment setup with no import errors
- [ ] ✅ GPU available and detected
- [ ] ✅ Test job completes successfully
- [ ] ✅ Training starts without immediate crashes
- [ ] ✅ Loss values are finite (not NaN/Inf)
- [ ] ✅ Checkpoints being saved regularly
- [ ] ✅ Training completes without errors

## Critical Files Checklist
- [ ] `setup_hpc.sh` - Environment setup script
- [ ] `train_hpc.sh` - Training SLURM job script  
- [ ] `test_hpc.sh` - Testing SLURM job script
- [ ] `configs/train_full_diffusion.yaml` - Training configuration
- [ ] `requirements_lora_vla.txt` - Python dependencies
- [ ] `HPC_WORKFLOW.md` - Detailed instructions

**🚨 Most Important**: Batch size = 4 (not 1) to fix diffusion model issues!

Print this checklist and check off items as you complete them. Good luck! 🚀 