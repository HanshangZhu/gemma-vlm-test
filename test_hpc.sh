#!/bin/bash -l
###############################################################################
# tinyvla_test.sh – 5-minute sanity check on an 80 GB A100 node
###############################################################################

#$ -cwd                               # log/output in the dir you qsub from
#$ -N tinyvla-test                    # job name
#$ -l gpu=1,gpumem=80G                # 1 × 80 GB A100
#$ -pe smp 4                          # four CPU cores is plenty
#$ -l mem=24G                         # ~6 GB per core (adjust if you like)
#$ -l tmpfs=10G                       # scratch on local NVMe
#$ -l h_rt=00:05:00                   # five-minute wall-clock
#$ -j y                               # merge stdout/err into one file
#$ -o tinyvla_test_$JOB_ID.log        # log file name

# Uncomment **only** if you have confirmed Daenerys is up:
# #$ -q Daenerys

echo "Job   : $JOB_ID"
echo "Host  : $(hostname)"
echo "Start : $(date)"

echo "----------- nvidia-smi ----------"
module purge
module load cuda/12.0/gnu/11.2.0
nvidia-smi

echo "----------- sleep 300 ----------"
sleep 300

echo "End   : $(date)"
