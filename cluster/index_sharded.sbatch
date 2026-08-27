#!/bin/bash

#SBATCH -J index_sharded               # Job name
#SBATCH --array=0-1
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=8          # Number of CPU cores per task
#SBATCH --nodes=1                  # Ensure that all cores are on the same machine with nodes=1
#SBATCH --partition=2080-galvani   # Which partition will run your job
#SBATCH --time=4-00:00             # Allowed runtime in D-HH:MM
#SBATCH --gres=gpu:1               # (optional) Requesting type and number of GPUs
#SBATCH --mem=50G                  # Total memory pool for all cores (see also --mem-per-cpu); exceeding this number will cause your job to fail.
#SBATCH --output=cluster/logs/index_sharded-%A_%a.out       # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --output=cluster/logs/index_sharded-%A_%a.err       # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=FAIL            # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=bileam.scheuvens@student.uni-tuebingen.de   # Email to which notifications will be sent

# Diagnostic and Analysis Phase - please leave these in.
scontrol show job $SLURM_JOB_ID
pwd
nvidia-smi # only if you requested gpus

# Setup Phase
# add possibly other setup code here, e.g.
# - copy singularity images or datasets to local on-compute-node storage like /scratch_local
# - loads virtual envs, like with anaconda
# - set environment variables
# - determine commandline arguments for `srun` calls

# Compute Phase
cd "$SLURM_SUBMIT_DIR"
singularity exec --nv --env-file .env --bind $SLURM_SUBMIT_DIR:/workspace cluster/container.sif /bin/bash -c ". /venv/bin/activate && cd /workspace && python3 -m build_eval_index experiment=hardneg_2500 shard_i=$SLURM_ARRAY_TASK_ID k_shards=$((SLURM_ARRAY_TASK_MAX+1))"
