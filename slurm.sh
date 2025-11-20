#!/bin/bash
#SBATCH --job-name=simplify_cuda
#SBATCH --output=simplify_%j.out
#SBATCH --error=simplify_%j.err
#SBATCH --partition=interactive
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

# Initialize modules
source /etc/profile.d/modules.sh

# Clean out old modules
module --force purge

# Load base environment so CUDA modules are visible
module load env/deprecated/2020b

# Now load CUDA
module load system/CUDA/12.6.0

# Compile on the cluster (to avoid GLIBC issues)
nvcc simplify.cu -o simplify

# Run
./simplify
