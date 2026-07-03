#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=32
#SBATCH --mem-per-cpu=5960
#SBATCH --gres=gpu:lovelace_l40:1
#SBATCH --time=48:00:00
#SBATCH --output=logs-transformer-hpt/hpt-dual_%j.out
#SBATCH --error=logs-transformer-hpt/hpt-dual_%j.err

module --force purge

cd /springbrook/share/wbs/bstvvz

# activate venv
source .venv/bin/activate

mkdir -p logs-transformer-hpt
python src/hpt_dual_path.py
