#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=32
#SBATCH --mem-per-cpu=5960
#SBATCH --gres=gpu:lovelace_l40:1
#SBATCH --time=24:00:00
#SBATCH --output=logs-ft-transformer/ft_%j.out
#SBATCH --error=logs-ft-transformer/ft_%j.err

module --force purge

cd /springbrook/share/wbs/bstvvz

# activate venv
source .venv/bin/activate

mkdir -p logs-ft-transformer
python src/ft_transformer_benchmark.py
