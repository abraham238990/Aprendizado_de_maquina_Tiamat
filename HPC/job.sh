#!/bin/bash
#SBATCH --job-name=optuna_hiperparametros
#SBATCH --output=optuna_hiperparametros_%j.log
#SBATCH --output=tiamat_%j.out
#SBATCH --error=tiamat_%j.err
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G

# Evitar conflitos
export OMP_NUM_THREADS=1

# Execução do script Python no ambiente virtual 
/home/abraham2610048/env_analise/bin/python /home/abraham2610048/aprendizado_maquina/job.py