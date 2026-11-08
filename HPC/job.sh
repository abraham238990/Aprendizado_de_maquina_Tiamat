#!/bin/bash
#SBATCH --job-name=optuna_hiperparametros
#SBATCH --output=optuna_hiperparametros_%j.log
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# Evitar conflitos
export OMP_NUM_THREADS=1

# Execucao do script Python no ambiente virtual 
/home/abraham2610048/env_analise/bin/python /home/abraham2610048/aprendizado_maquina/job.py