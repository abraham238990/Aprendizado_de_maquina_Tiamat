#!/bin/bash
#SBATCH --job-name=optuna_hiperparametros
#SBATCH --output=optuna_hiperparametros_%j.log
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00

# Execução do script Python no ambiente virtual
/home/abraham2610048/env_analise/bin/python /home/abraham2610048/aprendizado_maquina/job.py
