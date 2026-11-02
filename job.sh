#!/bin/bash
#SBATCH --job-name=optuna_hiperparametros
#SBATCH --output=optuna_hiperparametros_%j.log
<<<<<<< HEAD
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# Evitar conflitos
export OMP_NUM_THREADS=1

# Execucao do script Python no ambiente virtual 
/home/abraham2610048/env_analise/bin/python /home/abraham2610048/aprendizado_maquina/job.py
=======
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00

# Execução do script Python no ambiente virtual
/home/abraham2610048/env_analise/bin/python /home/abraham2610048/aprendizado_maquina/job.py
>>>>>>> 08970b513e49b5c478f87028f8ad75cb1ed485eb
