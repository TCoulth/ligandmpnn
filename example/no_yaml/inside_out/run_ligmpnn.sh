#!/bin/bash

#SBATCH -J test_ligandmpnn
#SBATCH -p kuhlab
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 0-01:00:00
#SBATCH --mem=8g
#SBATCH --qos gpu_access
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate ligandmpnn
module load gcc
module load cuda

python /proj/kuhl_lab/LigandMPNN/run.py \
        --seed 111 \
        --pdb_path "./1BC8.pdb" \
        --out_folder "./outputs/" \
        --checkpoint_ligand_mpnn  /proj/kuhl_lab/LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt \
        --inside_out_decoding \
        --fixed_residues "C44 C45 C3"
