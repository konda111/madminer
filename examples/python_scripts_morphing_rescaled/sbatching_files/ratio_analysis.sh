#!/bin/bash
#PBS -l nodes=1:ppn=1:bigmem1
#PBS -q bigmem1
#PBS -m ae
#PBS -e /remote/gpu02/crescenzo/MadMiner/examples/python_scripts_morphing_rescaled/outputs/analysis
#PBS -o /remote/gpu02/crescenzo/MadMiner/examples/python_scripts_morphing_rescaled/outputs/analysis
cd /remote/gpu02/crescenzo/MadMiner/examples/python_scripts_morphing_rescaled
mkdir -p ./outputs/analysis

mkdir -p ./data/training


NUM=$NUM PYTHONPATH=/remote/gpu02/crescenzo/environment/lib/python3.11 /remote/gpu02/crescenzo/environment/bin/python3 4_analyze_training_data.py