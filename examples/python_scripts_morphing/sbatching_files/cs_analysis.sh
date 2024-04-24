#!/bin/bash
#PBS -l nodes=1:ppn=1:bigmem1
#PBS -q bigmem1
#PBS -m ae
#PBS -e /remote/gpu02/crescenzo/MadMiner/examples/python_scripts_morphing/outputs/analysis
#PBS -o /remote/gpu02/crescenzo/MadMiner/examples/python_scripts_morphing/outputs/analysis
cd /remote/gpu02/crescenzo/MadMiner/examples/python_scripts_morphing
mkdir -p ./outputs/analysis
ARR_NUM=${PBS_ARRAYID}

mkdir -p ./data/generation_${ARR_NUM}
mkdir -p ./data/generation_${ARR_NUM}_cross_sections


NUM=$NUM ARR_NUM=${PBS_ARRAYID} PYTHONPATH=/remote/gpu02/crescenzo/environment/lib/python3.11 /remote/gpu02/crescenzo/environment/bin/python3 2_analyze_cs_data.py