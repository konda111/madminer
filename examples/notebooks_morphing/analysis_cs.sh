#!/bin/bash
#PBS -l nodes=1:ppn=1:bigmem1
#PBS -q bigmem1
#PBS -m ae
#PBS -e /remote/gpu02/crescenzo/MadMiner/examples/notebooks_morphing/outputs/abalyze_cs
#PBS -o /remote/gpu02/crescenzo/MadMiner/examples/notebooks_morphing/outputs/abalyze_cs

cd /remote/gpu02/crescenzo/MadMiner/examples/notebooks_morphing
mkdir -p ./data
mkdir -p outputs/abalyze_cs

export OMP_NUM_THREADS=16
NUM=$NUM PYTHONPATH=/remote/gpu02/crescenzo/environment/lib/python3.11 /remote/gpu02/crescenzo/environment/bin/python3 2_cs_analysis_ratio_morphing.py