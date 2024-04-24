#!/bin/bash
#PBS -l nodes=1:ppn=1:bigmem1
#PBS -q bigmem1
#PBS -m ae
#PBS -e /remote/gpu02/crescenzo/MadMiner/examples/python_scripts_morphing/outputs/run_sim
#PBS -o /remote/gpu02/crescenzo/MadMiner/examples/python_scripts_morphing/outputs/run_sim

echo $NUM $PBS_ARRAYID