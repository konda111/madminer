#!/bin/bash
#PBS -l nodes=1:ppn=1:bigmem1
#PBS -q bigmem1
#PBS -m ae
#PBS -e /remote/gpu02/crescenzo/MadMiner/examples/notebooks_morphing/outputs/run_sim
#PBS -o /remote/gpu02/crescenzo/MadMiner/examples/notebooks_morphing/outputs/run_sim
cd /remote/gpu02/crescenzo/MadMiner/examples/notebooks_morphing
mkdir -p ./outputs/run_sim

./mg_processes/signal_$NUM/madminer/run.sh