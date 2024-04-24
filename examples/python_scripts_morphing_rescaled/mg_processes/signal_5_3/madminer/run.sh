#!/bin/bash

# Master script to generate events for MadMiner

# Usage: run.sh [MG_directory] [MG_process_directory] [log_directory]

mgdir=${1:-/remote/gpu02/crescenzo/MG5_aMC_v3_5_3}
mgprocdir=${2:-./mg_processes/signal_5_3}
mmlogdir=${3:-logs/signal_5}

$mgprocdir/madminer/scripts/run_0.sh $mgdir $mgprocdir $mmlogdir