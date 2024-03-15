for v in 0 1 2 3 4 5
do
    qsub -v NUM="$v" simulation_ratio.sh
done
