for v in 0 1 2 3 4 5 
do
    qsub -v NUM="$v" -t 4,5 cs_analysis.sh 
done
