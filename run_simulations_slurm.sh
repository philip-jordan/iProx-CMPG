for ENV in pollution energy; do
    for M in 8 4 2; do
        for S in 0 1 2 3 4 5 6 7 8 9; do
            sbatch --time=24:00:00 --output="$(pwd)/logs.txt" -n 1 --mem-per-cpu="4MB" --wrap="python main.py --env_type $ENV --num_players $M --seed $S"
        done
    done
done
