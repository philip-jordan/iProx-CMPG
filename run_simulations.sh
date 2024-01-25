for ENV in pollution energy; do
    for M in 8 4 2; do
        for S in 0 1 2 3 4 5 6 7 8 9; do
            python3 main.py --env_type $ENV --num_players $M --seed $S
        done
    done
done
