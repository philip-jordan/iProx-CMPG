import numpy as np
import random
import argparse
from player import Player
from environments import DemandResponseMarketEnv, PollutionTaxEnv
from algorithm import iprox_cmpg


def run_experiment(m, env_type, seed):
    np.random.seed(seed)
    random.seed(seed)

    if env_type == "pollution":
        env = PollutionTaxEnv(m)
        stepsizes = {2: 0.005, 4: 0.002, 8: 0.0007}
        num_samples = {2: 1000, 4: 1000, 8: 2500}
        max_iters = 20
        if m not in stepsizes:
            raise ValueError("Invalid number of players for pollution environment")
    elif env_type == "energy":
        env = DemandResponseMarketEnv(m)
        stepsizes = {2: 0.002, 4: 0.001, 8: 0.0003}
        num_samples = {2: 100, 4: 150, 8: 200}
        max_iters = 60
        if m not in stepsizes:
            raise ValueError(
                "Invalid number of players for demand response market environment"
            )

    players = [Player(env) for _ in range(m)]
    iprox_cmpg(
        mu=[1 / env.num_states] * env.num_states,
        players=players,
        env=env,
        max_iters=max_iters,
        max_inner_iters=20,
        inner_eta=stepsizes[m],
        outer_eta=0.1,
        gamma=0.9,
        T=10,
        num_samples=num_samples[m],
        seed=seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_type", type=str, choices=["pollution", "energy"], default="pollution"
    )
    parser.add_argument("--num_players", type=int, choices=[2, 4, 8], default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_experiment(args.num_players, args.env_type, args.seed)
