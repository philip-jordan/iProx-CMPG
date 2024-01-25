import numpy as np
from utils import visit_dist, value_function, q_function
import copy
import os


def iprox_cmpg( mu, players, env, max_iters, max_inner_iters, inner_eta, outer_eta, gamma, T, num_samples, seed):
    potential_hist, value_c_hist = [], []

    for t in range(max_iters):
        print(t, "/", max_iters)
        snapshot_policies = copy.deepcopy([p.get_policy() for p in players])
        potential, value_c = [], []

        for _ in range(max_inner_iters):
            b_dist = [0] * env.num_states
            for state in range(env.num_states):
                a_dist = visit_dist(state, players, env, gamma, T, num_samples)
                b_dist[state] = np.dot(a_dist, mu)

            grads = np.zeros((len(players), env.num_states, env.num_actions))
            grads_c = np.zeros((len(players), env.num_states, env.num_actions))
            value_fun, value_fun_cost, potential_value = value_function(
                players, env, gamma, T, num_samples
            )

            potential.append(np.mean(list(potential_value.values())))
            value_c.append(np.mean(list(value_fun_cost.values())))

            # collect gradients
            for i, player in enumerate(players):
                for state in range(env.num_states):
                    for action in range(env.num_actions):
                        q_value, q_value_cost = q_function(i, state, action, players, env, gamma, value_fun, value_fun_cost, num_samples)
                        grads[i, state, action] = b_dist[state] * q_value
                        grads_c[i, state, action] = b_dist[state] * q_value_cost

            # primal update
            for i, player in enumerate(players):
                cur_policy = player.get_policy()
                for state in range(env.num_states):
                    if np.dot(mu, list(value_fun_cost.values())) - env.cost_UB < 0:
                        player.update_policy(state, grads[i, state], inner_eta)
                    else:
                        player.update_policy(
                            state,
                            -grads_c[i, state] - (1 / outer_eta) * (cur_policy[state] - snapshot_policies[i][state]),
                            inner_eta,
                        )
        potential_hist += potential
        value_c_hist += value_c

        experiment_dir = (
            "experiments/" + env.short_name + "/" + str(len(players)) + "/" + str(seed)
        )
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)

        np.save(experiment_dir + "/potential_hist.npy", np.array(potential_hist))
        np.save(experiment_dir + "/value_c_hist.npy", np.array(value_c_hist))
