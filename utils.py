import numpy as np

"""
Code in this file is based on an implementation provided for [Leonardos et al. (2022)],
see https://openreview.net/forum?id=gfwON7rAm4
"""


def project_onto_simplex(v, z=1):
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w


def visit_dist(state, players, env, gamma, T, num_samples):
    # This is the unnormalized visitation distribution. Since we take finite trajectories,
    # the normalization constant is (1-gamma**T)/(1-gamma).
    visit_states = {st: np.zeros(T) for st in range(env.num_states)}
    for _ in range(num_samples):
        cur_state = env.reset(state)
        for t in range(T):
            visit_states[cur_state][t] += 1
            joint_action = [p.get_action(state) for p in players]
            cur_state, _, _, _, _ = env.step(joint_action)
    dist = [
        np.dot(v / num_samples, gamma ** np.arange(T))
        for (_, v) in visit_states.items()
    ]
    return dist


def value_function(players, env, gamma, T, num_samples):
    value_fun = {(s, i): 0 for s in range(env.num_states) for i in range(len(players))}
    value_fun_cost = {s: 0 for s in range(env.num_states)}
    potential_value = {s: 0 for s in range(env.num_states)}
    for _ in range(num_samples):
        for state in range(env.num_states):
            env.reset(state)
            for t in range(T):
                joint_action = [p.get_action(state) for p in players]
                _, rewards, cost, potential, _ = env.step(joint_action)
                for i in range(len(players)):
                    value_fun[state, i] += (gamma**t) * rewards[i]
                value_fun_cost[state] += (gamma**t) * cost
                potential_value[state] += (gamma**t) * potential
    value_fun.update((x, v / num_samples) for (x, v) in value_fun.items())
    value_fun_cost.update((x, v / num_samples) for (x, v) in value_fun_cost.items())
    potential_value.update((x, v / num_samples) for (x, v) in potential_value.items())
    return value_fun, value_fun_cost, potential_value


def q_function(i, state, action, players, env, gamma, value_fun, value_fun_cost, num_samples):
    tot_reward, tot_cost = 0, 0
    for _ in range(num_samples):
        env.reset(state)
        joint_action = [p.get_action(state) for p in players]
        joint_action[i] = action
        next_state, rewards, cost, _, _ = env.step(joint_action)
        tot_reward += rewards[i] + gamma * value_fun[next_state, i]
        tot_cost += cost + gamma * value_fun_cost[next_state]
    return tot_reward / num_samples, tot_cost / num_samples
