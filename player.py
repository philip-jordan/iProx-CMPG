import numpy as np
from utils import project_onto_simplex
from environments import PollutionTaxEnv, DemandResponseMarketEnv


class Player:
    def __init__(self, env):
        self.policy = np.zeros((env.num_states, env.num_actions)) / env.num_actions
        if isinstance(env, PollutionTaxEnv):
            self.policy[:, 0] = 0.7
            self.policy[:, 1] = 0.3
        elif isinstance(env, DemandResponseMarketEnv):
            self.policy[:, 0] = 1
        self.num_actions = env.num_actions

    def get_action(self, state):
        return np.random.choice(self.num_actions, p=self.policy[state])

    def get_policy(self):
        return self.policy

    def update_policy(self, state, grad, eta):
        self.policy[state] = project_onto_simplex(self.policy[state] + eta * grad)
        pass
