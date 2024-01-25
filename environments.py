import gym
import random
from itertools import combinations


ALPHA = 2
BETA = 0.25
C = 1.25


class DemandResponseMarketEnv(gym.Env):
    def __init__(self, num_players, num_states=5, num_actions=5):
        super(DemandResponseMarketEnv, self).__init__()
        self.short_name = "energy"
        self.num_players = num_players
        self.num_states = num_states
        self.num_actions = num_actions
        self.cost_UB = 16
        self.reset()

    def reset(self, state=None):
        self.state = self.num_states - 1 if state is None else state
        return self.state

    # action is tuple of size num_players representing joint action
    def step(self, action):
        rewards = [
            a**2 * ALPHA - a**2 * BETA * sum(action) - a * C**self.state
            for a in action
        ]
        cost = sum(action)
        w = random.randint(0, self.num_states - 1)
        if random.choices([True, False], weights=[0.9, 0.1]):
            self.state = round(2 * sum(action) / len(action) + w)  # just guessing here
        else:
            self.state = w
        self.state = max(min(self.state, self.num_states - 1), 0)
        potential = (
            ALPHA * sum([a for a in action])
            - BETA * sum([a**2 for a in action])
            - BETA * sum([a1 * a2 for (a1, a2) in combinations(action, 2)])
            - self.num_players * C**self.state
        )
        return self.state, rewards, cost, potential, False


POLLUTION_FREE, POLLUTED = 0, 1
CLEAN, DIRTY = 0, 1
PROFIT_PER_ITEM = 2
TAX_BY_NUM_PLAYERS = {2: 4, 4: 8, 8: 32}


class PollutionTaxEnv(gym.Env):
    def __init__(self, num_players, num_states=2, num_actions=2):
        super(PollutionTaxEnv, self).__init__()
        self.short_name = "pollution"
        self.num_players = num_players
        self.num_states = num_states
        self.num_actions = num_actions
        self.cost_UB = 12
        self.reset()

    def reset(self, state=None):
        self.state = POLLUTION_FREE if state is None else state
        return self.state

    # action is tuple of size num_players representing joint action
    def step(self, action):
        rewards = [
            (PROFIT_PER_ITEM if a == CLEAN else 2 * PROFIT_PER_ITEM)
            - (TAX_BY_NUM_PLAYERS[self.num_players] if self.state == POLLUTED else 0)
            for i, a in enumerate(action)
        ]
        cost = 2 * (self.num_players - sum(action)) / self.num_players
        potential = sum(rewards)
        self.state = POLLUTION_FREE if all([a == CLEAN for a in action]) else POLLUTED
        return self.state, rewards, cost, potential, False
