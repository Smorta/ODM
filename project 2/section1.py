import math
import numpy as np


class domain:
    def __init__(self):
        self.gamma = 0.95
        self.time_step = 0.1
        self.integration_step = 0.001
        self.actions = [-4, 4]

    def reward(self, state, action):
        next_p, next_s = self.dynamic(state, action)
        if next_p < -1 or next_s > 3:
            return -1
        if next_p > 1 and next_s <= 3:
            return 1
        else:
            return 0

    @staticmethod
    def hill(p):
        if p < 0:
            return pow(p, 2) + p
        return math.sqrt(1 + 5 * pow(p, 2))

    @staticmethod
    def hill_d(p):
        if p < 0:
            return 2 * p + 1
        return 1 / ((1 + 5 * (p ** 2)) ** 1.5)

    @staticmethod
    def hill_dd(p):
        if p < 0:
            return 2
        return (-20 * p) / (3 * (1 + 5 * p ** 2) ** (5 / 2))

    @staticmethod
    def terminal_state(state):
        position = state[0]
        speed = state[1]
        return abs(position) >= 1 or abs(speed) >= 3

    def dynamic(self, state, action):
        g = 9.81
        next_p = state[0]
        next_s = state[1]
        nbr_int_step = int(self.time_step / self.integration_step)
        for i in range(nbr_int_step):
            s = next_s
            p = next_p
            hill_d = self.hill_d(p)
            deno = (1 + self.hill_d(p) ** 2)
            s_d = (action - (g * hill_d) - (hill_d * self.hill_dd(p) * s ** 2)) / deno
            p_d = s
            next_p = p + self.integration_step * p_d
            next_s = s + self.integration_step * s_d
        return next_p, next_s


class agent_accelerate:
    def __init__(self):
        pass

    def chose_action(self, state):
        return 4


if __name__ == "__main__":
    s = (-0.1, 0)
    domain = domain()
    agent = agent_accelerate()
    trajectory = []
    while not domain.terminal_state(s):
        # policy is always accelerate
        a = agent.chose_action(s)
        r = domain.reward(s, a)
        next_s = domain.dynamic(s, a)
        trajectory.append([s, a, r, next_s])
        s = next_s
    for traj in trajectory:
        print(traj)
    print('length of the trajectory = ', len(trajectory))
