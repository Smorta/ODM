import math
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor

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
        return 1 / (1 + 5 * (p ** 2)) ** 1.5

    @staticmethod
    def hill_dd(p):
        if p < 0:
            return 2
        return (-20 * p) / 3 * (1 + 5 * p ** 2) ** (5 / 3)

    @staticmethod
    def terminal_state(state):
        position = state[0]
        speed = state[1]
        return abs(position) >= 1 or abs(speed) >= 3

    def dynamic(self, state, action):
        m = 1
        g = 9.81
        next_p = state[0]
        next_s = state[1]
        nbr_int_step = int(self.time_step / self.integration_step)
        for i in range(nbr_int_step):
            s = next_s
            p = next_p
            s_d = (action / (m * (1 + pow(self.hill_d(p), 2)))) - ((g * self.hill_d(p)) / (1 + pow(self.hill_d(p), 2))) \
                  - ((pow(s, 2) * self.hill_d(p) * self.hill_dd(p)) / (1 + pow(self.hill_d(p), 2)))
            p_d = s
            next_p = p + self.integration_step * p_d
            next_s = s + self.integration_step * s_d
        return next_p, next_s


class agent_accelerate:
    def __init__(self):
        pass

    def chose_action(self, state):
        return 4


def sup_learning_tech(X, y, tech=0):
    if tech == 0:
        return LinearRegression().fit(X, y)
    elif tech == 1:
        return ExtraTreesRegressor().fit(X, y)  # We can change number of estimators, currently = 100
    elif tech == 2:
        return MLPRegressor().fit(X, y)  # Change the intern structure here
    else:
        print("Error")
        return 0


def fitted_Q(Q, N, trajectory, tech):
    X = []
    y = []
    for state in trajectory:
        x_i = state[0]
        u_i = state[1]
        r_i = state[2]
        x_next_i = state[3]

        X.append((x_i, u_i))

        if N == 1:
            y.append(r_i)
        else:
            new_r = r_i + 0.95 * max(Q(x_next_i, -4),
                                     Q(x_next_i, 4))
            y.append(new_r)
    new_Q = sup_learning_tech(X, y, tech)
    return new_Q
