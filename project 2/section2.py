import math
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

class domain:
    def __init__(self):
        self.gamma = 0.95
        self.time_step = 0.1
        self.integration_step = 0.001
        self.actions = [-4, 4]

    def reward(self, state, action):
        next_p, next_s = self.dynamic(state, action)
        if next_p < -1 or abs(next_s) > 3:
            return -1
        if next_p > 1 and abs(next_s) <= 3:
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


class agent_decelerate:
    def __init__(self):
        pass

    def chose_action(self, state):
        return -4


class agent_random:
    def __init__(self):
        pass

    def chose_action(self, state):
        rand = np.random.uniform()
        if rand < 0.5:
            return -4
        else:
            return 4


class agent_accelerate:
    def __init__(self):
        pass

    def chose_action(self, state):
        return 4


def compute_J(state, domain, agent, N):
    J = np.zeros(N)
    a = agent.chose_action(state)
    r = domain.reward(state, a)
    next_s = domain.dynamic(state, a)
    J[0] = r
    state = next_s
    for i in range(1, N):
        # policy is always accelerate
        if domain.terminal_state(state):
            J[i] = J[i - 1]
            continue
        a = agent.chose_action(state)
        r = domain.reward(state, a)
        next_s = domain.dynamic(state, a)
        J[i] = r + domain.gamma * J[i - 1]
        state = next_s
    return J


def monte_carlo_J(domain, agent, nbr_state, N):
    start_list = np.random.uniform(-0.1, 0.1, nbr_state)
    J_tot = np.zeros(N)

    for i in tqdm(range(len(start_list))):
        s = (start_list[i], 0)
        J = compute_J(s, domain, agent, N)
        J_tot = J_tot + J

    return J_tot / nbr_state


def plot_J(J):
    plt.figure()
    plt.plot(J)
    plt.xlabel('N')
    plt.ylabel(r'$J_N^{\mu}$')
    plt.grid()
    plt.show()



if __name__ == "__main__":
    domain = domain()
    agent = agent_accelerate()
    N = 200
    nbr_start = 50
    J = monte_carlo_J(domain, agent, nbr_start, N)
    plot_J(J)
