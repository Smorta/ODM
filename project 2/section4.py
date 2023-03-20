import math
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from tqdm import tqdm

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
        m = 1
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


def sup_learning_tech(X, y, tech=0):
    if tech == 0:
        return LinearRegression().fit(X, y)
    elif tech == 1:
        return ExtraTreesRegressor(n_estimators=10).fit(X, y)  # We can change number of estimators, currently = 100
    elif tech == 2:
        return MLPRegressor(hidden_layer_sizes=(10, 20, 20, 10), activation='relu', max_iter=800).fit(X, y)  # Change the intern structure here
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

        X.append([x_i[0], x_i[1], u_i])

        if N == 1:
            y.append(r_i)
        else:
            new_r = r_i + 0.95 * max(Q.predict([[x_next_i[0], x_next_i[1], -4]]),
                                     Q.predict([[x_next_i[0], x_next_i[1], 4]]))[0]
            y.append(new_r)

    new_Q = sup_learning_tech(X, y, tech)
    return new_Q


def discret_Q(Q_regr, resolution):
    p_array = np.arange(-1, 1, 0.01)
    s_array = np.arange(-3, 3, 0.01)
    Q_grid = np.zeros((2, p_array.shape[0], s_array.shape[0]))
    print("discrete_Q")
    i = 0
    for p in tqdm(p_array):
        j = 0# for the speed
        for s in s_array:  # for the position
            Q_grid[0][i][j] = Q_regr.predict([[p, s, -4]])[0]
            Q_grid[1][i][j] = Q_regr.predict([[p, s, 4]])[0]
            j += 1
        i += 1
    return Q_grid


def dicret_policy(Q_grid):
    spatial_step = Q_grid.shape[0]
    speed_step = Q_grid.shape[1]
    policy_grid = np.zeros((spatial_step, speed_step))
    for i in tqdm(range(spatial_step)):  # for the speed
        for j in range(speed_step):  # for the position
            if Q_grid[0][i][j] > Q_grid[1][i][j]:
                policy_grid[i][j] = -4
            else:
                policy_grid[i][j] = 4
    return policy_grid


def display_colored_grid(Q_grid):
    spatial_step = Q_grid.shape[0]
    speed_step = Q_grid.shape[1]
    p_list, s_list = np.meshgrid(np.linspace(-1, 1, spatial_step), np.linspace(-3, 3, speed_step))

    l_a = p_list.min()
    r_a = p_list.max()
    l_b = s_list.min()
    r_b = s_list.max()
    l_c, r_c = np.amax(Q_grid), np.amin(Q_grid)

    figure, axes = plt.subplots()
    Q_plot = np.swapaxes(Q_grid, 0, 1)
    c = axes.pcolormesh(p_list, s_list, Q_plot, cmap='bwr_r', vmin=l_c, vmax=r_c, shading='auto')
    # axes.set_title(title)
    axes.axis([l_a, r_a, l_b, r_b])
    axes.set_xlabel("Position")
    axes.set_ylabel("Speed")
    figure.colorbar(c)

    plt.show()


def stop_rule_1(epsilon, trajectory, tech):
    Q_prev = np.zeros((2, 20, 60))
    print("enter sr1")
    Q_reg = fitted_Q(None, 1, trajectory, tech)

    Q_dis = discret_Q(Q_reg, 0.1)
    N = 1
    while np.amax(Q_dis - Q_prev) > epsilon:
        print(np.amax(Q_dis - Q_prev))
        N += 1
        Q_prev = Q_dis.copy()
        Q_reg = fitted_Q(Q_reg, N, trajectory, tech)
        Q_dis = discret_Q(Q_reg, 0.1)
    return Q_reg


def stop_rule_2(epsilon, domain, trajectory, tech):
    Q_reg = fitted_Q(None, 1, trajectory, tech)
    Br = 1  # maximum reward
    Optimalstep = math.log(epsilon * (1 - domain.gamma) / Br, domain.gamma)
    steps = int(Optimalstep)
    N = 1
    for i in tqdm(range(steps)):
        Q_reg = fitted_Q(Q_reg, N, trajectory, tech)
        N += 1
    return Q_reg


class agent_random:
    def __init__(self):
        pass

    def chose_action(self, state):
        rand = np.random.uniform()
        if rand < 0.5:
            return -4
        else:
            return 4


def offline(nbr_episodes, domain, agent):
    trajectory = []
    initial_positions = np.random.uniform(-1, 1, nbr_episodes)
    i = 0
    for init_pos in initial_positions:
        print(i)
        i += 1
        s = (init_pos, 0)
        while not domain.terminal_state(s):
            a = agent.chose_action(s)
            r = domain.reward(s, a)
            next_s = domain.dynamic(s, a)
            trajectory.append((s, a, r, next_s))
            s = next_s
    return trajectory

if __name__ == "__main__":
    domain = domain()
    agent = agent_random()
    trajectory = offline(60, domain, agent)
    Q_reg = stop_rule_2(1, domain, trajectory, 1)
    Q_dis = discret_Q(Q_reg, 0.05)
    policy = dicret_policy(Q_dis)
    display_colored_grid(Q_dis[0])
    display_colored_grid(Q_dis[1])
    display_colored_grid(policy)
