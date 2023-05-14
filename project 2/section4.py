import math
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from tqdm import tqdm
from section2 import monte_carlo_J, domain


class agent_accelerate:
    def __init__(self):
        pass

    def chose_action(self, state):
        return 4


class agent_Q:
    def __init__(self, Q):
        self.Q = Q

    def chose_action(self, state):
        if self.Q.predict([[state[0], state[1], 4]])[0] > self.Q.predict([[state[0], state[1], -4]])[0]:
            return 4
        return -4


def sup_learning_tech(X, y, tech=0):
    if tech == 0:
        return LinearRegression().fit(X, y)
    elif tech == 1:
        return ExtraTreesRegressor(n_estimators=10).fit(X, y)  # We can change number of estimators, currently = 100
    elif tech == 2:
        return MLPRegressor(hidden_layer_sizes=(10, 20, 20, 20, 10), max_iter=800, activation='tanh').fit(X, y)  # Change the intern structure here
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
    p_array = np.arange(-1, 1, resolution)
    s_array = np.arange(-3, 3, resolution)
    Q_grid = np.zeros((2, p_array.shape[0], s_array.shape[0]))
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
    spatial_step = Q_grid.shape[1]
    speed_step = Q_grid.shape[2]
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
    c = axes.pcolormesh(p_list, s_list, Q_plot, cmap='bwr', vmin=l_c, vmax=r_c, shading='auto')
    # axes.set_title(title)
    axes.axis([l_a, r_a, l_b, r_b])
    axes.set_xlabel("Position")
    axes.set_ylabel("Speed")
    figure.colorbar(c)

    plt.show()


def norm_inf_Q(Q_reg, Q_old_reg, state_list):
    diff = np.zeros((len(state_list)))
    for i in range(len(state_list)):
        state = state_list[i]
        diff[i] = Q_reg.predict([[state[0][0], state[0][1], state[1]]])[0] - Q_old_reg.predict([[state[0][0], state[0][1], state[1]]])[0]
    norm_inf = abs(np.max(diff))
    return norm_inf


def stop_rule_1(epsilon, N_max, trajectory, tech):
    Q_old_reg = fitted_Q(None, 1, trajectory, tech)
    Q_reg = fitted_Q(Q_old_reg, 2, trajectory, tech)
    norm_inf = norm_inf_Q(Q_reg, Q_old_reg, trajectory)
    N = 3
    while epsilon < norm_inf and N < N_max:
        print(N)
        Q_old_reg = Q_reg
        Q_reg = fitted_Q(Q_reg, N, trajectory, tech)
        norm_inf = norm_inf_Q(Q_reg, Q_old_reg, trajectory)
        N += 1
    return Q_reg


def stop_rule_2(epsilon, domain, trajectory, tech):
    Q_reg = fitted_Q(None, 1, trajectory, tech)
    Br = 1  # maximum reward
    Optimalstep = math.log(epsilon * ((1 - domain.gamma) ** 2) / (2 * Br), domain.gamma)
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


def offline_1(nbr_episodes, domain, agent):
    trajectory = []
    initial_positions = np.random.uniform(-0.1, 0.1, nbr_episodes)
    for init_pos in tqdm(initial_positions):
        s = (init_pos, 0)
        while not domain.terminal_state(s):
            a = agent.chose_action(s)
            r = domain.reward(s, a)
            next_s = domain.dynamic(s, a)
            trajectory.append((s, a, r, next_s))
            s = next_s
    return trajectory


def offline_2(nbr_episodes, domain, agent):
    trajectory = []
    initial_positions = np.random.uniform(-1, 1, nbr_episodes)
    for i in tqdm(range(nbr_episodes)):
        s = (initial_positions[i], 0)
        while not domain.terminal_state(s):
            a = agent.chose_action(s)
            r = domain.reward(s, a)
            next_s = domain.dynamic(s, a)
            trajectory.append((s, a, r, next_s))
            s = next_s
    return trajectory


if __name__ == "__main__":
    domain = domain()
    agent_rand = agent_random()

    trajectory = offline_2(60, domain, agent_rand)
    Q_reg = stop_rule_2(0.01, domain, trajectory, 2)
    Q_dis = discret_Q(Q_reg, 0.01)
    policy = dicret_policy(Q_dis)
    display_colored_grid(Q_dis[0])
    display_colored_grid(Q_dis[1])
    display_colored_grid(policy)
    agent_q = agent_Q(Q_reg)
    J = monte_carlo_J(domain, agent_q, 50, 400)[-1]
    print('\nJ=', J, '\n')

    trajectory = offline_1(60, domain, agent_rand)
    Q_reg = stop_rule_2(0.01, domain, trajectory, 2)
    Q_dis = discret_Q(Q_reg, 0.01)
    policy = dicret_policy(Q_dis)
    display_colored_grid(Q_dis[0])
    display_colored_grid(Q_dis[1])
    display_colored_grid(policy)
    agent_q = agent_Q(Q_reg)
    J = monte_carlo_J(domain, agent_q, 50, 400)[-1]
    print('\nJ=', J, '\n')
