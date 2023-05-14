import math
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from tqdm import tqdm
from section2 import monte_carlo_J, domain
from section4 import offline_1, offline_2


class agent_random:
    def __init__(self):
        pass

    def chose_action(self, state):
        rand = np.random.uniform()
        if rand < 0.5:
            return -4
        else:
            return 4


class agent_FIQ:
    def __init__(self, Q):
        self.Q = Q

    def chose_action(self, state):
        if self.Q.predict([[state[0], state[1], 4]])[0] > self.Q.predict([[state[0], state[1], -4]])[0]:
            return 4
        return -4


class FQI:
    def __init__(self, domain, trajectory, tech):
        self.trajectory = trajectory
        self.tech = tech
        self.domain = domain
        self.Q = None
        self.N = 1

    def update_q(self):
        X = []
        y = []
        for state in self.trajectory:
            x_i = state[0]
            u_i = state[1]
            r_i = state[2]
            x_next_i = state[3]

            X.append([x_i[0], x_i[1], u_i])

            if self.N == 1:
                y.append(r_i)
            else:
                new_r = r_i + 0.95 * max(self.Q.predict([[x_next_i[0], x_next_i[1], -4]]),
                                         self.Q.predict([[x_next_i[0], x_next_i[1], 4]]))[0]
                y.append(new_r)

        self.N += 1
        self.Q = self.sup_learning_tech(X, y, self.tech)

    @staticmethod
    def sup_learning_tech(X, y, tech=0):
        if tech == 0:
            return LinearRegression().fit(X, y)
        elif tech == 1:
            return ExtraTreesRegressor(n_estimators=10).fit(X, y)  # We can change number of estimators, currently = 100
        elif tech == 2:
            return MLPRegressor(hidden_layer_sizes=(5, 10, 10, 10, 5), max_iter=800, activation='tanh').fit(X,
                                                                                                              y)  # Change the intern structure here
        else:
            print("Error")
            return 0

    def train_model(self, epsilon):
        self.Q = None
        self.N = 1
        self.update_q()
        Br = 1  # maximum reward
        Optimalstep = math.log(epsilon * ((1 - self.domain.gamma) ** 2) / (2 * Br), self.domain.gamma)
        steps = int(Optimalstep)
        for _ in tqdm(range(steps)):
            self.update_q()
        return self.Q

    def learning_curve(self, steps):
        self.Q = None
        self.N = 1
        self.update_q()
        J = np.zeros((steps))
        for i in tqdm(range(steps)):
            self.update_q()
            agent_q = agent_FIQ(self.Q)
            J[i] = monte_carlo_J(self.domain, agent_q, 50, 400)[-1]
        return J


class Network(nn.Module):

    def __init__(self, input_size):
        super(Network, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, 10)
        self.fc2 = nn.Linear(10, 20)
        self.fc3 = nn.Linear(20, 20)
        self.fc4 = nn.Linear(20, 10)
        self.fc5 = nn.Linear(10, 1)

    def forward(self, state):
        x1 = torch.relu(self.fc1(state))
        x2 = torch.relu(self.fc2(x1))
        x3 = torch.relu(self.fc3(x2))
        x4 = torch.tanh(self.fc4(x3))
        q_values = self.fc5(x4)
        return q_values


class PQL:

    def __init__(self, input_size):
        self.Q = Network(input_size)
        self.domain = domain
        self.optimizer = optim.Adam(self.Q.parameters(), lr=0.01)

    def learn(self, trajectory, nb_epoch):
        for i in range(nb_epoch):
            loos_array = np.zeros((len(trajectory)))
            for j, state in enumerate(trajectory):
                x_i = state[0]
                u_i = state[1]
                r_i = state[2]
                x_next_i = state[3]
                input = torch.Tensor([x_i[0], x_i[1], u_i]).float().unsqueeze(0)
                output = self.Q(input)[0][0]
                next_left_input = torch.Tensor([x_next_i[0], x_next_i[1], -4]).float().unsqueeze(0)
                next_left_output = self.Q(next_left_input).detach()[0][0]
                next_right_input = torch.Tensor([x_next_i[0], x_next_i[1], 4]).float().unsqueeze(0)
                next_right_output = self.Q(next_right_input).detach()[0][0]
                target = r_i + self.domain.gamma * max(next_left_output, next_right_output)  # bellman equation
                td_loss = nn.MSELoss()(target, output)
                loos_array[j] = td_loss.item()
                self.optimizer.zero_grad()
                td_loss.backward()
                self.optimizer.step()

    def get_q(self, input):
        new_state = torch.Tensor(input).float().unsqueeze(0)
        q_value = self.Q(new_state)
        return q_value


class agent_PQL:
    def __init__(self, model):
        self.model = model

    def chose_action(self, state):
        left_input = [state[0], state[1], -4]
        left_q = self.model.get_q(left_input)
        right_input = [state[0], state[1], 4]
        right_q = self.model.get_q(right_input)
        if left_q > right_q:
            return -4
        return 4


def plot_J(J):
    plt.figure()
    plt.plot(J)
    plt.xlabel('N')
    plt.ylabel(r'$J_N^{\mu}$')
    plt.grid()
    plt.show()


def discret_PQL(pqm_model, resolution):
    p_array = np.arange(-1, 1, resolution)
    s_array = np.arange(-3, 3, resolution)
    Q_grid = np.zeros((2, p_array.shape[0], s_array.shape[0]))
    i = 0
    for p in tqdm(p_array):
        j = 0 # for the speed
        for s in s_array:  # for the position
            Q_grid[0][i][j] = pqm_model.get_q([p, s, -4])[0]
            Q_grid[1][i][j] = pqm_model.get_q([p, s, 4])[0]
            j += 1
        i += 1
    return Q_grid


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


def comparaison_protocol(episodes_list, domain):
    agent = agent_random()
    fqi_J = np.zeros((len(episodes_list)))
    PQL_J = np.zeros((len(episodes_list)))
    transition_nbr = np.zeros((len(episodes_list)))
    for i, episode in enumerate(episodes_list):
        trajectory = offline_2(episode, domain, agent)
        fqi = FQI(domain, trajectory, 2)
        fqi.train_model(0.01)
        agent_q = agent_FIQ(fqi.Q)
        fqi_J[i] = monte_carlo_J(domain, agent_q, 50, 400)[-1]
        pmq = PQL(3)
        pmq.learn(trajectory, 1)
        agent_pql = agent_PQL(pmq)
        PQL_J[i] = monte_carlo_J(domain, agent_pql, 50, 400)[-1]
        transition_nbr[i] = len(trajectory)

    plt.figure()
    plt.plot(transition_nbr, fqi_J)
    plt.plot(transition_nbr, PQL_J)
    plt.xlabel('n')
    plt.ylabel(r'$J_N^{\hat{\mu}^*}$')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    domain = domain()
    agent = agent_random()
    trajectory = offline_2(100, domain, agent)

    # fqi = FQI(domain, trajectory, 2)
    # J = fqi.learning_curve(220)
    # plot_J(J)

    # pmq = PQL(3)
    # pmq.learn(trajectory, 1)
    # agent_pmq = agent_PQL(pmq)
    # Q_dis = discret_PQL(pmq, 0.01)
    # Q_policy = dicret_policy(Q_dis)
    # display_colored_grid(Q_dis[0])
    # display_colored_grid(Q_dis[1])
    # display_colored_grid(Q_policy)
    # J = monte_carlo_J(domain, agent_pmq, 50, 400)[-1]
    # print(J)

    comparaison_protocol([25, 50, 100, 150, 200, 400], domain)
    # comparaison_protocol([2, 3, 4], domain)
