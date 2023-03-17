import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random


class domain:
    def __init__(self, type):
        self.current_state = (3, 0)
        self.gamma = 0.99
        self.rewards = np.array([[-3, 1, -5, 0, 19],
                                 [6, 3, 8, 9, 10],
                                 [5, -8, 4, 1, -8],
                                 [6, -9, 4, 19, -5],
                                 [-20, -17, -4, -3, 9]])
        self.m = np.shape(self.rewards)[0]
        self.n = np.shape(self.rewards)[0]
        self.type = type

    def get_current_state(self):
        return self.current_state

    def get_type(self):
        return self.type

    def reward(self, state, action):
        x, y = self.dynamic(state, action)
        return self.rewards[x, y]

    def step(self, action):
        prev_state = self.current_state
        self.current_state = self.dynamic(self.current_state, action)
        return [prev_state, action, self.reward(prev_state, action), self.current_state]

    def function_j(self, agent, N):
        J = np.zeros((5, 5))
        for n in range(N):
            J_prev = J.copy()
            for i in range(5):
                for j in range(5):
                    state = (i, j)
                    action = agent.chose_action(state)
                    next_state = self.det_dynamic(state, action)
                    if self.type == 'det':
                        J[i, j] = self.rewards[next_state[0], next_state[1]] + self.gamma * J_prev[
                            next_state[0], next_state[1]]
                    else:
                        J[i, j] = 0.5 * (self.rewards[next_state[0], next_state[1]] + self.gamma * J_prev[
                            next_state[0], next_state[1]]) \
                                  + 0.5 * (self.rewards[0, 0] + self.gamma * J_prev[0, 0])
        return J

    def det_dynamic(self, state, action):
        return min(max(state[0] + action[0], 0), self.n - 1), min(max(state[1] + action[1], 0), self.m - 1)

    def dynamic(self, state, action):
        if self.type == 'det':
            return self.det_dynamic(state, action)
        else:
            rand = np.random.uniform()
            if rand <= 0.5:
                return min(max(state[0] + action[0], 0), self.n - 1), min(max(state[1] + action[1], 0), self.m - 1)
            else:
                return 0, 0


class optimal_agent:
    def __init__(self, mdp):
        self.mdp = mdp
        pass

    def chose_action(self, state):
        return self.mdp.get_best_action(state)


class agent_rand:
    def __init__(self):
        pass

    def chose_action(self, state):
        rand = np.random.uniform()

        if rand < 0.25:
            return -1, 0
        elif rand < 0.5:
            return 0, -1
        elif rand < 0.75:
            return 1, 0

        return 0, 1


class MDP:
    def __init__(self, domain):
        self.policy_grid = None
        self.actions = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        self.domain = domain

    def get_r(self, state, action):
        if self.domain.get_type == "stocha":
            return 0.5 * self.domain.reward(state, action) + 0.5 * self.domain.rewards[0, 0]
        return self.domain.reward(state, action)

    def get_p(self, state, action, next_state):
        if self.domain.get_type == "stocha":
            p = 0
            if next_state == self.domain.det_dynamic(state, action):
                p += 0.5
            if next_state == (0, 0):
                p += 0.5
            return p
        elif self.domain.det_dynamic(state, action) == next_state:
            return 1
        return 0

    def update_Q(self, Q_prev):
        Q = np.zeros((5, 5, 4))
        for i in range(5):
            for j in range(5):
                k = 0
                for action in self.actions:
                    gamma = self.domain.gamma
                    next_q = 0
                    next_state = self.domain.det_dynamic((i, j), action)
                    if self.domain.type == "stocha":
                        r = (self.domain.rewards[next_state[0], next_state[1]] + self.domain.rewards[0, 0]) / 2
                        next_q += 0.5 * max(Q_prev[next_state[0], next_state[1], 0],
                                            Q_prev[next_state[0], next_state[1], 1],
                                            Q_prev[next_state[0], next_state[1], 2],
                                            Q_prev[next_state[0], next_state[1], 3])
                        next_q += 0.5 * max(Q_prev[0, 0, 0], Q_prev[0, 0, 1],
                                            Q_prev[0, 0, 2], Q_prev[0, 0, 3])
                        Q[i, j, k] = r + gamma * next_q
                    else:
                        r = self.domain.rewards[next_state[0], next_state[1]]
                        next_q = max(Q_prev[next_state[0], next_state[1], 0],
                                     Q_prev[next_state[0], next_state[1], 1],
                                     Q_prev[next_state[0], next_state[1], 2],
                                     Q_prev[next_state[0], next_state[1], 3])
                        Q[i, j, k] = r + gamma * next_q
                    k += 1
        return Q

    def compute_Q(self, N):
        Q = np.zeros((5, 5, 4))
        for i in range(N):
            Q_prev = Q.copy()
            Q = self.update_Q(Q_prev)
        return Q

    def get_best_action(self, state):
        if self.policy_grid is not None:
            x, y = state
            return self.actions[int(self.policy_grid[x, y])]
        return -1

    def get_opt_policy(self):
        if self.policy_grid is not None:
            return self.policy_grid
        return -1

    def compute_best_policy(self):
        Q = self.compute_Q(1)
        k = 0
        if self.policy_grid is None:
            self.policy_grid = np.zeros((5, 5))
        while True:
            k += 1
            prev_policy = self.policy_grid.copy()
            self.compute_policy(Q)

            if k == 20:
                break

            Q = self.update_Q(Q)

        print(k, 'iteration needed to reach convergence')

    def compute_policy(self, Q):
        if self.policy_grid is None:
            self.policy_grid = np.zeros((5, 5))
        for i in range(5):
            for j in range(5):
                self.policy_grid[i, j] = np.argmax(Q[i, j])


class Q_learning:
    def __init__(self, d):
        self.policy_grid = None
        self.actions = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        self.domain = d
        self.trajectory = []
        self.N = np.zeros((5, 5))
        self.alpha = 0.05
        self.gamma = 0.99
        self.epsilon = 0.5
        self.Q = np.zeros((5, 5, 4))
        self.init_pos = (3, 0)

    def generate_traj(self, agent, T):
        state = self.domain.get_current_state()
        for i in range(T):
            action = agent.chose_action(state)
            next_state = self.domain.dynamic(state, action)
            reward = self.domain.rewards[next_state]
            state = next_state
            self.N[state[0], state[1]] += 1
            self.trajectory.append((state, action, reward))

    def compute_Q(self):
        for k in range(len(self.trajectory) - 1):
            Q_prev = self.Q.copy()
            gamma = self.domain.gamma
            state = self.trajectory[k][0]
            action = self.actions.index(self.trajectory[k][1])
            next_state = self.trajectory[k + 1][0]
            r = self.trajectory[k][2]
            next_q = r + gamma * max(Q_prev[next_state[0], next_state[1], 0],
                                     Q_prev[next_state[0], next_state[1], 1],
                                     Q_prev[next_state[0], next_state[1], 2],
                                     Q_prev[next_state[0], next_state[1], 3])
            self.Q[state[0], state[1], action] = (1 - self.alpha) * self.Q[
                state[0], state[1], action] + self.alpha * next_q
        return self.Q

    def compute_policy(self, Q):
        if self.policy_grid is None:
            self.policy_grid = np.zeros((5, 5))
        for i in range(5):
            for j in range(5):
                self.policy_grid[i, j] = np.argmax(Q[i, j])

    def get_best_action(self, state):
        if self.policy_grid is not None:
            x, y = state
            return self.actions[int(self.policy_grid[x, y])]
        return -1

    def j_opti_grid(self, N):
        pg = self.policy_grid
        J = np.zeros((5, 5))
        for n in range(N):
            J_prev = J.copy()
            for i in range(5):
                for j in range(5):
                    state = (i, j)
                    action = self.actions[int(pg[i, j])]
                    next_state = self.domain.det_dynamic(state, action)
                    if self.domain.type == 'det':
                        J[i, j] = self.domain.rewards[next_state[0], next_state[1]] + self.gamma * J_prev[
                            next_state[0], next_state[1]]
                    else:
                        J[i, j] = 0.5 * (self.domain.rewards[next_state[0], next_state[1]] + self.gamma * J_prev[
                            next_state[0], next_state[1]]) \
                                  + 0.5 * (self.domain.rewards[0, 0] + self.gamma * J_prev[0, 0])
        return J

    def plt_online(self, online, num, title="fig"):
        plt.figure(num)
        if self.domain.type == "stocha":
            normInf = []
            for i in range(10):
                normInf.append(online())

            plt.plot(range(100), np.percentile(normInf, 50, axis=0), label="Infinite norm")
            plt.plot(range(100), np.std(normInf, axis=0), label="Standard deviation")
            plt.fill_between(range(100),
                             np.percentile(normInf, 0, axis=0),
                             np.percentile(normInf, 100, axis=0), alpha=0.3)
            plt.ylabel(r'$|J_{µ_{\hat{Q}}}^{N} - J_{µ^{*}}^{N}|$')
            plt.xlabel("Number of episodes")
            plt.legend()
            plt.grid()
        else:
            plt.plot(online())
            plt.ylabel(r'$|J_{µ_{\hat{Q}}}^{N} - J_{µ^{*}}^{N}|$')
            plt.xlabel("Number of episodes")
            plt.grid()
        plt.savefig(title + ".pdf")

    def online_first(self, T=1000):
        transitions = T  # 1000
        episodes = 100  # 100
        normInf = []
        mdp = MDP(self.domain)
        mdp.compute_best_policy()
        op_a = optimal_agent(mdp)
        self.Q = np.zeros((5, 5, 4))  # Initialise Q to 0
        self.compute_policy(self.Q)

        for i in range(episodes):
            state = self.init_pos
            for j in range(transitions):
                rand = np.random.uniform()
                if rand < self.epsilon:
                    action_index = int(self.policy_grid[state])
                else:
                    action_index = np.random.randint(4)
                next_state = self.domain.dynamic(state, self.actions[action_index])
                r = self.domain.rewards[next_state]

                next_q = r + self.gamma * max(self.Q[next_state[0], next_state[1], 0],
                                              self.Q[next_state[0], next_state[1], 1],
                                              self.Q[next_state[0], next_state[1], 2],
                                              self.Q[next_state[0], next_state[1], 3])
                self.Q[state[0], state[1], action_index] = (1 - self.alpha) * self.Q[state[0], state[1], action_index] + \
                                                           self.alpha * next_q
                state = next_state
                self.compute_policy(self.Q)

            # CALCULATE || JNQ - JN ||_inf
            J_NQ = self.j_opti_grid(980)  # Comment choisir N?
            J_N = self.domain.function_j(op_a, 980)

            normInf.append(np.max(abs(J_NQ - J_N)))
        return normInf

    def online_second(self, T=1000):

        transitions = T  # 1000
        episodes = 100  # 100
        normInf = []

        mdp = MDP(self.domain)
        mdp.compute_best_policy()
        op_a = optimal_agent(mdp)
        self.Q = np.zeros((5, 5, 4))  # Initialise Q to 0
        self.compute_policy(self.Q)

        for i in range(episodes):
            state = self.init_pos
            alpha = self.alpha
            for j in range(transitions):
                rand = np.random.uniform()
                if rand < self.epsilon:
                    action_index = int(self.policy_grid[state])
                else:
                    action_index = np.random.randint(4)
                next_state = self.domain.dynamic(state, self.actions[action_index])

                r = self.domain.rewards[next_state]

                next_q = r + self.gamma * max(self.Q[next_state[0], next_state[1], 0],
                                              self.Q[next_state[0], next_state[1], 1],
                                              self.Q[next_state[0], next_state[1], 2],
                                              self.Q[next_state[0], next_state[1], 3])
                self.Q[state[0], state[1], action_index] = (1 - alpha) * self.Q[
                    state[0], state[1], action_index] + alpha * next_q
                state = next_state
                self.compute_policy(self.Q)
                alpha = alpha * 0.8

            # CALCULATE || JNQ - JN ||_inf
            J_NQ = self.j_opti_grid(980)  # Comment choisir N?
            J_N = self.domain.function_j(op_a, 980)
            normInf.append(np.max(abs(J_NQ - J_N)))
        return normInf

    def online_third(self, T=1000):
        buff = replay()
        transitions = T  # 1000
        episodes = 100  # 100
        normInf = []

        mdp = MDP(self.domain)
        mdp.compute_best_policy()
        op_a = optimal_agent(mdp)
        self.Q = np.zeros((5, 5, 4))  # Initialise Q to 0
        self.compute_policy(self.Q)

        for i in range(episodes):
            state = self.init_pos
            for j in range(transitions):
                rand = np.random.uniform()
                if rand < self.epsilon:
                    action_index = int(self.policy_grid[state])  #
                else:
                    action_index = np.random.randint(4)
                next_state = self.domain.dynamic(state, self.actions[action_index])
                reward = self.domain.rewards[next_state]

                buff.add(state, action_index, reward, next_state)
                state = next_state
                replay_trans = buff.draw()

                if replay_trans:
                    for k in range(len(replay_trans)):
                        Q_prev = self.Q.copy()
                        s = replay_trans[k][0]
                        a = replay_trans[k][1]
                        r = replay_trans[k][2]
                        n_s = replay_trans[k][3]
                        next_q = r + self.gamma * max(Q_prev[n_s[0], n_s[1], 0],
                                                      Q_prev[n_s[0], n_s[1], 1],
                                                      Q_prev[n_s[0], n_s[1], 2],
                                                      Q_prev[n_s[0], n_s[1], 3])
                        self.Q[s[0], s[1], a] = (1 - self.alpha) * Q_prev[
                            s[0], s[1], a] + self.alpha * next_q
                    self.compute_policy(self.Q)

            # CALCULATE || JNQ - JN ||_inf
            J_NQ = self.j_opti_grid(980)  # Comment choisir N?
            J_N = self.domain.function_j(op_a, 980)
            normInf.append(np.max(abs(J_NQ - J_N)))
        return normInf


class replay:
    def __init__(self):
        self.size = 0
        self.memory = []

    def add(self, state, action, reward, next_state):
        self.size += 1
        self.memory.append((state, action, reward, next_state))

    def show(self):
        print(self.size)
        print(self.memory)

    def draw(self, N=10):
        if self.size >= 10:
            return random.choices(self.memory, k=N)
        return None


def heatmap_visit(mdp):
    cols = ['1', '2', '3', '4', '5']
    rows = ['1', '2', '3', '4', '5']

    fig, ax = plt.subplots()
    im = ax.imshow(mdp.N, cmap='YlOrRd')

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(cols)), labels=cols)
    ax.set_yticks(np.arange(len(rows)), labels=rows)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(rows)):
        for j in range(len(cols)):
            text = ax.text(j, i, mdp.N[i, j],
                           ha="center", va="center", color="black")

    ax.set_title("number of visits for each case")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    d = domain('det')
    a = agent_rand()
    q_model = Q_learning(d)
    # q_model.generate_traj(a, 10 ** 7)
    # Q = q_model.compute_Q()
    # q_model.compute_policy(Q)
    # q_model.online_first()
    q_model.plt_online(q_model.online_third)
#  q_model.online_third()
# q_model.j_opti_grid(100)
