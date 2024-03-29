import numpy as np


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
                        J[i, j] = self.rewards[next_state[0], next_state[1]] + self.gamma * J_prev[next_state[0], next_state[1]]
                    else:
                        J[i, j] = 0.5 * (self.rewards[next_state[0], next_state[1]] + self.gamma * J_prev[next_state[0], next_state[1]])\
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


class optimal_agent:
    def __init__(self, mdp):
        self.mdp = mdp
        pass

    def chose_action(self, state):
        return self.mdp.get_best_action(state)


class model:
    def __init__(self):
        self.R_tot = {}
        self.N = {}
        self.N_action = {}

    def update(self, state, action, reward, next_state):
        if (state, action) not in self.R_tot.keys():
            self.R_tot[(state, action)] = 0
        self.R_tot[(state, action)] += reward

        if (state, action) not in self.N.keys():
            self.N[(state, action)] = 0
        self.N[(state, action)] += 1

        if (state, action, next_state) not in self.N_action.keys():
            self.N_action[(state, action, next_state)] = 0
        self.N_action[(state, action, next_state)] += 1

    def get_p(self, state, action, next_state):
        if (state, action, next_state) in self.N_action.keys():
            return self.N_action[(state, action, next_state)] / self.N[(state, action)]
        return -1

    def get_r(self, state, action):
        if (state, action) in self.R_tot.keys():
            return self.R_tot[(state, action)] / self.N[(state, action)]
        return -1


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
                        r = (self.domain.rewards[next_state[0], next_state[1]] + self.domain.rewards[0, 0])/2
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

            if np.array_equal(prev_policy, self.policy_grid):
                break

            Q = self.update_Q(Q)

        print(k)

    def compute_policy(self, Q):
        if self.policy_grid is None:
            self.policy_grid = np.zeros((5, 5))
        for i in range(5):
            for j in range(5):
                self.policy_grid[i, j] = np.argmax(Q[i, j])


# d = domain('det')
# a = agent_rand()
# for i in range(10):
#     current_action = a.chose_action(d.get_current_state())
#     tmp = d.step(current_action)
#     print('x' + str(i) + ' =', tmp[0], ', u =', tmp[1], ', r =', tmp[2], ', x' + str(i+1) + ' =', tmp[3])
#
# print((0.99**980/(1-0.99))*19)
#
# tab = np.zeros((5, 5))
# N = 980
# J = d.function_j(a, N)
# print(J)

d = domain('stocha')
mdp = MDP(d)
mdp.compute_best_policy()
a = optimal_agent(mdp)
J = d.function_j(a, 980)
print(J)
print(mdp.get_opt_policy())
