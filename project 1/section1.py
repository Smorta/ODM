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

    def generate_traj(self, agent, N):
        for i in range(N):
            current_action = agent.chose_action(d.get_current_state())
            tmp = self.step(current_action)
            print('state = ', tmp[0], ', u = ', tmp[1], ', r = ', tmp[2], ', next_state = ', tmp[3])


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


if __name__ == "__main__":
    agent = agent_rand()
    print("Deterministic : \n")
    d = domain('det')
    d.generate_traj(agent, 10)
    print("\n\nStochastic : \n")
    d = domain('stocha')
    d.generate_traj(agent, 10)
