import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from tqdm import tqdm
import gymnasium as gym


def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=25)
    anim.save(path + filename, writer='imagemagick', fps=60)


class FIQ:
    def __init__(self, env_name, Nbr_action, gamma=0.95):
        self.trajectory = []
        self.Q = None
        self.emv_name = env_name
        self.env = gym.make(env_name)
        self.Nbr_features = self.env.observation_space.shape
        self.Nbr_action = Nbr_action
        self.Action_space = self.env.action_space
        self.action_list = np.linspace(self.env.action_space.low[0], self.env.action_space.high[0], Nbr_action)
        self.gamma = gamma

    def render(self, mode):
        self.env = gym.make(self.emv_name, render_mode=mode)

    @staticmethod
    def sup_learning_tech(X, y, tech=0):
        if tech == 0:
            return ExtraTreesRegressor(n_estimators=10).fit(X, y)  # We can change number of estimators, currently = 100
        elif tech == 1:
            return MLPRegressor(hidden_layer_sizes=(10, 20, 20, 10), max_iter=800, activation='tanh').fit(X,y)  # Change the intern structure here
        else:
            print("Error")
            return 0

    def gen_traj(self, Nbr_episode):
        self.trajectory = []

        for _ in range(Nbr_episode):
            state, _ = self.env.reset()
            while True:
                action = self.env.action_space.sample()
                next_state, reward, done, _, _ = self.env.step(action)
                if done:
                    reward = -100
                self.trajectory.append([state, action, reward, next_state])
                state = next_state

                if done:
                    break

    def best_action(self, state):
        q_list = np.zeros(len(self.action_list))
        for i in range(len(self.action_list)):
            x = np.append(state, self.action_list[i])
            q_list[i] = self.Q.predict([x])
        return self.action_list[np.argmax(q_list)]

    def perfomance_measure(self, Nbr_episode):
        mean = 0
        for _ in range(Nbr_episode):
            mean += self.test_policy()[1]
        return mean / Nbr_episode

    def test_policy(self, render=False):
        if render:
            self.render("rgb_array")
        frames = []
        G = 0
        step = 0
        state, _ = self.env.reset()
        while True:
            if render:
                frames.append(self.env.render())
            action = self.best_action(state)
            next_state, reward, done, _, _ = self.env.step([action])
            state = next_state
            G += reward
            step += 1
            if done:
                break
        return frames, G

    def fitted_Q(self, N, tech):
        if N == 1:
            self.Q = None
        X = []
        y = []
        for traj in self.trajectory:
            s_i = traj[0]
            u_i = traj[1]
            r_i = traj[2]
            s_next_i = traj[3]

            x = np.append(s_i, u_i)
            X.append(x)

            if N == 1:
                y.append(r_i)
            else:
                q_list = np.zeros(len(self.action_list))
                for i in range(len(self.action_list)):
                    x = np.append(s_next_i, self.action_list[i])
                    q_list[i] = self.Q.predict([x])
                new_r = r_i + 0.95 * max(q_list)
                y.append(new_r)

        new_Q = self.sup_learning_tech(X, y, tech)
        self.Q = new_Q

    def norm_inf_Q(self, Q_reg, Q_old_reg, state_list):
        diff = np.zeros((len(state_list)))
        for i in range(len(state_list)):
            state = state_list[i]
            diff[i] = Q_reg.predict([[state[0][0], state[0][1], state[1]]])[0] - \
                      Q_old_reg.predict([[state[0][0], state[0][1], state[1]]])[0]
        norm_inf = abs(np.max(diff))
        return norm_inf

    def stop_rule(self, epsilon, tech):
        self.fitted_Q(1, 0)
        Br = 1  # maximum reward
        Optimalstep = math.log(epsilon * ((1 - self.gamma) ** 2) / (2 * Br), self.gamma)
        steps = int(Optimalstep)
        N = 1
        for i in tqdm(range(steps)):
            self.fitted_Q(N, tech)
            N += 1


if __name__ == "__main__":
    fqi = FIQ('InvertedPendulum-v4', 2)
    G_array = []
    fqi.gen_traj(50)
    fqi.fitted_Q(1, 0)
    N = 1
    for i in tqdm(range(100)):
        fqi.fitted_Q(N, 0)
        G = fqi.perfomance_measure(10)
        G_array.append(G)
        N += 1
    frames, _ = fqi.test_policy(render=True)
    save_frames_as_gif(frames, path='./', filename='gym_animation1.gif')
    frames, _ = fqi.test_policy(render=True)
    save_frames_as_gif(frames, path='./', filename='gym_animation2.gif')
    fqi.env.close()

    plt.figure()
    plt.plot(G_array)
    plt.show()
    print('final performance: ', G_array[-1])