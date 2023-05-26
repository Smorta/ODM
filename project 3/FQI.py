import math
import numpy as np
import pickle
import sys
import pandas as pd
import getopt
import argparse
from matplotlib import pyplot as plt
from matplotlib import animation
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
import time
from utils import ReplayBuffer
import gymnasium as gym


def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):
    # Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(f):
        patch.set_data(frames[f])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=25)
    anim.save(path + filename, writer='imagemagick', fps=60)


class FQI:
    def __init__(self, env_name, Nbr_action, bufferSize=10000):
        self.trajectory = []
        self.Q = None
        self.emv_name = env_name
        self.env = gym.make(env_name)
        self.Nbr_features = self.env.observation_space.shape[0]
        self.Nbr_action = Nbr_action
        self.Action_space = self.env.action_space
        self.action_list = np.linspace(self.env.action_space.low[0], self.env.action_space.high[0], Nbr_action)
        self.gamma = 0.99
        self.memory = ReplayBuffer(bufferSize, self.Nbr_features, self.env.action_space.shape[0])

    def save_model(self, path, name='model.pkl'):
        file_path = path + name
        with open(file_path, 'wb') as file:
            pickle.dump(self.Q, file)

    def load_model(self, path):
        file_path = path
        with open(file_path, 'rb') as file:
            self.Q = pickle.load(file)

    def render(self, mode):
        self.env = gym.make(self.emv_name, render_mode=mode)

    @staticmethod
    def sup_learning_tech(X, y, tech=0):
        if tech == 0:
            return ExtraTreesRegressor(n_estimators=10).fit(X, y)  # We can change number of estimators, currently = 100
        elif tech == 1:
            return MLPRegressor(hidden_layer_sizes=(10, 20, 20, 10), max_iter=800, activation='tanh').fit(X,
                                                                                                          y)  # Change the intern structure here
        else:
            print("Error")
            return 0

    def gen_traj(self, Nbr_episode):
        for _ in range(Nbr_episode):
            state, _ = self.env.reset()
            step = 0
            while True:
                action = self.env.action_space.sample()
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                if terminated:
                    reward = -100
                self.memory.add(state, action, reward, next_state, terminated)
                state = next_state

                if terminated or truncated or step > 200:
                    break
                step += 1

    def best_action(self, state):
        q_list = np.zeros(len(self.action_list))
        for i in range(len(self.action_list)):
            x = np.append(state, self.action_list[i])
            q_list[i] = self.Q.predict([x])
        return self.action_list[np.argmax(q_list)]

    def perfomance_measure(self, Nbr_episode, early_stop=None):
        scores = []
        steps = []
        for i in range(Nbr_episode):
            frames, step, score = self.play_episode(early_stop=early_stop)
            steps.append(step)
            scores.append(score)
        return min(steps), max(steps), np.mean(steps), np.mean(scores)

    def play_episode(self, render=False, early_stop=None):
        if render:
            self.render("rgb_array")
        frames = []
        step = 0
        G = 0
        state, _ = self.env.reset()
        while True:
            if render:
                frames.append(self.env.render())
            action = self.best_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step([action])
            if terminated:
                reward = -100
            self.memory.add(state, action, reward, next_state, terminated)
            state = next_state
            G = reward + self.gamma * G
            step += 1
            if early_stop is not None and step >= early_stop:
                break
            if terminated or truncated:
                break
        return frames, step, G

    def fitted_Q(self, N, tech):
        if N == 1:
            self.Q = None
        X = []
        y = []
        for i in range(min(self.memory.used_size, self.memory.buffer_size)):
            s_i = self.memory.state_buffer[i]
            u_i = self.memory.action_buffer[i]
            r_i = self.memory.reward_buffer[i]
            s_next_i = self.memory.next_state_buffer[i]
            done_i = self.memory.done_buffer[i]

            x = np.append(s_i, u_i)
            X.append(x)

            if N == 1:
                y.append(r_i)
            else:
                q_list = np.zeros(len(self.action_list))
                for j in range(len(self.action_list)):
                    x = np.append(s_next_i, self.action_list[j])
                    q_list[j] = self.Q.predict([x])
                new_r = r_i + 0.95 * max(q_list) * (1 - done_i)
                y.append(new_r)

        new_Q = self.sup_learning_tech(X, y, tech)
        self.Q = new_Q

    def stop_rule(self, epsilon):
        Br = 1  # maximum reward
        Optimalstep = math.log(epsilon * ((1 - self.gamma) ** 2) / (2 * Br), self.gamma)
        steps = int(Optimalstep)
        return steps


def plot_learning_curve(G_array, name):
    G_low = []
    G_high = []
    G_mean = []
    x = np.arange(len(G_array))
    for i in range(len(G_array)):
        G_low.append(G_array[i][0])
        G_high.append(G_array[i][1])
        G_mean.append(G_array[i][2])

    fig, ax = plt.subplots()
    ax.plot(G_mean)
    ax.fill_between(x, G_low, G_high, alpha=0.2)
    plt.xlabel('Number of iterations')
    plt.ylabel('Performance')
    plt.grid(True)
    plt.savefig('./fqi/figs/' + name)


if __name__ == "__main__":
    arg_help = "Please refers to the readme".format(sys.argv[0])
    try:
        env = sys.argv[1]
        mode = sys.argv[2]
    except:
        print(arg_help)
        sys.exit(2)

    if mode == 'train':
        early_stop = 500
        model_name = 'model'
        bufferSize = 20000
        capTime = 30
        try:
            Nbr_action = int(sys.argv[3])
            parser = argparse.ArgumentParser()
            parser.add_argument("env")
            parser.add_argument("mode")
            parser.add_argument("Nbr_action")
            parser.add_argument("-n", "--name", help="model name")
            parser.add_argument("--stop", help="early stop")
            parser.add_argument("--BufferSize", help="BufferSize")
            parser.add_argument("--CapTime", help="BufferSize")
            args = parser.parse_args()
            if args.name is not None:
                model_name = args.name
            if args.stop is not None:
                early_stop = int(args.stop)
            if args.BufferSize is not None:
                bufferSize = int(args.BufferSize)
            if args.CapTime is not None:
                capTime = int(args.CapTime)

        except:
            print(arg_help)
            sys.exit(2)

        fqi = FQI(env, 5, bufferSize)
        fqi.gen_traj(20)
        fqi.fitted_Q(1, 0)

        total_duration = 0
        G_array = []
        N = 1
        solved = False
        solve = 3
        while not solved:
            start_time = time.time()
            fqi.fitted_Q(N, 0)
            G = fqi.perfomance_measure(10, early_stop=early_stop)
            G_array.append(G)
            duration = time.time() - start_time
            total_duration += duration
            print('iteration:', N, '; mean_performance: ', G[2], '; duration:', str(round(duration, 2)) + "s",
                  '(time elapse :' + str(round(total_duration / 60, 2)) + ' min)')
            if G[2] >= early_stop:
                solve -= 1
            else:
                solve = 3
                solved = False
            if solve == 0 or total_duration / 60 > capTime:
                solved = True
            N += 1

        frames, _, score = fqi.play_episode(render=True)
        gif_path = env + "_" + model_name + '.gif'
        save_frames_as_gif(frames, path='./fqi/gifs/', filename=gif_path)
        fqi.env.close()

        fqi.save_model('./fqi/model/', env + "_" + str(Nbr_action) + "_" + model_name + '.pkl')
        print('model saved at', './fqi/model/', env + "_" + str(Nbr_action) + "_" + model_name + '.pkl')

        fig_name = env + model_name + 'learning_curve.pdf'
        plot_learning_curve(G_array, fig_name)
        print('learning curve saved at', './fqi/figs/', fig_name)

        data = pd.DataFrame(G_array, columns=['low', 'high', 'mean', 'J'])
        data.to_csv('./fqi/data/' + env + "_" + model_name + "_" + 'learning_curve.csv', index=False)
        print('training data saved at', './fqi/figs/', fig_name)

        print('End of training in ', N, 'iterations and ', round(total_duration / 60, 2), 'min')
        print('final performance: ', G_array[-1])

    elif mode == 'test':
        model_name = 'test'
        Nbr_action = 3
        try:
            parser = argparse.ArgumentParser()
            parser.add_argument("env")
            parser.add_argument("mode")
            parser.add_argument("Nbr_action")
            parser.add_argument("path")
            parser.add_argument("-n", "--name", help="model name")
            args = parser.parse_args()
            if args.name is not None:
                model_name = args.name
            if args.path is not None:
                model_path = args.path
            if args.Nbr_action is not None:
                Nbr_action = int(args.Nbr_action)
        except:
            print(arg_help)
            sys.exit(2)

        fqi = FQI(env, Nbr_action)
        fqi.load_model(model_path)
        frames, _, score = fqi.play_episode(render=True)
        gif_path = env + model_name + '.gif'
        save_frames_as_gif(frames, path='./fqi/gifs/', filename=gif_path)
        fqi.env.close()
        fqi.render(None)
        print("start testing")
        performance = fqi.perfomance_measure(50)
        print('Mean perfomance (step) of the model over 50 iteration : ')
        print('low_bound', performance[0])
        print('high_bound', performance[1])
        print('mean', performance[2])
        print('J of the model over 50 iteration : ', performance[3])
