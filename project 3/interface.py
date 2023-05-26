import math
import numpy as np
import pickle
import sys
import pandas as pd
import getopt
import os
import argparse
from matplotlib import pyplot as plt
from matplotlib import animation
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
import time
from utils import ReplayBuffer
import gymnasium as gym
from FQI import FQI
from REINFORCE_softmax import policy

if __name__ == "__main__":
    arg_help = "Please refers to the readme".format(sys.argv[0])

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("env")
        parser.add_argument("algo")
        parser.add_argument("Nbr_action")
        parser.add_argument("model_path")
        parser.add_argument("expected_reward")
        args = parser.parse_args()
        env_name = args.env
        algo = args.algo
        Nbr_action = int(args.Nbr_action)
        model_path = args.model_path

        if args.expected_reward == "True":
            expected_reward = True
        elif args.expected_reward == "False":
            expected_reward = False
        else:
            print(arg_help)
            sys.exit(2)

    except:
        print(arg_help)
        sys.exit(2)

    if algo == 'FQI':
        fqi = FQI(env_name, Nbr_action)
        fqi.load_model(model_path)
        fqi.render("human")
        state, _ = fqi.env.reset()
        step = 0
        G = 0
        while True:
            action = fqi.best_action(state)
            next_state, reward, terminated, truncated, _ = fqi.env.step([action])
            if terminated:
                reward = -100
            fqi.memory.add(state, action, reward, next_state, terminated)
            state = next_state
            G += reward
            step += 1
            if terminated or truncated:
                break
        print("Total reward: ", G)
        print("Total step: ", step)

    elif algo == 'softmax_simple':
        current_directory = os.getcwd()
        file_path = os.path.join(current_directory, "project 3", "theta_simple_pendulum_30000.npy")
        theta = np.load(file_path)
        
        if expected_reward:
            values = []
            for _ in range(50):
                env = gym.make("InvertedPendulum-v4", render_mode= None)
                s = env.reset()[0]
                
                step = 0
                G = 0
                while True: 
                    pi = policy(s, theta)
                    actions = np.linspace(-3, 3, 5)
                    action = [np.random.choice(actions , p=pi)]
                    next_s, r, terminated, truncated, info = env.step(action)
                    s = next_s
                    G += r
                    step += 1
                
                    if terminated or truncated:
                        break
                values.append(G)
            print(f"The expected reward is : {np.mean(values)}")
        else:
            env = gym.make("InvertedPendulum-v4", render_mode="human")
            s = env.reset()[0]
            step = 0
            G = 0
            while True: 
                pi = policy(s, theta)
                actions = np.linspace(-3, 3, 5)
                action = [np.random.choice(actions , p=pi)]
                next_s, r, terminated, truncated, info = env.step(action)
                s = next_s
                G += r
                step += 1
                if terminated or truncated:
                    break
            print("Total reward: ", G)
            print("Total step: ", step)

    elif algo == 'softmax_double':
        current_directory = os.getcwd()
        file_path = os.path.join(current_directory, "project 3", "theta_double_pendulum_100000.npy")
        theta = np.load(file_path)
        if expected_reward:
            values = []
            for _ in range(50):
                env = gym.make("InvertedDoublePendulum-v4", render_mode= None)
                s = env.reset()[0]
                step = 0
                G = 0
                while True: 
                    pi = policy(s, theta)
                    actions = np.linspace(-3, 3, 9)
                    action = [np.random.choice(actions , p=pi)]
                    next_s, r, terminated, truncated, info = env.step(action)
                    s = next_s
                    G += r
                    step += 1
                    if terminated or truncated:
                        break
                values.append(G)
            print(f"The expected reward is : {np.mean(values)}")
        else:
            env = gym.make("InvertedDoublePendulum-v4", render_mode="human")
            s = env.reset()[0]
            step = 0
            G = 0
            while True: 
                pi = policy(s, theta)
                actions = np.linspace(-3, 3, 9)
                action = [np.random.choice(actions , p=pi)]
                next_s, r, terminated, truncated, info = env.step(action)
                s = next_s
                G += r
                step += 1
                if terminated or truncated:
                    break
            print("Total reward: ", G)
            print("Total step: ", step)

    elif algo == 'DDPG':
        pass