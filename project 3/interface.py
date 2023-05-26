import numpy as np
import sys
import os
import argparse
from tqdm import tqdm
import gymnasium as gym
from FQI import FQI
from REINFORCE_softmax import policy
from DDPG import DDPG

if __name__ == "__main__":
    arg_help = "Please refers to the readme".format(sys.argv[0])
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("env")
        parser.add_argument("algo")
        parser.add_argument("arg3")
        parser.add_argument("arg4")
        parser.add_argument("arg5")
        args = parser.parse_args()
        env_name = args.env
        algo = args.algo
    except:
        print(arg_help)
        sys.exit(2)

    if algo == 'FQI':
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

        fqi = FQI(env_name, Nbr_action)
        fqi.load_model(model_path)
        fqi.render("human")
        nbr_episode = 1
        if expected_reward:
            nbr_episode = 50
        values = []
        for _ in tqdm(range(nbr_episode)):
            state, _ = fqi.env.reset()
            step = 0
            G = 0
            while True:
                action = fqi.best_action(state)
                next_state, reward, terminated, truncated, _ = fqi.env.step([action])
                fqi.memory.add(state, action, reward, next_state, terminated)
                state = next_state
                G = reward + 0.99 * G
                step += 1
                if terminated or truncated:
                    break
            values.append(G)
        print(f"The expected reward is : {np.mean(values)}")

    elif algo == 'softmax_simple':
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
        theta = np.load("./theta_simple_pendulum_30000.npy")
        
        if expected_reward:
            values = []
            for _ in range(50):
                env = gym.make("InvertedPendulum-v4", render_mode=None)
                s = env.reset()[0]
                
                step = 0
                G = 0
                while True: 
                    pi = policy(s, theta)
                    actions = np.linspace(-3, 3, 5)
                    action = [np.random.choice(actions , p=pi)]
                    next_s, r, terminated, truncated, info = env.step(action)
                    s = next_s
                    G = r + 0.99 * G
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
        current_directory = os.getcwd()
        theta = np.load("./theta_double_pendulum_100000.npy")
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
        try:
            parser = argparse.ArgumentParser()
            parser.add_argument("env")
            parser.add_argument("algo")
            parser.add_argument("Actor_path")
            parser.add_argument("Critics_path")
            parser.add_argument("expected_reward")
            args = parser.parse_args()
            env_name = args.env
            algo = args.algo
            actor_path = args.Actor_path
            critic_path = args.Critics_path

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

        ddpg = DDPG(alpha_actor=0.1, alpha_critic=0.1, env_name=env_name)
        ddpg.load_models(actor_path, critic_path)
        ddpg.render("human")
        nbr_episode = 1
        if expected_reward:
            nbr_episode = 50
        values = []
        for _ in tqdm(range(nbr_episode)):
            state, _ = ddpg.env.reset()
            step = 0
            G = 0
            while True:
                action = ddpg.choose_action(state)
                next_state, reward, terminated, truncated, _ = ddpg.env.step([action])
                state = next_state
                G = reward + 0.99 * G
                step += 1
                if terminated or truncated:
                    break
            values.append(G)
        print(f"The expected reward is : {np.mean(values)}")