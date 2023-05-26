import gymnasium as gym
import numpy as np
import time

def policy(state, theta):
    exp = np.exp(state.dot(theta)) 	
    return exp/np.sum(exp) 

def simple():
    env = gym.make("InvertedPendulum-v4", render_mode="human")
    theta = np.load("theta_simple_pendulum.npy")
    
    s = env.reset()[0]
    while True: 
        pi = policy(s, theta)
        actions = np.linspace(-3, 3, 5)
        action = [np.random.choice(actions , p=pi)]
        next_s, r, terminated, truncated, info = env.step(action)
        time.sleep(0.1)
        s = next_s
        if terminated or truncated:
            break
    env.close()

def double():
    env = gym.make("InvertedDoublePendulum-v4", render_mode="human")
    theta = np.load("theta_double_pendulum.npy")
    s = env.reset()[0]
    while True: 
        pi = policy(s, theta)
        actions = np.linspace(-3, 3, 9)
        action = [np.random.choice(actions , p=pi)]
        next_s, r, terminated, truncated, info = env.step(action)
        time.sleep(0.1)     
        s = next_s
        if terminated or truncated:
            break
    env.close()

def main():
    simple()
    double()
    
if __name__ == "__main__":
    main()   