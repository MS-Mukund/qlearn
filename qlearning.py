
'''importing modules'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import sys
from time import sleep
sys.path.append('/home/mekasaimukund/miniconda3/envs/nn/lib/python3.10/site-packages')

import gym
import pygame
from collections import defaultdict

# params
eps = 0.1
alpha = 0.1
gamma = 0.9
epoch = 500

pygame.init()
print('Imported modules')

# environment setup
env = gym.make('Taxi-v3')
env.reset()
env.render()

Q_table = np.zeros( (env.observation_space.n, env.action_space.n) )
frames = list()

def render_and_exit(frames, e):
    for frame in frames:
        print(frame)

    print('Final Q-table: {}'.format(Q_table))
    print('Time: {}'.format(e))

    exit()


def take_action(eps, state):
# epsilon greedy policy
    if random.random() < eps:
        return env.action_space.sample()
    else:
        return np.argmax(Q_table[state])
        

def update_Q_table(alpha, gamma, state, action, reward, next_state):
    Q_table[state, action] = Q_table[state, action] + alpha*( reward + gamma *np.max(Q_table[next_state, :]) - Q_table[state, action] ) 

cur_state = env.reset()

for e in range(epoch):
    action = take_action(eps, cur_state)
    
    next_state, reward, goal, info = env.step(action)
    print(next_state, reward, goal, info)

    # frames.append({
    #     'frame': env.render(mode='ansi'),
    #     'episode': str(e),
    #     'state': cur_state,
    #     'action': action,
    #     'reward': reward
    #     }
    # )

    if goal == True:
        render_and_exit(frames, e)

    update_Q_table(alpha, gamma, cur_state, action, reward, next_state)
    cur_state = next_state

print(Q_table)
