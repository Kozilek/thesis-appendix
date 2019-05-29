import time
import gym
import numpy as np
import os
import argparse

#import tensorflow as tf
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.bench import Monitor
from stable_baselines.common import set_global_seeds
from stable_baselines.ppo2 import PPO2
from Environment.baseline_pbenv import PyBulletEnvironment

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-trial', help='Trial Path', type=str, default='Trial01.npy')
parser.add_argument('-f', help='Agent Path', type=str)
parser.add_argument('-n', help='Number of Steps', type=int)
parser.add_argument('-GUI', help='Use GUI', type=int, default=1)
parser.add_argument('-repeats', help='Number of Repeats', type=int, default = 1)
parser.add_argument('-sleep_time', help='Time slept between steps', type=float, default = 0.0)
parser.add_argument('-start_frame', help='Frame to start from', type=int, default = 0)


args = parser.parse_args()

# Define metaparameters
data_path = args.trial
max_steps =  args.n
GUI = args.GUI > 0
reward_func = 'Phi3'
model_path = args.f
repeats = args.repeats
sleep_time = args.sleep_time
start_frame = args.start_frame

# Environment Helper Function
def make_env(rank, framerange = None, seed=0):
	def _init():
		env = PyBulletEnvironment(max_steps, GUI, data_path = data_path, reward_func = reward_func)
		env.seed(seed + rank)
		if framerange != None:
			env.set_framebound(framerange[0],framerange[1])
		#env = Monitor(env, log_dir + 'log_{}/'.format(rank), allow_early_resets=True)
		return env
	set_global_seeds(seed)
	return _init

# Make Environments
envs = [make_env(rank = 0, seed = 0, framerange = (start_frame, start_frame))]
env = SubprocVecEnv(envs)
'''
env = PyBulletEnvironment(max_steps, GUI, data_path = data_path, reward_func = reward_func)
env = DummyVecEnv(env)
'''

# Load Training Agent
agent = PPO2.load(model_path, env = env)


# Run Agent on Environment
reward_log = []
for i in range(repeats):
	state = env.reset()
	deterministic = False
	sum_reward = 0
	for _ in range(max_steps):
		action, _ = agent.predict(state, deterministic = deterministic)
		#print(action)
		state, reward, done, info = env.step(action)
		time.sleep(sleep_time)
		sum_reward += reward[0]
	reward_log.append(sum_reward / max_steps)
print('Reward Fraction Achieved',np.mean(reward_log))



