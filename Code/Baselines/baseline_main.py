import time
import gym
import numpy as np
import os

import tensorflow as tf
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.bench import Monitor
from stable_baselines.common import set_global_seeds
from stable_baselines.ppo2 import PPO2
from Environment.baseline_pbenv import PyBulletEnvironment

# Define metaparameters
nprocs = 4
batch_size = 256 // 8
nbatches = 4
data_path = 'Trial01.npy'
max_steps =  15
GUI = False
reward_func = 'Phi4'
nsteps = (max_steps * batch_size) // nprocs
affix = 'test'
timer = time.time()
nframes = 50

# Track Progress
best_mean_reward, step_count = -np.inf, 0
log_dir = "./tmp/{}/".format(affix)
os.makedirs(log_dir, exist_ok=True)
for rank in range(nprocs):
	os.makedirs(log_dir + 'log_{}/'.format(rank), exist_ok=True)

def callback(_locals, _globals):
	"""
	Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
	:param _locals: (dict)
	:param _globals: (dict)
	"""
	global step_count, best_mean_reward, timer
	# Print stats every 1000 calls
	if (step_count + 1) % 1 == 0:
		# Evaluate policy performance
		x = np.array([])
		y = np.array([])
		for rank in range(nprocs):
			_x, _y = ts2xy(load_results(log_dir + 'log_{}/'.format(rank)), 'timesteps')
			#print(_x, _y)
			#print(type(_x))
			#print(_x.shape)
			x = np.append(x, _x[-batch_size//nprocs:])
			y = np.append(y, _y[-batch_size//nprocs])
		if len(x) > 0:
			mean_reward = np.mean(y[-batch_size:])

			# New best model, you could save the agent here
			if mean_reward > best_mean_reward:
				best_mean_reward = mean_reward
				# Example for saving best model
				print("Saving new best model")
				_locals['self'].save(log_dir + 'best_model.pkl')
			print(x[-1], 'timesteps')
			print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f} - Time since last call: {:.2f}\n".format(best_mean_reward, mean_reward, time.time() - timer))
			timer = time.time()
	step_count += 1
	return True

# Environment Helper Function
def make_env(rank, framerange = None, seed=0):
	def _init():
		env = PyBulletEnvironment(max_steps, False, data_path = data_path, reward_func = reward_func)
		env.seed(seed + rank)
		if framerange != None:
			env.set_framebound(framerange[0],framerange[1])
		env = Monitor(env, log_dir + 'log_{}/'.format(rank), allow_early_resets=True)
		return env
	set_global_seeds(seed)
	return _init

# Make Environments
frame_segs = [int(x * ((nframes - min(max_steps*2, nframes) - 1) / (nprocs+1))) for x in range(nprocs + 2)]
envs = [make_env(rank = i, seed = 0, framerange=(frame_segs[i], frame_segs[i + 2])) for i in range(nprocs)]
env = SubprocVecEnv(envs)

# Create Networks
policy_kwargs = dict(act_fun = tf.nn.relu, net_arch=[1024, 512])

# Create Training Agent
agent = PPO2(MlpPolicy, env, 
				gamma = 0.95, 
				lam = 0.95,
				n_steps = nsteps, 
				verbose=0,
				policy_kwargs=policy_kwargs,
				cliprange=0.2,
				learning_rate= 5 * 1e-5,
				nminibatches=16
				)

# Start Learning
agent.learn(nbatches * max_steps * batch_size, callback = callback)

# Save File
agent.save(affix)