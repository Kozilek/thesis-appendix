import numpy as np
import os

from tensorforce.agents import PPOAgent
from mpi_runner import MPIRunner
from Environment.tforce_env_full_state import PyBulletEnvironment
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

RESUME = False

environment = PyBulletEnvironment(300, False, reward_func = 'Phi4')

network_spec = [dict(type='dense', size=1024)] + ([dict(type='dense', size=512)])

batch_size = 1024#4096 #4*256

agent = PPOAgent(
    # Network Specification
    states=environment.states,
    actions=environment.actions,
    network=network_spec,
    # Batching
    update_mode=dict(
        unit='episodes',
        batch_size=batch_size#,
        #frequency=10
    ),
    # Memory Model
    memory=dict(
        type='latest',
        include_next_states=False,
        capacity=500000
    ),
    # Policy Gradient model (Consider swapping for network)
    baseline_mode='states',
    baseline=dict(
        type='mlp',
        sizes=[layer['size'] for layer in network_spec]
    ),
    baseline_optimizer=dict(
        type='multi_step',
        optimizer=dict(
            type='momentum',
            learning_rate=1e-2,
            momentum=0.9
        ),
        num_steps=10
    ),
    gae_lambda=0.95,
    
    # PPOAgent
    likelihood_ratio_clipping=0.2,
    step_optimizer=dict(
        type='momentum',
        learning_rate = 0.05 * 1e-5,
        momentum=0.9
    ),
    subsampling_fraction=0.0625,
    discount=0.95,
    optimization_steps=50,
    execution=dict(
        type='single',
        session_config=None,
        distributed_spec=None
    )
)

# Create storage directory
affix = 'test' #'convexout_0p5_f0_b4096_d2_phi4_m20_lr5e-4'
path = os.path.dirname('./Saved/{}/'.format(affix))
#print(path)
#print('./Saved/{}'.format(affix))
path_best = os.path.dirname('./Saved/{}/best/'.format(affix))
if not os.path.exists(path):
    os.makedirs(path)
if not os.path.exists(path_best):
    os.makedirs(path_best)    

if RESUME:
    agent.restore_model(directory='./Saved/{}/'.format(affix))

# Create the runner
runner = MPIRunner(agent=agent, environment=environment, batch_size = batch_size, nprocs = 4, agent_config = None, save_path = './Saved/{}/model'.format(affix))
runner.max_reward = 0

start_ep = agent.episode

# Callback function printing episode statistics
def episode_finished(r):
    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                                 reward=r.episode_rewards[-1]))
    if r.episode % batch_size == 0 and r.episode != 0:
        mean = np.mean(r.episode_rewards[-batch_size:])
        if (mean >r.max_reward):
        #    path = agent.save_model(directory='./Saved/{}/model_best_'.format(affix))
        #    print('Saved file to ', path)
            r.max_reward = mean
        print('Mean reward over previous {} runs: {}. Current best mean reward: {}'.format(batch_size, mean, r.max_reward))
    #if r.episode % batch_size == 0 and r.episode != 0:
        #path = agent.save_model(directory='./Saved/{}/model_current_'.format(affix))
        #print('Saved file to ', path)
    return True

# Start learning
max_ep = 1
runner.run(num_episodes=batch_size * 300, max_episode_timesteps=max_ep, episode_finished=None, physics = True,subdivisions = 1, dist_mode='segmented')

# Print statistics
print("Learning finished. Total episodes: {ep}. Average reward of last batch: {ar}.".format(
    ep=len(runner.episode_rewards),
    ar=np.mean(np.array(runner.episode_rewards[-batch_size:]) / max_ep))
)
#print(runner.episode_timesteps)
np.save('reward_{}_{}-{}.npy'.format(affix, start_ep, agent.episode),np.array(runner.episode_rewards) / max_ep)
np.save('supplemental_{}_{}_{}.npy'.format(affix, start_ep, agent.episode), np.array([np.array(runner.episode_timesteps),np.array(runner.episode_times)]))

runner.close()
