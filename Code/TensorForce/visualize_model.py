import numpy as np
import os

from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner
from Environment.tforce_env_full_state import PyBulletEnvironment

from tqdm import tqdm

RENDER = True

environment = PyBulletEnvironment(300, RENDER, reward_func='Phi4')
environment.set_framebound(0,0)

network_spec = [
    dict(type='dense', size=1024),
    dict(type='dense', size=512),
    dict(type='dense', size=512)
]

batch_size = 4096

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
            type='adam',
            learning_rate=1e-3
        ),
        num_steps=10
    ),
    gae_lambda=0.95,
    
    # PPOAgent
    likelihood_ratio_clipping=0.2,
    step_optimizer=dict(
        type='adam',
        learning_rate=2.5 * 1e-3
    ),
    subsampling_fraction=0.0625,
    optimization_steps=50,
    execution=dict(
        type='single',
        session_config=None,
        distributed_spec=None
    )
)

agent.restore_model(directory='./load/')
print('Model restored')
#print(agent.act(states=np.zeros(environment.states['shape']), deterministic=False, buffered=False, independent = True))

# TODO : Create my own runner with multithreading
# Create the runner
runner = Runner(agent=agent, environment=environment)

# Load latest checkpoint
if RENDER:
    input('Ready to start.')
    runner.run(episodes=1, max_episode_timesteps=20, deterministic = False)
    print('Score achieved : {}'.format(np.array(runner.episode_rewards[-1]) / np.array(runner.episode_timesteps[-1])))

runner.close()
