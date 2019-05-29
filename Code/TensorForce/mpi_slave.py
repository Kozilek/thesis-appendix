import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'


# TODO : BENCHMARK THIS
# Why is this attaining a too high score
import sys
import io
#import os
text_trap = io.StringIO()
#sys.stdout = text_trap
#sys.stderr = text_trap

import tensorflow as tf


from mpi4py import MPI
from tensorforce.agents import PPOAgent
import numpy as np
from six.moves import xrange
import time
import tqdm
from Environment.tforce_env_full_state import PyBulletEnvironment as pb_physics
from Environment.tforce_env_no_phys import PyBulletEnvironment as pb_nophys
tf.logging.set_verbosity(tf.logging.ERROR)

'''
Slave process for the mpi_runner
'''

# Setup mpi communications
comm = MPI.Comm.Get_parent()# Param broadcast
process_id = comm.Get_rank()
nprocs = comm.Get_size()

# Receive parameters
params = comm.bcast(None, root = 0)
(agent_config, env_config, save_path, data_path, nbatches, deterministic, max_episode_timesteps, subdivisions, dist_mode, network_spec, batch_size, learning_rate) = params
should_stop = False
#print(save_path)

# Setup variables
repeat_actions = 1
episode_rewards = []
episode_timesteps = []
episode_times = []

# Receive episode allocation
batch_allocation = comm.scatter(None, root=0)
episode_allocation = batch_allocation * nbatches



# Initialize the physics server
if env_config['physics']:
    env = pb_physics(max_episode_timesteps, GUI = False, data_path = data_path, reward_func = env_config.get('reward', 'Phi3'))
else:
    env = pb_nophys(max_episode_timesteps, GUI = False, data_path = data_path)
print('Process {} succesfully started server.'.format(process_id))  

# Compute frame allocation
if dist_mode == 'segmented':
    nframes = env.nframes
    frame_segs = [int(x * ((nframes - min(max_episode_timesteps*2, nframes) - 1) / (nprocs+1))) for x in range(nprocs + 2)]
    start_frame = frame_segs[process_id]
    end_frame = frame_segs[process_id + 2]
    env.set_framebound(start_frame, end_frame)

'''
network_spec = [
    dict(type='dense', size=1024),
    dict(type='dense', size=512)
]

# Construct State Space
batch_size = 4096
'''
agent = PPOAgent(
    # Network Specification
    states=env.states,
    actions=env.actions,
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
        learning_rate=learning_rate,
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
#print('Agent created')
# Synchronize with master agent
agent.restore_model(directory=save_path)
#print('Agent restored')

# Synchronize
comm.Barrier()
#print('Yo')

episode = 0
data_buffer = [] 
if process_id == 0:
    pbar = tqdm.tqdm(total = nprocs * batch_allocation)
# Run this single worker (episode loop) as long as episode threshold have not been reached.
while not should_stop:
    state = env.reset()
    #print('Calling reset')
    agent.reset()
    #print('Reset resolved')
    episode_reward = 0

    # Time step (within episode) loop
    time_step = 0
    time_start = time.time()
    while True:
        #print('Calling act')
        action, internals, states = agent.act(states=state, deterministic=deterministic, buffered=False, independent = True)
        #print('Act resolved')
        reward = 0
        for repeat in xrange(repeat_actions):
            state, terminal, step_reward = env.execute(action=action)
            reward += step_reward
            if terminal:
                break

        time_step += 1
        episode_reward += reward
        data_buffer.append((state, action, internals, reward, terminal))

        if terminal or time_step == max_episode_timesteps:
            break


        # Abort the episode (discard its results) when global says so.
        # TODO : This may be too frequent polling
        #self.should_stop |= connections['should_stop'][1].poll()
        if should_stop:
            break
    if should_stop:
        break

    if process_id == 0:
        pbar.update(nprocs)
    ### TODO : This should be part of the all-to-one reduce action
    episode_rewards.append(episode_reward)
    #print(episode_reward)
    episode_timesteps.append(time_step)
    episode_times.append(time.time() - time_start)

    

    episode += 1
    # Determine if agent should be updated
    #if episode > batch_allocation:
    #    print('Process {} goofed up!'.format(process_id))
    if (int(episode % (batch_allocation / subdivisions)) == 0) or episode % batch_allocation == 0:
        # Reduce data
        # Just use the reduce method
        #print('Process {} approaching gather barrier.'.format(process_id))
        comm.Barrier()
        if process_id == 0 and episode % batch_allocation == 0:
            pbar.close()
        tmp1 = comm.gather(data_buffer, root=0)
        # Clear data buffer
        data_buffer = []
        if episode % batch_allocation == 0:
            # Recieve new model
            # Smart way: Broadcast model weights
            # Easy way: Restore from file
            #print('Process', self.process_id, 'reporting in!')
            #print('Sub-process observes before load: ',agent.episode)
            comm.Barrier()
            #print('Process {} restoring model.'.format(process_id))
            agent.restore_model(directory=save_path)
            agent.reset()
            #print('Sub-process observes after load : ',agent.episode)
            if process_id == 0:
                if episode < episode_allocation:
                    pbar = tqdm.tqdm(total = nprocs * batch_allocation)


    if episode >= episode_allocation:
        should_stop = True