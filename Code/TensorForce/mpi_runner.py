# Copyright 2017 reinforce.io. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

'''
This code features heavy modifications from the threaded_runner.py code provided from the tensorforce github, which is available under the Apache 2.0 license.
Support has been removed from several decrapated functions.
'''

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import importlib
from inspect import getargspec
from six.moves import xrange
import time
import warnings
import itertools
import numpy as np
from tqdm import tqdm
from mpi4py import MPI
import sys

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorforce import TensorForceError
from tensorforce.execution.base_runner import BaseRunner
from tensorforce.agents.learning_agent import LearningAgent
from tensorforce.agents import agents as AgentsDictionary
from tensorforce.agents import PPOAgent

class MPIRunner(BaseRunner):
    """
    Runner multiprocess execution of multiple agents.
    """

    def __init__(self, agent, environment, batch_size, nprocs, agent_config, repeat_actions=1, save_path=None, save_episodes=None, save_frequency=None,
                 save_frequency_unit=None, data_path = 'Trial01.npy'):
        """
        Initialize a ThreadedRunner object.

        Args:
            save_path (str): Path where to save the shared model.
            save_episodes (int): Deprecated: Every how many (global) episodes do we save the shared model?
            save_frequency (int): The frequency with which to save the model (could be sec, steps, or episodes).
            save_frequency_unit (str): "s" (sec), "t" (timesteps), "e" (episodes)
            agents (List[Agent]): Deprecated: List of Agent objects. Use `agent`, instead.
            environment (List[Environment]): List of Environment objects.
        """
        super(MPIRunner, self).__init__(agent, environment, repeat_actions)

        #if len(agent) != len(environment):
        #    raise TensorForceError("Each agent must have its own environment. Got {a} agents and {e} environments.".
        #                           format(a=len(self.agent), e=len(self.environment)))

        self.save_path = save_path
        self.save_episodes = save_episodes
        if self.save_episodes is not None:
            warnings.warn("WARNING: `save_episodes` parameter is deprecated, use `save_frequency` AND "
                          "`save_frequency_unit` instead.",
                          category=DeprecationWarning)
            self.save_frequency = self.save_episodes
            self.save_frequency_unit = "e"
        else:
            self.save_frequency = save_frequency
            self.save_frequency_unit = save_frequency_unit

        # Stop-condition flag that each worker abides to (aborts if True).
        self.should_stop = False

        # Global time counter (sec).
        self.time = None

        # Some Deepmimic Specifics
        self.batch_size = batch_size
        self.nprocs = nprocs
        self.data_path = data_path
        self.agent_config = agent_config
        self.episode_count = 0

    def close(self):
        self.agent.close()  # only close first agent as we just have one shared model
        #for e in self.environment:
        #    e.close()
        self.environment.close()

    def run(
        self,
        num_episodes=-1,
        max_episode_timesteps=-1,
        episode_finished=None,
        summary_report=None,
        summary_interval=0,
        num_timesteps=None,
        deterministic=False,
        episodes=None,
        max_timesteps=None,
        testing=False,
        sleep=None,
        physics = True,
        subdivisions = 4,
        dist_mode = 'random'
    ):
        """
        Executes this runner by starting all Agents in parallel (each one in one process).

        Args:
            episodes (int): Deprecated; see num_episodes.
            max_timesteps (int): Deprecated; see max_episode_timesteps.
        """

        # Renamed episodes into num_episodes to match BaseRunner's signature (fully backw. compatible).
        if episodes is not None:
            num_episodes = episodes
            warnings.warn("WARNING: `episodes` parameter is deprecated, use `num_episodes` instead.",
                          category=DeprecationWarning)
        assert isinstance(num_episodes, int)
        # Renamed max_timesteps into max_episode_timesteps to match single Runner's signature (fully backw. compatible).
        if max_timesteps is not None:
            max_episode_timesteps = max_timesteps
            warnings.warn("WARNING: `max_timesteps` parameter is deprecated, use `max_episode_timesteps` instead.",
                          category=DeprecationWarning)
        assert isinstance(max_episode_timesteps, int)

        if self.nprocs < 2:
            warnings.warn("WARNING: `nprocs` parameter is less than two. Use a different runner.",
                          category=DeprecationWarning)

        assert self.batch_size % subdivisions == 0

        self.reset()
        #print(self.save_path)

        # figure out whether we are using the deprecated way of "episode_finished" reporting
        old_episode_finished = False
        if episode_finished is not None and len(getargspec(episode_finished).args) == 1:
            old_episode_finished = True

        # Reset counts/stop-condition for this run.
        self.global_episode = 0
        self.global_timestep = 0
        self.should_stop = False
        self.env_config = {'physics':physics, 'reward':self.environment.reward_func}

        # TODO : This
        # Create episode allocations
        nbatches = int(num_episodes / self.batch_size)
        allocations = [int(self.batch_size / self.nprocs) for i in range(self.nprocs)]
        for i in range(self.batch_size % self.nprocs):
            allocations[i] += 1
        #print('Allocs, alloc sum, batch size:', allocations, np.sum(allocations), self.batch_size)
        #sys.stdout.flush()
        #time.sleep(50)

        # Save current model
        self.agent.save_model(directory = self.save_path, append_timestep = False)

        # Define shared parameters
        params = (              self.agent_config,              # Agent parameters
                                self.env_config ,               # Environment parameters
                                '/'.join(self.save_path.split('/')[:-1]) + '/',                 # save_dir
                                self.data_path,                 # data_path = 'Trial01.npy',
                                nbatches,
                                deterministic,                  # deterministic=False,
                                max_episode_timesteps,          # max_episode_timesteps=-1,
                                subdivisions,
                                dist_mode,
                                self.agent.network,
                                self.agent.update_mode['batch_size'],
                                self.agent.optimizer['optimizer']['optimizer']['learning_rate']
                                ) 

        # Start processes
        comm = MPI.COMM_SELF.Spawn(sys.executable, args = ['mpi_slave.py'], maxprocs = self.nprocs)

        # Broadcast shared parameters
        comm.bcast(params[:], root = MPI.ROOT)

        # Scatter allocations
        comm.scatter(allocations[:], root = MPI.ROOT)

        # Synchronize
        comm.Barrier()

        # Stay idle until killed by SIGINT or a global stop condition is met.
        try:
            #batch_count = 0
            # TODO: Increase observation frequency
            for batch_count in range(nbatches):
                time_start = time.time()
                ep_count = 0
                reward_log = []
                #with tqdm(total = self.batch_size) as bbar:
                for subdiv_count in range(subdivisions):
                    self.agent.reset()
                    data = []
                    # Wait for processes to finish episodes (Not necessary)
                    #print('Joining reduce barrier\n--------')
                    comm.Barrier()
                    #print('Releasing reduce barrier\n')

                    ### TODO: Recieve data and observe
                    #print('Now Gathering')
                    #sys.stdout.flush()
                    data = comm.gather(None, root=MPI.ROOT)
                    data =  [y for x in data for y in x]
                    #print('Finished Gathering')
                    #sys.stdout.flush()
                    reward_count = 0 
                    #print(data[0])
                    timestep_count = 0
                    for (state, actions, internals, reward, terminal) in data:
                        timestep_count += 1
                        reward_count += reward
                        reward_log.append(reward)
                        if terminal:
                            #self.agent.reset()
                            self.episode_rewards.append(reward_count)
                            self.episode_timesteps.append(timestep_count)
                            self.episode_times.append(float('NaN'))
                            self.episode_count += 1
                            ep_count += 1
                            timestep_count = 0
                            reward_count = 0
                            #print('Observed episode')
                            # Call episode_finished
                            '''
                            if old_episode_finished:
                                pass
                            elif not episode_finished(self, thread_id):
                                break
                            '''
                    data = np.transpose(data)
                    #print('Finished Processing')
                    #sys.stdout.flush()
                    #bbar.update(int(self.batch_size / subdivisions))
                    sys.stdout.flush()  
                    #print('Now Observing')
                    #sys.stdout.flush()
                    if(ep_count == self.batch_size):
                        print('\nTraining Neural Network.')
                        #print(data[4])
                        sys.stdout.flush()
                    self.agent.atomic_observe(
                        states=tuple(data[0]),
                        actions=tuple(data[1]),
                        internals=tuple(data[2]),
                        reward=tuple(data[3]),
                        terminal=tuple(data[4])
                    )
                        #print('Finished Observing, {}:{}'.format(ep_count, self.agent.episode))
                        #sys.stdout.flush()
                assert self.agent.episode % self.batch_size == 0, 'Insufficient episodes collected for batch. {} episodes registered, where multiple of {} is needed.'.format(self.agent.episode, self.batch_size)
                self.agent.reset()
                #pbar.update(1)
                sys.stdout.flush()
                batch_time = time.time() - time_start
                print('Batch {} finished with average episode reward fraction {} over an average {:.2f} frames after {:.2f}s.'.format(int(self.agent.episode / self.batch_size),
                    np.mean(np.array(self.episode_rewards[-self.batch_size:])/max_episode_timesteps),
                    np.mean(np.array(self.episode_timesteps[-self.batch_size:])), 
                    batch_time))
                sys.stdout.flush()

                self.global_episode = self.agent.episode
                #print('Runner observes:', self.agent.episode)
                sys.stdout.flush()

                ### TODO: Save
                #print('Now Saving')
                #sys.stdout.flush()
                self.agent.save_model(directory = self.save_path, append_timestep = False)
                #print('Finished Saving')
                #time.sleep(10)
                # Join and release barrier to let processes load new model state
                #print('Joining broadcast barrier\n--------')
                comm.Barrier()
                #sys.stdout.flush()
                #print('Releasing broadcast barrier\n')
                self.episode_times[-self.batch_size:] = [batch_time / self.batch_size]*self.batch_size


        except KeyboardInterrupt:
            print('Keyboard interrupt, sending stop command to threads')
            #MPI_WORLD.Abort()

        # Raise termination flag for all processes (tag 11)
        # Slaves should terminate themselves
        '''
        for proc_id in range(nprocs):
            handle = comm.isend(1, dest=p, tag=11)
            handle.wait()        
        '''

        print('All processes recieved kill signal')

    # Backwards compatibility for deprecated properties (in case someone directly references these).
    @property
    def agents(self):
        return self.agent

    @property
    def environments(self):
        return self.environment

    @property
    def episode_lengths(self):
        return self.episode_timesteps

    @property
    def global_step(self):
        return self.global_timestep