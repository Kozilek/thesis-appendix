#import tensorflow as tf
#import sys
import numpy as np
#import argparse
import pybullet as pb
#import pybullet_data
import time
from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner
from pyquaternion import Quaternion
#from tqdm import tqdm

class Robot:
	pass

class PyBulletEnvironment():
	'''
	Class implementing pybullet as an environment for tensorforce.
	'''

	def __init__(self, n_steps, GUI = False, data_path = 'Trial01.npy'):
		# Manage step progress
		assert(n_steps > 0)
		self.n_steps = n_steps
		self.cur_step = 0
		self.frames_standing = 0
		self.frames_not_standing = 0

		# Connect to a physics server
		self.GUI = GUI
		if GUI :
			self.pybullet_client_ID = pb.connect(pb.GUI)
		else :
			self.pybullet_client_ID = pb.connect(pb.DIRECT)
		

		# Instanciate and configure physics objects 
		self.robot = Robot()
		self.robot.initial_z = None
		
		self.motor_names = None
		self.motor_power = None
		self.motor_limit = None
		self.revolute_joints = None
		self.label2index = dict()

		# Load pre-processed data
		self.data = np.load(data_path)		
		self.preset()

		# Construct State Space
		self._states = dict()
		self._states['type'] = 'float'
		self._states['shape'] = 1 + 13*(3 + 4 + 3 + 3)

		# Construct Action Space
		self._actions = dict()
		self._actions['type'] = 'float'
		self._actions['shape'] = self.revolute_joints
		self._actions['max_value'] = 1
		self._actions['min_value'] = 0

	def __str__(self):
		return 'PyBullet Physics Session {}'.format(self.pybullet_client_ID)

	def seed(self, seed):
		return None

	def close(self):
		# Disconnect and terminate physics client
		pb.disconnect(self.pybullet_client_ID)

	def preset(self):
		pb.resetSimulation(physicsClientId = self.pybullet_client_ID)
		pb.setGravity(0,0,-9.8, physicsClientId = self.pybullet_client_ID)
		self.tau = 1.0/60
		pb.setPhysicsEngineParameter(fixedTimeStep=self.tau, numSolverIterations=5, numSubSteps=2, physicsClientId = self.pybullet_client_ID)

		# Load objects
		pb.loadURDF("plane.urdf", physicsClientId = self.pybullet_client_ID)
		self.robotId = pb.loadURDF("humanoid.xml",flags = pb.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS, physicsClientId = self.pybullet_client_ID)
		
		# Initialize robot
		# TODO ; Choose init_frame randomly
		self.init_frame = 0
		wxyz = self.data[self.init_frame]['qwxyz_root']
		pb.resetBasePositionAndOrientation(self.robotId, self.data[self.init_frame]['root_pos'], wxyz[[1,2,3,0]], physicsClientId = self.pybullet_client_ID)

		# Extract model information
		self.motor_names = []
		self.motor_power = []
		self.motor_limit = []
		self.revolute_joints = 0
		self.end_effectors = ['link_right_ankle', 'link_left_ankle', 'link_right_wrist', 'link_left_wrist']
		self.reward_links = ['link_chest',
							'link_left_ankle',
							'link_right_ankle',
							'link_right_knee',
							'link_left_knee',
							'link_neck',
							'link_right_elbow',
							'link_left_elbow',
							'link_left_hip',
							'link_right_hip',
							'link_left_shoulder',
							'link_right_shoulder']
		self.label2index = dict()

		for joint_index in range(pb.getNumJoints(self.robotId, physicsClientId = self.pybullet_client_ID)):
			joint_data = pb.getJointInfo(self.robotId, joint_index, physicsClientId = self.pybullet_client_ID)
			link_name = joint_data[12].decode("ascii")
			self.label2index[link_name] = joint_index

			if joint_data[2] != pb.JOINT_REVOLUTE: 
				continue

			self.revolute_joints += 1
			self.motor_names.append(link_name)
			lower_lim, upper_lim = (joint_data[8], joint_data[9])
			self.motor_limit.append((lower_lim, upper_lim))
			self.motor_power.append(joint_data[10])		
		
		self.link_masses = {'root':pb.getDynamicsInfo(self.robotId, -1)[0]}
		for link_label in self.reward_links:
			self.link_masses[link_label] = pb.getDynamicsInfo(self.robotId, self.label2index[link_label])[0]

		self.nframes = len(self.data)
		self.motors = [self.label2index[label] for label in self.motor_names]

		# Gather and return current state
		#state = self.collect_observations()
		#return state

	def reset(self):
		# Set initial frame
		self.init_frame = np.random.randint(0, self.nframes - min(10, self.nframes))


		# Set robot position
		wxyz = self.data[self.init_frame]['qwxyz_root']
		pb.resetBasePositionAndOrientation(self.robotId, self.data[self.init_frame]['root_pos'], wxyz[[1,2,3,0]], physicsClientId = self.pybullet_client_ID)

		# Set joint values to init_frame values
		nframes = self.nframes
		for joint_index in range(pb.getNumJoints(self.robotId, physicsClientId = self.pybullet_client_ID)):
			joint_data = pb.getJointInfo(self.robotId, joint_index, physicsClientId = self.pybullet_client_ID)
			if joint_data[2] == pb.JOINT_FIXED:
				continue
			joint_name = joint_data[12].decode("ascii")
			value = self.data[self.init_frame]['value_' + joint_name]
			# TODO: May benefit from being extracted in preprocessing
			finite_diff = (self.data[min(self.init_frame + 1, nframes -1)]['value_' + joint_name] - self.data[max(self.init_frame - 1, 0)]['value_' + joint_name]) / (2 * self.tau)
			if self.init_frame + 1 > nframes or self.init_frame - 1 < 0:
				finite_diff = finite_diff * 2
			# Introduce random deviation
			#value = value + 0.01 * (np.random.rand()-0.5)
			pb.resetJointState(self.robotId, joint_index, value, targetVelocity = finite_diff, physicsClientId = self.pybullet_client_ID) 
			#print(finite_diff, pb.getJointState(self.robotId, joint_index)[1])
		
		# Bookkeeping
		self.robot.initial_z = None
		self.cur_step = 0
		self.frames_standing = 0
		self.frames_not_standing = 0
		self.Early_Termination = False
		self.link_states = dict()
		self.prev_jstate = None

		# Record and return state
		state = self.collect_observations()
		return state


	@property
	def states(self): 
		return self._states
	
	@property
	def actions(self):
		return self._actions

	def execute(self, action):
		# Apply motor forces
		target_positions = [self.motor_limit[i][0] + action[i] * (self.motor_limit[i][1] - self.motor_limit[i][0]) for i in range(len(action))]
		#pb.setJointMotorControlArray(self.robotId, self.motors,controlMode=pb.POSITION_CONTROL, targetPositions = target_positions, forces=np.array(self.motor_power), physicsClientId = self.pybullet_client_ID)
		offset = 0
		for joint_index in range(pb.getNumJoints(self.robotId, physicsClientId = self.pybullet_client_ID)):
			joint_data = pb.getJointInfo(self.robotId, joint_index, physicsClientId = self.pybullet_client_ID)
			if joint_data[2] == pb.JOINT_FIXED:
				offset += 1
				continue
			pb.resetJointState(self.robotId, joint_index, target_positions[joint_index - offset], physicsClientId = self.pybullet_client_ID)

		# Step simulation twice
		# TODO : This is now the bottleneck
		#pb.stepSimulation(physicsClientId = self.pybullet_client_ID)
		#pb.stepSimulation(physicsClientId = self.pybullet_client_ID)
		self.cur_step += 2

		if self.GUI:
			time.sleep(0.01)
			distance=5
			yaw = 0
			humanPos, humanOrn = pb.getBasePositionAndOrientation(self.robotId, physicsClientId = self.pybullet_client_ID)
			pb.resetDebugVisualizerCamera(distance,yaw,-20,humanPos, physicsClientId = self.pybullet_client_ID);

		# Observe state
		state = self.collect_observations()

		# Compute reward
		# Compute imitation reward
		weight_imitation = 1.0
		reward_imitation = 0

		# Determine current motion data frame
		work_frame = self.init_frame + self.cur_step
		frame_data = self.data[work_frame] #self.data[self.init_frame]# 

		# Quaternion Difference
		angle_sum = 0
		for (label_index, label) in enumerate(self.reward_links):
			jointState = pb.getJointState(self.robotId, label_index, physicsClientId = self.pybullet_client_ID)
			target_qwxyz = Quaternion(frame_data['qwxyz_'+label])
			current_qwxyz = Quaternion(self.prev_jstate[4 + label_index*7:4 + label_index*7 +4])
			#print(label, Quaternion.absolute_distance(target_qwxyz, current_qwxyz), '\n[' ,current_qwxyz, '], \n[',  target_qwxyz, ']\n[',target_qwxyz / current_qwxyz,']')
			dist = Quaternion.absolute_distance(target_qwxyz, current_qwxyz)
			angle_sum += dist*dist

		reward_imitation += 0.65 * np.exp(-2 * angle_sum)
		# Angular Velocities
		ang_vel_sum = 0
		'''
		index_offset = 1 + 10*(1 + len(self.reward_links))
		for (label_index, label) in enumerate(['root'] + self.reward_links):
			target_ang_vel = np.array(frame_data['ang_vel_'+ label])
			current_ang_vel = np.array(state[index_offset + label_index*3])
			dist = np.linalg.norm(target_ang_vel - current_ang_vel)
			ang_vel_sum += dist * dist

		#print('Angle:', angle_sum, np.exp(-2 * angle_sum))
		# TODO: Fix this
		reward_imitation += 0.1 * np.exp(-0.1 * ang_vel_sum)
		#print('Vel  :',ang_vel_sum, np.exp(-0.1 * ang_vel_sum))
		

		# TODO : Optimize reward evaluation, this is fast
		# End effector matching
		ee_deviation = 0
		for ee_label in self.end_effectors:
			ee_index = self.label2index[ee_label]
			if ee_index in self.link_states.keys():
				ee_pos = self.link_states[ee_index]
			else:
				ee_pos = pb.getLinkState(self.robotId, ee_index, physicsClientId = self.pybullet_client_ID)[0]
			ee_target = frame_data['ee_' + ee_label] # TODO : Error: This is local
			#print(ee_label,'EE Pos:',ee_pos, ee_target)
			ee_diff = np.linalg.norm(ee_pos - ee_target)
			ee_deviation += ee_diff * ee_diff

		reward_imitation += 0.15 * np.exp(-40 * ee_deviation)

		# Center of Mass deviation
		# TODO : Optimize reward evaluation, this is very slow. Find a better way to determine CoM
		
		base_mass = self.link_masses['root']
		total_mass = base_mass
		CoM = np.array(state[1:4]) * base_mass
		
		for link_label in self.reward_links:
			link_mass = self.link_masses[link_label]
			link_pos  = self.link_states[self.label2index[link_label]]#pb.getLinkState(self.robotId, self.label2index[link_label])[0] # self.prev_jstate[8 + (link_index)*7:8 + (link_index)*7 + 3]
			CoM += np.array(link_pos) * link_mass
			total_mass += link_mass
		
		CoM = CoM / total_mass
		
		#CoM = pb.getBasePositionAndOrientation(self.robotId, physicsClientId = self.pybullet_client_ID)[0]
		CoM_deviation = np.linalg.norm(frame_data['CoM'] - CoM)
		reward_imitation += 0.1 * np.exp(-10 * CoM_deviation*CoM_deviation)
		'''
		if False:
			print('ANG: ', angle_sum)
			print('CoM: ', CoM_deviation)
			print('EE: ', ee_deviation)
		
		#print('CoM  :',CoM_deviation, np.exp(-10 * CoM_deviation*CoM_deviation))
		#print(CoM, np.array(state[0:3]), self.data[work_frame]['CoM'])

		# Compute reward for achieving the goal
		weight_goal = 1
		reward_goal = 0
		# Nothing here for the static test
		
		# Compute final reward
		reward = weight_imitation * reward_imitation + weight_goal * reward_goal
		#print(weight_imitation * reward_imitation, weight_goal * reward_goal)

		# Return state, terminal bool and reward
		self.link_states = dict()
		
		# TODO : Implement early termination procedures
		#self.Early_Termination |= CoM_deviation > 1.0
		reward = reward * (not self.Early_Termination)
		
		done = (self.cur_step + 2 >= self.n_steps) or (self.init_frame + self.cur_step + 2 >= self.nframes )# or Early_Termination
		return (state, done, reward)

	def world2local(self, joint_index, body_xyz, body_orn):
		link_xyz, link_qwxyz = np.array(pb.getLinkState(self.robotId, joint_index, physicsClientId = self.pybullet_client_ID)[:2])
		self.link_states[joint_index] = link_xyz
		
		link_xyz = link_xyz - np.array(body_xyz)
		body_quat = Quaternion(np.array(body_orn)[[3,0,1,2]]).inverse
		local_xyz = body_quat.rotate(link_xyz)
		
		link_qwxyz = np.array(link_qwxyz)[[3,0,1,2]]
		link_qwxyz = Quaternion(link_qwxyz).unit
		local_qwxyz = (link_qwxyz*body_quat).unit
		
		return np.concatenate((local_xyz, local_qwxyz.elements[:]))

	def collect_observations(self):
		indices = [self.label2index[label] for label in self.reward_links]

		# Root position and orientation
		# 7 values
		body_xyz, (qx, qy, qz, qw) = pb.getBasePositionAndOrientation(self.robotId, physicsClientId = self.pybullet_client_ID)
		
		# Joint position and 
		link_positions = np.array([self.world2local(index, body_xyz, [qx, qy, qz, qw]) for index in indices]).flatten()
		#print(link_positions)
		# Phase variable
		# 1 value
		phi = ((self.init_frame + self.cur_step) % (self.nframes + 1))/ self.nframes

		output = np.concatenate([[phi]] + [list(body_xyz)] + [[qw, qx, qy, qz]] + [link_positions])
		#print(output)
		if(self.prev_jstate is None):
			self.prev_jstate = output

		# Linear velocity
		# Links
		linear_vel = [(np.array(output[8 + i*7:8 + (i)*7 + 3]) - np.array(self.prev_jstate[8 + i*7:8 + (i)*7 + 3]))/(2*self.tau) for i in range(len(self.reward_links))]
		# Root
		linear_vel = [(np.array(output[1:4]) - np.array(self.prev_jstate[1:4]))/(2*self.tau)] + linear_vel
		linear_vel = np.array(linear_vel).flatten()

		# Angular velocity
		angular_vel = []
		# Root
		q1 = Quaternion(output[4:8])
		q2 = Quaternion(self.prev_jstate[4:8])
		q3 = (q1 / q2).unit
		rotation = q3.radians / (2 * self.tau)
		angular_vel.append(q3.axis * rotation)
		# Links
		for i in range(len(self.reward_links)):
			#print(output[8 + 3 + i*7:8 + 3 + i*7 +4])
			q1 = Quaternion(output[8 + 3 + i*7:8 + 3 + i*7 +4])
			q2 = Quaternion(self.prev_jstate[8 + 3 + i*7:8 + 3 + i*7 +4])
			q3 = (q1 / q2).unit
			rotation = q3.radians / (2 * self.tau)
			angular_vel.append(q3.axis * rotation)
		angular_vel = np.array(angular_vel).flatten()

		# Store state for next frame
		self.prev_jstate = output
		#print([angular_vel])
		return np.concatenate([[phi]] + [list(body_xyz)] + [[qw, qx, qy, qz]] + [link_positions] + [linear_vel] + [angular_vel])