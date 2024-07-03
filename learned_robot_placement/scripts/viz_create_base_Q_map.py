import os

import numpy as np
import torch
import math

import copy
import argparse
import h5py
import pdb

from mushroom_rl.algorithms.actor_critic import BHyRL # mixed action space
from mushroom_rl.algorithms.actor_critic import SAC_hybrid # mixed action space
from sac_hybrid_data_prior_tiago_dual_reaching import CriticNetwork, ActorNetwork # Need these networks to load the previous task's agent
from bhyrl_tiago_dual_room_reaching import BHyRL_CriticNetwork, BHyRL_ActorNetwork

ISAAC_PATH = os.environ.get('ISAAC_PATH')
results_dir=ISAAC_PATH+"/tiago_isaac/experiments/logs/Q_maps"
if torch.cuda.is_available():
    device = "cuda"
    print("[GPU MEMORY size in GiB]: " + str((torch.cuda.get_device_properties(0).total_memory-torch.cuda.memory_reserved(0))/1024**3))
else:
    device = "cpu"
dtype = torch.float64 # Choose float32 or 64 etc.

# Map settings，地图的设置部分
map_xy_resolution = 0.03 # in metres，地图的xy分辨率
map_xy_radius = 1.0 # metres，地图的半径
angular_res = np.pi/8 # or 22.5 degrees per bin)，角度分辨率，每个点有16个角度
Q_scaling = 100.0 # TODO: play with this， Q值的缩放因子
x_bins = math.ceil(map_xy_radius*2/map_xy_resolution) # 在给定分辨率下x、y和角度方向上的网格数
y_bins = math.ceil(map_xy_radius*2/map_xy_resolution) # 在给定分辨率下x、y和角度方向上的网格数
theta_bins = math.ceil((2*np.pi)/angular_res) # 在给定分辨率下x、y和角度方向上的网格数
x_ind_offset = y_bins*theta_bins
y_ind_offset = theta_bins
theta_ind_offset = 1
num_voxels = x_bins*y_bins*theta_bins

# 存储了多个先前任务的代理
# Load agents from prior tasks (since we are learning the residuals). NOTE: Mind the order of the prior tasks
prior_agents = list()
prior_agents.append(SAC_hybrid.load(ISAAC_PATH+'/tiago_isaac/experiments/logs/SAC_hybrid/2022-02-17-02-23-19/2022-02-17-05-33-31/agent-123.msh')) # higher entropy
prior_agents.append(BHyRL.load(ISAAC_PATH+'/tiago_isaac/experiments/logs/BHyRL/2022-02-20-17-59-58-5mEntropy/2022-02-20-18-02-05/agent-213.msh'))
# Agent
# prior_agents.append(BHyRL.load(ISAAC_PATH+'/tiago_isaac/experiments/logs/BHyRL_labroom/2022-02-23-04-57-46/2022-02-23-04-59-18/agent-310.msh'))
prior_agents.append(BHyRL.load(ISAAC_PATH+'/tiago_isaac/experiments/logs/BHyRL_labroom/2022-04-08-09-09-12/2022-04-08-09-09-47/agent-242.msh'))

agent_net_batch_size = 128
agent_world_xy_radius = 3.0
agent_world_norm_theta = np.pi
agent_action_xy_radius = 1.0
agent_action_ang_lim = np.pi

# mdp get prior state function. Needed for Boosted RL agents
# 获取先前状态的函数。增强型RL代理所需
def get_prior_task_states(new_task_states, ik_task=False, reach_task=False): # convert the state from a new task's state space to the prior task's state space
    # 定义函数get_prior_task_states，用于获取先前任务状态，参数包括新任务状态new_task_states、是否逆运动学任务ik_task和是否到达任务reach_task，默认为False。
    if(len(new_task_states.shape) < 2):
        # 如果新任务状态的形状不是二维的，如果接收到的状态是一个单独的数组，则将其转换为二维数
        # If received state is a single array, unsqueeze it
        new_task_states = np.array([new_task_states])
    weights = np.ones(shape=new_task_states.shape[0]) # also return weighting for the states
    # 初始化权重数组weights，其形状与new_task_states相同，用于存储状态的权重信息
    # 根据先前的任务，移除多余的状态变量
    # Remove excess state variables as per prior task

    if(ik_task):
        # 只保留新任务状态的前6个元素（即6D目标状态）
        prior_task_states = copy.deepcopy(new_task_states[:,0:6]) # only using the 6D goal state
        # Prior task is simple IK 6D reaching in a 1m world radius
        # 先前任务是简单的6D逆运动学任务，目标是在1米的世界半径内到达
        # 将xyz距离缩放到agent_world_xy_radius
        prior_task_states[:,0:3] *= agent_world_xy_radius # Re-scale xyz distances
        # 计算状态的xy平面距离
        xy_distances = np.linalg.norm(prior_task_states[:,0:2], axis=1)
        # weights[np.linalg.norm(xy_distances > 1.0] = 0.0 # accept only states where xy distance is less than 1 metre
        # weights = np.clip(1/xy_distances, 0.0, 1.0) # weight as per 1/distance. Clip to maximum weight of 1.0
        # 使用基于距离的权重计算方法，通过1-tanh函数对距离进行归一化
        weights = np.clip(1.0 - np.tanh(xy_distances-1.0), 0.0, 1.0) # Weights as per distance metric: 1-tanh. Clip to maximum weight of 1.0
    elif(reach_task):
        # 同样只保留新任务状态的前6个元素
        prior_task_states = copy.deepcopy(new_task_states[:,0:6]) # only using the 6D goal state
        # 将xyz距离缩放到agent_world_xy_radius
        # prior task is a 5m free reaching task
        prior_task_states[:,0:3] *= agent_world_xy_radius # Re-scale xyz distances
        # 将xyz距离缩放到5.0
        # convert to distances for prior 5m reaching task
        prior_task_states[:,0:3] /= 5.0
    else:
        # 直接将新任务状态复制到先前任务状态中
        # Prior task is the same (final task)
        prior_task_states = copy.deepcopy(new_task_states)
    # 返回计算得到的权重数组以及处理后的先前任务状态数组
    return weights, prior_task_states

# Get saved, un-transformed states from file:
with open(ISAAC_PATH+'/tiago_isaac/experiments/state_poses0.np','rb') as f:
    state_poses = np.load(f)

# 这里是对state进行转换获取相应的state
# Modify and get relevant states
grasp_obj_trans = state_poses[0,:]
grasp_obj_rot = state_poses[1,:]
obstacle_trans = state_poses[2,:]
obstacle_rot = state_poses[3,:]
# Find max, min vertices
max_xy_offset = np.array([0.55/2, 0.77/2])
min_xy_offset = -max_xy_offset
# rotate by theta
bbox_tf = np.zeros((3,3))
bbox_tf[:2,:2] = np.array([[np.cos(obstacle_rot[2]), -np.sin(obstacle_rot[2])],[np.sin(obstacle_rot[2]), np.cos(obstacle_rot[2])]])
bbox_tf[:,-1] = np.array([0.0, 0.0, 1.0])
max_xy_vertex = np.array([[max_xy_offset[0],max_xy_offset[1],1.0]]).T
min_xy_vertex = np.array([[min_xy_offset[0],min_xy_offset[1],1.0]]).T
new_max_xy_vertex = (bbox_tf @ max_xy_vertex)[0:2].T.squeeze() + obstacle_trans[0:2]
new_min_xy_vertex = (bbox_tf @ min_xy_vertex)[0:2].T.squeeze() + obstacle_trans[0:2]

obs_z = 0.48 # fixed
obs_theta = obstacle_rot[2]
# Generate grasp
grasp_obj_trans[2] += 0.08 # z offset for top grasps
# array([ 0.31196376,  0.82159081, -0.31096255])
# Adding z from ground: 1.0530189275741577
grasp_obj_trans[2] += 1.0530189275741577
print(f"grasp_obj_trans: {grasp_obj_trans}")
# array([ 0.31196376,  0.82159081, 0.7420563775741578])
grasp_obj_rot[1] += np.radians(88) # Pitch by almost 90 degrees
print(f"grasp_obj_rot: {grasp_obj_rot}")
# array([-0.00640782,  1.52538862,  2.32906671])

# 当前的状态
state = np.array([grasp_obj_trans[0], grasp_obj_trans[1], grasp_obj_trans[2],
                    grasp_obj_rot[0], grasp_obj_rot[1], grasp_obj_rot[2],
                    new_max_xy_vertex[0], new_max_xy_vertex[1], new_min_xy_vertex[0], new_min_xy_vertex[1],
                    obs_z, obs_theta])
# TODO：定义了动作，对state进行了规范化，这里的state存在多个
state_normalized = np.array([state])
# 0:3是[x, y, theta]
state_normalized[0:3] /= agent_world_xy_radius
# 3:6是角度
state_normalized[3:6] /= agent_world_norm_theta
state_normalized[6:-1] /= agent_world_xy_radius
state_normalized[-1] /= agent_world_norm_theta

# TODO：获取机器人的理论上的最佳动作
# Optional: Query policy for optimal (in-theory) action:
act = prior_agents[-1].policy.draw_action(state_normalized)[0]
x_scaled = act[0]*np.cos(act[1]*np.pi)
y_scaled = act[0]*np.sin(act[1]*np.pi)
theta_scaled = np.degrees(act[2]*np.pi)
print(f"Action from policy: x:{x_scaled}, y:{y_scaled}, theta(deg):{theta_scaled}")

# Create base inverse reachability map:

# Actions: generate discrete actions (x-y-theta) around the robot as per resolution
# Use a map_xy_radius (try 1 or 1.2 metres) radius around the robot
# TODO: Figure out meshgrids so we don't have to use for loops
# xs = torch.linspace(-map_xy_radius, map_xy_radius, steps=x_bins)
# ys = torch.linspace(-map_xy_radius, map_xy_radius, steps=y_bins)
# thetas = torch.linspace(-np.pi, np.pi, steps=theta_bins)
# mesh_x, mesh_y, mesh_theta = torch.meshgrid(xs, ys, thetas, indexing='xy')
# 基于给定的分辨率和半径，在机器人周围生成离散的动作空间，以便构建逆运动学可达性地图
# 前三个维度是[x, y, theta]，最后一个维度是Q值
base_reach_map = torch.zeros((num_voxels, 4), dtype=dtype, device=device) # 4 values in each row: x,y,theta and Q
idx = 0

# 相当于是在-map_xy_radius到map_xy_radius的范围内画矩形，且每个位置都会同时存在多个actions
for x_val in torch.linspace(-map_xy_radius, map_xy_radius, steps=x_bins):
    for y_val in torch.linspace(-map_xy_radius, map_xy_radius, steps=x_bins):
        for theta_val in torch.linspace(-np.pi, np.pi, steps=theta_bins):
            base_reach_map[idx,0:-1] = torch.hstack([x_val, y_val, theta_val])
            idx+=1

# 只选择在极坐标范围内的动作
# Only take spheres in polar co-ordinates within polar co-ord range (radius)
# 计算长度，其实这里保留了所有的角度而非只有某一个角度
valid_idxs = torch.linalg.norm(base_reach_map[:,0:2],dim=1) <= agent_action_xy_radius
base_reach_map = base_reach_map[valid_idxs]

# 将动作转换为极坐标形式，将当前动作转换为极坐标的形式
# convert x-y to r-phi
# TODO:获取action部分，其实就是根据当前位置构造的动作，注意这里的动作其实和state本身无关，因为动作本身就是一个相对于机器人坐标系的量
action_query = torch.zeros(base_reach_map.shape[0],5, device=device) # This is as per the exact action space. For these agents it is 5 dim, 3 cont and 2 discrete
action_query[:,0] = torch.linalg.norm(base_reach_map[:,0:2],dim=1) # norm of x and y gives r，其实就是根号下(x^2 + y^2)求解r
action_query[:,1] = torch.atan2(base_reach_map[:,1],base_reach_map[:,0]) # arctan2(y/x) gives phi，求解角度
action_query[:,2] = base_reach_map[:,2].clone()
action_query[:,3] = torch.tensor([1.0],device=device)
# Normalize actions，对动作进行归一化
action_query /= torch.tensor([agent_action_xy_radius,agent_action_ang_lim,agent_action_ang_lim,1.0,1.0], device=device)

# query agent's critics in batch and get the q values (scores)
# Loop over prior critics (boosting case)
# Unfortunately you can only have a batch size equal to what the networks use, so we have to loop
# 批量查询智能体的评分（Q值）,确保所有动作维度的取值范围在0到1之间。归一化是通过将动作除以各自的最大取值范围得到的
# 按照agent_net_batch_size进行遍历
num_loops = math.ceil(action_query.shape[0]/agent_net_batch_size)
for n in range(num_loops):
    action_query_batch = action_query[n*agent_net_batch_size:(n+1)*agent_net_batch_size,:]
    # 这一部分其实就是将agent的值进行相加
    for idx, prior_agent in enumerate(prior_agents):
        # Use weights for the prior rho values. Also use appropriate state-spaces as per the prior task
        # TODO：获取状态state部分
        weights, prior_state = get_prior_task_states(state_normalized, ik_task=(idx==0), reach_task=(idx==1)) # task[0],task[1] are IK prior and 5m reaching tasks resp
        weights = torch.tensor(weights, device=device).repeat(action_query_batch.shape[0],1) # resize to be same as num of actions
        state_query_batch = torch.tensor(prior_state, device=device).repeat(action_query_batch.shape[0],1) # resize to be same as num of actions
        # TODO:Q值的查询就是在这部分进行实现的
        rho_prior = weights.squeeze() * prior_agent._target_critic_approximator.predict(state_query_batch, action_query_batch, output_tensor=True, prediction='min').values
        # 这里其实有很多的值包括了agent_net_batch大小的值
        base_reach_map[n*agent_net_batch_size:(n+1)*agent_net_batch_size, -1] += rho_prior # Store Q values in base_reach_map

base_reach_map = base_reach_map.detach().cpu().numpy() # Move to numpy
# Accumulate 2D+orientation scores into every 2D voxel
indx = 0
first = True
while indx < base_reach_map.shape[0]:
    # 获取base_reach_map的x和y坐标
    sphere_2d = base_reach_map[indx][:2]
    # 计算base_reach_map中的重复值，这里有一部分重复值
    # Count num_repetitions of current 2D sphere (in the next y_ind_offset subarray)，y+ind_offset其实就是theta_bin
    num_repetitions = (base_reach_map[indx:indx+y_ind_offset][:,:2] == sphere_2d).all(axis=1).sum().astype(dtype=np.int16)
    # sphere是sphere_2d的堆叠
    sphere_3d = np.hstack((sphere_2d, 0.0))
    # 计算平均值
    # Store sphere and average Q as the score. (Also, scale by a factor)
    # 取了多个位置求平均值
    Q_avg = base_reach_map[indx:indx+num_repetitions, 3].mean()
    if first:
        first = False
        sphere_array = np.append(base_reach_map[indx][:2], [0.0, Q_avg])
        # sphere_array = np.append(reach_map_nonzero[indx][:3], num_repetitions) # Optional: Use num_repetitions as score instead
        pose_array = np.append(base_reach_map[indx][:2], np.array([0., 0., 0., 0., 0., 0., 0., 1.])).astype(np.single) # dummy value
    else:
        sphere_array = np.vstack((sphere_array, np.append(base_reach_map[indx][:2], [0.0, Q_avg])))
        # sphere_array = np.vstack((sphere_array, np.append(reach_map_nonzero[indx][:3], num_repetitions)))  # Optional: Use num_repetitions as score instead
        pose_array = np.vstack((pose_array, np.append(base_reach_map[indx][:2], np.array([0., 0., 0., 0., 0., 0., 0., 1.])).astype(np.single))) # dummy value
    indx += num_repetitions


# # Optional: Normalize Q values in the map
min_Q = sphere_array[:,-1].min()
max_Q = sphere_array[:,-1].max()
sphere_array[:,-1] -= min_Q
sphere_array[:,-1] /= (max_Q-min_Q)
sphere_array[:,-1] *= Q_scaling

# # Save 3D map as hdf5 file for visualizer (Mimic reuleux data structure)
with h5py.File(results_dir+"/3D_Q_state_poses0.h5", 'w') as f:
    sphereGroup = f.create_group('/Spheres')
    sphereDat = sphereGroup.create_dataset('sphere_dataset', data=sphere_array)
    sphereDat.attrs.create('Resolution', data=map_xy_resolution)
    # (Optional) Save all the 6D poses in each 3D sphere. Currently only dummy pose values (10 dimensional)
    poseGroup = f.create_group('/Poses')
    poseDat = poseGroup.create_dataset('poses_dataset', dtype=float, data=pose_array)