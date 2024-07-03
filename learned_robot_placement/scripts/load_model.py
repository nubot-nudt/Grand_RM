import os
from datetime import datetime
import argparse

import numpy as np
import hydra
from omegaconf import DictConfig

import learned_robot_placement
from learned_robot_placement.utils.hydra_cfg.hydra_utils import *
from learned_robot_placement.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict

from learned_robot_placement.utils.task_util import initialize_task
from learned_robot_placement.envs.isaac_env_mushroom import IsaacEnvMushroom

# Use Mushroom RL library
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from mushroom_rl.core import Core, Logger
from mushroom_rl.algorithms.actor_critic import *
from mushroom_rl.utils.dataset import compute_J, parse_dataset
from tqdm import trange
from mushroom_rl.core.logger.console_logger import ConsoleLogger

# (Optional) Logging with Weights & biases
# from learned_robot_placement.utils.wandb_utils import wandbLogger
# import wandb

import os

# 定义了Critic网络
class CriticNetwork(nn.Module):
    # 继承自nn.Module的Python类
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        """

        Args:
            input_shape:表示输入的形状
            output_shape:表示输出的形状
            n_features:表示中间层的特征数
            **kwargs:
        """
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        # input_shape 表示输入的形状，output_shape 表示输出的形状，n_features 表示中间层的特征数
        # 定义了从n_input到n_features再到n_output的三个全连接层
        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        # 用于对权重进行 Xavier 初始化，使用 ReLU 激活函数时常用这种初始化方法
        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action):
        """
        定义了模型的前向传播过程，即给定输入计算输出
        Args:
            state:状态
            action:动作

        Returns:

        """
        # 状态和动作在第一个维度上拼接起来
        state_action = torch.cat((state.float(), action.float()), dim=1)
        # 通过两个隐藏层进行传播
        features1 = F.relu(self._h1(state_action))
        features2 = F.relu(self._h2(features1))
        # 不使用激活函数，直接输出q
        q = self._h3(features2)
        # 用于压缩去除维度
        return torch.squeeze(q)


# 定义了Actor网络
class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        """

        Args:
            input_shape:表示输入的形状
            output_shape:表示输出的形状
            n_features:表示中间层的特征数
            **kwargs:
        """
        super(ActorNetwork, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state):
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        a = self._h3(features2)

        return a

if __name__ == '__main__':
    agent = BHyRL.load("/home/lu/Desktop/embodied_ai/Issac_sim_learning/logs/agent-300.msh")
    state = np.array([[-61.275944, 177.79266, 0.7480975, 0.32706332, 0.6269207, \
                       0.32706332, -0.6269207, -60.858105, 177.71121, -60.931427, \
                     177.78368, 0.60124415, -28.641954, -61.329483, 177.70912, \
                     -61.40142, 177.80516, 0.55488855, -28.663586, -61.151394, \
                     177.56287, -61.194923, 177.45445, 0.535578, -26.73584, \
                     -61.166504, 177.7542, -61.22435, 177.68016, 0.32561198, \
                     -27.09545]])

    a, log_prob_next = agent.policy.compute_action_and_log_prob(state)

    q = agent._target_critic_approximator.predict(
        state, a, prediction='min')

    print("q=%s"%(q))




