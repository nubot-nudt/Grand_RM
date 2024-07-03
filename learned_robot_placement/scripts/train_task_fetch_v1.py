# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
from mushroom_rl.utils.dataset import compute_J, parse_dataset2
from tqdm import trange
from mushroom_rl.core.logger.console_logger import ConsoleLogger
from tensorboardX import SummaryWriter
from learned_robot_placement.policy.graph_model import DGL_RGCN_RL, GCN_RL

# (Optional) Logging with Weights & biases
# from learned_robot_placement.utils.wandb_utils import wandbLogger
# import wandb

import os
import pickle
import dgl


class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

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

    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)
        features1 = F.relu(self._h1(state_action))
        features2 = F.relu(self._h2(features1))
        q = self._h3(features2)

        return torch.squeeze(q)


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
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

# 定义了Critic网络
class CriticNetworkGraph(nn.Module):
    # 继承自nn.Module的Python类
    def __init__(self, input_shape, output_shape, n_features, graph_model_critic, **kwargs):
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

        self.graph_model_critic = graph_model_critic

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

    def forward(self, state_goals, state_graphs, action):
        """
        定义了模型的前向传播过程，即给定输入计算输出
        Args:
            state:状态
            action:动作

        Returns:

        """
        # print("state_goals=")
        # print(state_goals.shape)
        #
        # print("state_graphs=")
        # print(state_graphs)
        #
        # print("action=")
        # print(action.shape)
        # input("critic_network_graph测试输入")

        state_embedding = self.graph_model_critic(state_graphs)

        # 状态和动作在第一个维度上拼接起来
        state_action = torch.cat((state_embedding.float(), action.float()), dim=1)

        # 通过两个隐藏层进行传播
        features1 = F.relu(self._h1(state_action))
        features2 = F.relu(self._h2(features1))
        # 不使用激活函数，直接输出q
        q = self._h3(features2)

        # 用于压缩去除维度
        return torch.squeeze(q)


# 定义了Actor网络
class ActorNetworkGraph(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, graph_model_actor, **kwargs):
        """

        Args:
            input_shape:表示输入的形状
            output_shape:表示输出的形状
            n_features:表示中间层的特征数
            **kwargs:
        """
        super(ActorNetworkGraph, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self.robot_ids = []

        self.graph_model_actor = graph_model_actor

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state_goals, state_graphs, a_cont=None):
        # total_len = state_graphs.num_nodes()

        # if a_cont is not None:
        #     print("state_goals=")
        #     print(state_goals.shape)
        #
        #     print("state_graphs=")
        #     print(state_graphs)
        #
        #     print("a_cont=")
        #     print(a_cont.shape)
        #     input("actor_network_graph测试输入")

        state_embedding = self.graph_model_actor(state_graphs)

        if a_cont is None:
            features1 = F.relu(self._h1(torch.squeeze(state_embedding, 1).float()))
        else:
            batch_concat = torch.cat((torch.squeeze(state_embedding, 1).float(), a_cont), dim=1)
            features1 = F.relu(self._h1(batch_concat.float()))
        features2 = F.relu(self._h2(features1))

        a = self._h3(features2)
        return a


def save_paramaters(model_parameters, epoch, string):
    # 创建一个空列表来保存参数
    params_list = []

    # 遍历参数生成器，并将参数添加到列表中
    for param in model_parameters:
        params_list.append(param.clone().detach())

    # 现在 params_list 中保存了每一层的参数，你可以对其进行检查
    with open('parameters_%s_%d.pkl'%(string, epoch), 'wb') as f:
        pickle.dump(params_list, f)

# 定义了实验场景
def experiment(cfg: DictConfig = None, cfg_file_path: str = "", seed: int = 0, results_dir: str = ""):
    """

    Args:
        cfg:一个配置字典
        cfg_file_path:配置文件的路径
        seed:随机数种子
        results_dir:结果目录

    Returns:

    """
    # Get configs获取配置
    # 读取cfg配置
    if (cfg_file_path):
        print('cfg_file_path=', cfg_file_path)
        # Get config file from path
        cfg = OmegaConf.load(cfg_file_path)
    # 加载检查点配置
    # if (cfg.checkpoint):
    #     print("Loading task and train config from checkpoint config file....")
    #     try:
    #         cfg_new = OmegaConf.load(os.path.join(os.path.dirname(cfg.checkpoint), '..',
    #                                               'config.yaml'))  # 如果配置中存在检查点路径'cfg.checkpoint'，则加载相应的检查点配置文件
    #         cfg.task = cfg_new.task
    #         cfg.train = cfg_new.train
    #     except Exception as e:
    #         print("Loading checkpoint config failed!")
    #         print(e)

    cfg_dict = omegaconf_to_dict(cfg)  # 转换配置为字典
    print_dict(cfg_dict)  # 将其打印出来
    headless = cfg.headless  # 设置headless参数
    render = cfg.render  # 设置渲染参数
    sim_app_cfg_path = cfg.sim_app_cfg_path  # 设置日志路径
    rl_params_cfg = cfg.train.params.config  # 设置rl环境参数
    algo_map = {"SAC_hybrid": SAC_hybrid,  # Mappings from strings to algorithms，设置要使用的rl算法
                "SAC": SAC,
                "BHyRL": BHyRL, }
    algo = algo_map[cfg.train.params.algo.name]  # 通过字典建立映射，通过参数algo: name: BHyRL进行选择

    # Set up environment 设置环境
    env = IsaacEnvMushroom(headless=headless, render=render, sim_app_cfg_path=sim_app_cfg_path)  # 设置Isaac的Mushroom仿真环境
    # 初始化任务环境，在此处定义任务环境，可以选择需要设定的任务是task中的FetchMultiObjFetching、FetchReaching和FetchWBExample
    task = initialize_task(cfg_dict, env)

    # Set up logging paths/directories 设置日志目录和文件
    exp_name = cfg.train.params.config.name
    exp_stamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')  # append datetime for logging，设置日志的时间戳
    results_dir = os.path.join(learned_robot_placement.__path__[0], 'logs', cfg.task.name, exp_name)
    if cfg.test: results_dir = os.path.join(results_dir, 'test')
    results_dir = os.path.join(results_dir, exp_stamp)
    os.makedirs(results_dir, exist_ok=True)

    # log experiment config 保存实验配置
    with open(os.path.join(results_dir, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))

    tensorboard_writer = SummaryWriter(logdir='logs_tensorboard/' + exp_name + '_' + exp_stamp)

    # Test/Train 测试/训练
    if (cfg.test):
        if (cfg.checkpoint):
            np.random.seed()
            # Logger
            logger = Logger(results_dir=results_dir, log_console=True)
            logger.strong_line()
            logger.info(f'Test: {exp_name}')
            logger.info(f'Test: Agent stored at ' + cfg.checkpoint)

            # Algorithm，加载预训练过的Agent算法
            logger.info(f'==============加载预训练的模型:%s============' % (cfg.checkpoint))
            agent = algo.load(cfg.checkpoint)

            # Runner，将Agent和env传给Core
            core = Core(agent, env)

            # 设置环境的渲染标志，以确定是否进行渲染
            env._run_sim_rendering = ((not cfg.headless) or cfg.render)
            # 获取数据集
            dataset = core.evaluate(n_episodes=50, render=cfg.render)
            J = np.mean(compute_J(dataset, env.info.gamma))
            R = np.mean(compute_J(dataset))
            s, *_ = parse_dataset2(dataset)
            E = agent.policy.entropy(s)  # 解析Agent的策略熵
            # 这里的J、R和E分别表示什么含义
            logger.info("Test: J=" + str(J) + ", R=" + str(R) + ", E=" + str(E))
        else:
            raise TypeError("Test option chosen but no valid checkpoint provided")
        env._simulation_app.close()
    else:
        # Approximators 近似器设置
        use_cuda = ('cuda' in cfg.rl_device)

        """Actor网络参数"""
        # 获取观测空间的形状
        actor_input_shape = env.info.observation_space.shape

        # Need to set these for hybrid action space!，混合空间的设置
        action_space_continous = (cfg.task.env.continous_actions,)  # 获取连续动作空间的形状
        action_space_discrete = (cfg.task.env.discrete_actions,)  # 获取离散动作空间的形状
        # Discrete approximator takes state and continuous action as input
        actor_discrete_input_shape = (env.info.observation_space.shape[0] + action_space_continous[0],)  # 获取离散动作空间的输入形状
        graph_model_actor = DGL_RGCN_RL()

        actor_mu_params = dict(network=ActorNetworkGraph,
                               n_features=rl_params_cfg.n_features,
                               input_shape=actor_input_shape,
                               output_shape=action_space_continous,
                               graph_model_actor=graph_model_actor,
                               use_cuda=use_cuda)
        actor_sigma_params = dict(network=ActorNetworkGraph,
                                  n_features=rl_params_cfg.n_features,
                                  input_shape=actor_input_shape,
                                  output_shape=action_space_continous,
                                  graph_model_actor=graph_model_actor,
                                  use_cuda=use_cuda)
        actor_discrete_params = dict(network=ActorNetworkGraph,
                                     n_features=rl_params_cfg.n_features,
                                     input_shape=actor_discrete_input_shape,
                                     output_shape=action_space_discrete,
                                     graph_model_actor=graph_model_actor,
                                     use_cuda=use_cuda)
        actor_optimizer = {'class': optim.Adam,
                           'params': {'lr': rl_params_cfg.lr_actor_net}}

        """Critic网络参数"""
        critic_input_shape = (actor_input_shape[0] + env.info.action_space.shape[0],)  # full action space
        # print("critic_input_shape=%s"%(critic_input_shape))
        # input("测试输入")
        graph_model_critic = DGL_RGCN_RL()
        critic_params = dict(network=CriticNetworkGraph,
                             optimizer={'class': optim.Adam,
                                        'params': {'lr': rl_params_cfg.lr_critic_net}},
                             loss=F.mse_loss,
                             n_features=rl_params_cfg.n_features,
                             input_shape=critic_input_shape,
                             output_shape=(1,),
                             graph_model_critic=graph_model_critic,
                             use_cuda=use_cuda)


        # Loop over num_seq_seeds: 循环运行实验设置
        for exp in range(cfg.num_seeds):
            np.random.seed()
            # Logger
            # 加载Debug信息
            logger = Logger(results_dir=results_dir, log_console=True)
            logger.strong_line()
            logger.info(f'Experiment: {exp_name}, Trial: {exp}')
            exp_eval_dataset = list()  # This will be a list of dicts with datasets from every epoch
            # wandb_logger = wandbLogger(exp_config=cfg, run_name=logger._log_id, group_name=exp_name+'_'+exp_stamp) # Optional

            # Agent
            """
            @param: env.info: 这可能是代理所操作的环境（environment）的信息或配置。它通常包含了环境的状态空间、动作空间等信息。
            @param: actor_mu_params: 这是用于创建代理的策略网络（actor network）的参数，可能包括神经网络的结构、激活函数等。
            @param: actor_sigma_params: 这是策略网络中用于控制动作分布的参数，通常与连续动作空间中的高斯分布有关。
            @param: actor_discrete_params: 如果代理操作的是离散动作空间，这个参数可能包含与策略网络相关的离散动作空间的参数。
            @param: actor_optimizer: 用于优化策略网络的优化器，可能是梯度下降法的一种变种，如Adam。
            @param: critic_params: 这可能是用于创建值函数网络（critic network）的参数，该网络用于估计状态值或状态-动作值。
            @param: batch_size: 用于训练神经网络时每个小批量的样本数量。
            @param: initial_replay_size: 强化学习中的经验回放（experience replay）起始时的缓冲区大小。
            @param: max_replay_size: 经验回放缓冲区的最大大小。
            @param: warmup_transitions: 强化学习算法开始训练之前用于填充经验回放缓冲区的随机动作数量。
            @param: tau: 用于更新目标网络的软更新参数。
            @param: lr_alpha: 用于控制更新策略网络中学习率的参数。
            @param: temperature: 在某些策略中，控制动作分布的温度参数。
            @param: log_std_min: 在一些策略中，控制策略网络输出的动作标准差的最小值。
            """
            logger.info("打印环境信息env.info=%s" % (env.info))
            logger.info("cfg.checkpoint=%s" % (str(cfg.checkpoint)))
            # input("测试")
            # cfg.checkpoint = 'learned_robot_placement/logs/FetchMultiObjFetching/FetchMultiObjFetchingBHyRL/2024-05-13-00-27-01_重要/2024-05-13-00-27-01/agent-1.msh'
            # agent = algo.load('learned_robot_placement/logs/FetchMultiObjFetching/FetchMultiObjFetchingBHyRL/2024-05-13-00-27-01_重要/2024-05-13-00-27-01/agent-1.msh')

            agent = algo(env.info, actor_mu_params, actor_sigma_params, actor_discrete_params, actor_optimizer,
                         critic_params,
                         batch_size=rl_params_cfg.batch_size, initial_replay_size=rl_params_cfg.initial_replay_size,
                         max_replay_size=rl_params_cfg.max_replay_size,
                         warmup_transitions=rl_params_cfg.warmup_transitions,
                         tau=rl_params_cfg.tau, lr_alpha=rl_params_cfg.lr_alpha, temperature=rl_params_cfg.temperature,
                         log_std_min=rl_params_cfg.log_std_min)

            # Setup boosting (for BHyRL):
            # 设置历史学习过的agent，在该agent的基础上继续学习
            if rl_params_cfg.prior_agents is not None:
                prior_agents = list()
                for agent_path in rl_params_cfg.prior_agents:  # 是rl设置的一个参数
                    # learned_robot_placement.__path__[0]就是位于learned_robot_placement下的目录
                    print("加载历史模型数据为%s" % (os.path.join(learned_robot_placement.__path__[0], agent_path)))
                    prior_agents.append(algo.load(os.path.join(learned_robot_placement.__path__[0], agent_path)))
                agent.setup_boosting(prior_agents=prior_agents, use_kl_on_pi=rl_params_cfg.use_kl_on_pi,
                                     kl_on_pi_alpha=rl_params_cfg.kl_on_pi_alpha)

            # Algorithm，算法，其输入参数是agent和env
            # 要弄清楚agent即对应的rl算法是什么，以及env环境参数
            core = Core(agent, env)

            # RUN
            eval_dataset = core.evaluate(n_steps=rl_params_cfg.n_steps_test, render=cfg.render)
            s, _, _, _, _, info, last = parse_dataset2(eval_dataset)
            # 这里的J、R和E分别表示什么含义
            J = np.mean(compute_J(eval_dataset, env.info.gamma))
            R = np.mean(compute_J(eval_dataset))
            E = agent.policy.entropy(s)

            # 计算成功率
            success_rate = np.sum(info) / np.sum(last)  # info contains successes. rate=num_successes/num_episodes
            avg_episode_length = rl_params_cfg.n_steps_test / np.sum(last)
            # 在这一句话可以打印信息
            # logger.epoch_info(0, success_rate=success_rate, J=J, R=R, entropy=E, avg_episode_length=avg_episode_length)
            logger.epoch_info(0, success_rate=success_rate, J=J, R=R, avg_episode_length=avg_episode_length)
            # save_paramaters(agent.policy._mu_approximator.model.network.graph_model_actor.parameters(), 0, 'actor')
            # save_paramaters(agent._critic_approximator.model._model[0].network.graph_model_critic.parameters(), 0, 'critic')

            # 监视模型的训练
            # tensorboard_writer.add_graph(agent.policy._mu_approximator.model.network)
            # tensorboard_writer.add_graph(agent.policy._sigma_approximator.model.network)
            # tensorboard_writer.add_graph(agent._critic_approximator.model._model[0].network)

            # 评估成功率
            exp_eval_dataset.append({"Epoch": 0, "success_rate": success_rate, "J": J, "R": R, "E":E,
                                     "avg_episode_length": avg_episode_length})
            # initialize replay buffer
            # initial_replay_size=1024
            core.learn(n_steps=rl_params_cfg.initial_replay_size, n_steps_per_fit=rl_params_cfg.initial_replay_size,
                       render=cfg.render)

            # 重复进行训练
            # n_epochs是训练的轮数，设置300或250
            for n in trange(rl_params_cfg.n_epochs, leave=False):
                core.learn(n_steps=rl_params_cfg.n_steps, n_steps_per_fit=1, render=cfg.render)

                # 评估数据集n_steps_test，多少轮进行一次评估
                eval_dataset = core.evaluate(n_steps=rl_params_cfg.n_steps_test, render=cfg.render)
                s, _, _, _, _, info, last = parse_dataset2(eval_dataset)
                J = np.mean(compute_J(eval_dataset, env.info.gamma))
                R = np.mean(compute_J(eval_dataset))
                E = agent.policy.entropy(s)  # 计算熵值

                # 成功率指标
                success_rate = np.sum(info) / np.sum(last)  # info contains successes. rate=num_successes/num_episodes
                avg_episode_length = rl_params_cfg.n_steps_test / np.sum(last)
                q_loss = core.agent._critic_approximator[0].loss_fit
                actor_loss = core.agent._actor_last_loss

                # 打印信息
                logger.epoch_info(n + 1, success_rate=success_rate, J=J, R=R, entropy=E,
                                  avg_episode_length=avg_episode_length,
                                  q_loss=q_loss, actor_loss=actor_loss)

                # save_paramaters(agent.policy._mu_approximator.model.network.graph_model_actor.parameters(), n+1, 'actor')
                # save_paramaters(agent._critic_approximator.model._model[0].network.graph_model_critic.parameters(), n+1, 'critic')

                if (rl_params_cfg.log_checkpoints):
                    logger.log_agent(agent, epoch=n + 1)  # Log agent every epoch
                    # logger.log_best_agent(agent, J) # Log best agent
                # current_log = {"success_rate": success_rate, "J": J, "R": R, "entropy": E,
                #                "avg_episode_length": avg_episode_length, "q_loss": q_loss, "actor_loss": actor_loss}
                current_log = {"success_rate": success_rate, "J": J, "R": R, "E": E,
                               "avg_episode_length": avg_episode_length, "q_loss": q_loss, "actor_loss": actor_loss}
                exp_eval_dataset.append(current_log)
                # wandb_logger.run_log_wandb(success_rate, J, R, E, avg_episode_length, q_loss)
                tensorboard_writer.add_scalar('Success Rate', success_rate, n + 1)
                tensorboard_writer.add_scalar('J', J, n + 1)
                tensorboard_writer.add_scalar('R', R, n + 1)
                tensorboard_writer.add_scalar('E', E, n + 1)
                tensorboard_writer.add_scalar('Average Episode Length', avg_episode_length, n + 1)
                tensorboard_writer.add_scalar('Q Loss', q_loss, n + 1)
                tensorboard_writer.add_scalar('actor_loss', actor_loss, n + 1)

            # Get video snippet of final learnt behavior (enable internal rendering for this)
            # prev_env_render_setting = bool(env._run_sim_rendering)
            # env._run_sim_rendering = True
            # img_dataset = core.evaluate(n_episodes=5, get_renders=True)
            # env._run_sim_rendering = prev_env_render_setting
            # log dataset and video
            # logger.log_dataset(exp_eval_dataset)
            # run_log_wandb(exp_config=cfg, run_name=logger._log_id, group_name=exp_name+'_'+exp_stamp, dataset=exp_eval_dataset)
            # img_dataset = img_dataset[::15] # Reduce size of img_dataset. Take every 15th image
            # wandb_logger.vid_log_wandb(img_dataset=img_dataset)

    # Shutdown
    env._simulation_app.close()
    tensorboard_writer.close()


@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs_and_run_exp(cfg: DictConfig):
    experiment(cfg)


if __name__ == '__main__':
    parse_hydra_configs_and_run_exp()
