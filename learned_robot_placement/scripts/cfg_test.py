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
from mushroom_rl.utils.dataset import compute_J, parse_dataset
from tqdm import trange
from mushroom_rl.core.logger.console_logger import ConsoleLogger

# (Optional) Logging with Weights & biases
# from learned_robot_placement.utils.wandb_utils import wandbLogger
# import wandb

import os


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
    # 加载检查点配置而非加载检查点本身
    if (cfg.checkpoint):
        print("Loading task and train config from checkpoint config file....")
        try:
            cfg_new = OmegaConf.load(os.path.join(os.path.dirname(cfg.checkpoint), '..',
                                                  'config.yaml'))  # 如果配置中存在检查点路径'cfg.checkpoint'，则加载相应的检查点配置文件
            cfg.task = cfg_new.task
            cfg.train = cfg_new.train
            print('cfg_new=%s'%(cfg_new))
        except Exception as e:
            print("Loading checkpoint config failed!")
            print(e)

    cfg_dict = omegaconf_to_dict(cfg)  # 转换配置为字典
    print_dict(cfg_dict)  # 将其打印出来


@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs_and_run_exp(cfg: DictConfig):
    experiment(cfg)


if __name__ == '__main__':
    parse_hydra_configs_and_run_exp()
