import numpy as np
from torch.utils.data import Dataset
import dgl
import torch

def collate(batch):
    # 将 batch 中的每个样本分解为 state、action、reward、next_state 和 absorbing
    state_goals, actions, rewards, next_states, absorbings, info, last = zip(*batch)
    # state_goal_tensor=torch.Size([128, 7])
    state_goal_tensor = torch.cat(state_goals)
    # actions_tensor=torch.Size([128, 5])
    actions_tensor = torch.tensor(np.array(actions))
    # rewards_tensor=torch.Size([128, 1])
    rewards_tensor = torch.tensor(rewards)
    rewards_tensor = torch.unsqueeze(rewards_tensor, 1)
    # next_state_goal_tensor=torch.Size([128, 7])
    next_state_goal_tensor = torch.cat(next_states)
    # absorbings_tensor=torch.Size([128, 1])
    absorbings_tensor = torch.tensor(absorbings)
    absorbings_tensor = torch.unsqueeze(absorbings_tensor, 1)
    # info_tensor=torch.Size([128, 1])
    info_tensor = torch.tensor(info)
    info_tensor = torch.unsqueeze(info_tensor, 1)
    # last_tensor.shape=torch.Size([128, 1])
    last_tensor = torch.tensor(last)
    last_tensor = torch.unsqueeze(last_tensor, 1)

    # 返回批量数据
    return state_goal_tensor, actions_tensor, rewards_tensor, next_state_goal_tensor, absorbings_tensor, info_tensor, last_tensor

def collate_graph(batch):
    # 将图数据转换为一个 DGL 图列表
    batched_graph = dgl.batch(batch)

    # 返回批量数据
    return {
        'graph': batched_graph
    }

class MyDataset(Dataset):
    def __init__(self, initial_size=1000, max_size=1000):
        self.data = []
        self._initial_size = initial_size
        self._max_size = max_size
        self._idx = 0
        self._full = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def initialized(self):
        """
        Returns:
            Whether the replay memory has reached the number of elements that
            allows it to be used.

        """
        return self.size > self._initial_size

    def add_data(self, sample):
        self.data.append(sample)
        self._idx += 1
        if self._idx == self._max_size:
            self._full = True
            self._idx = 0

    def add_batch(self, data_batch):
        self.data.extend(data_batch.data)
        len_ = len(data_batch)
        self._idx += len_
        if self._idx == self._max_size:
            self._full = True
            self._idx = 0

    def clear(self):
        self.data = list()
        self._idx = 0
        self._full = False

    def size(self):
        """
        Returns:
            The number of elements contained in the replay memory.

        """
        return self._idx if not self._full else self._max_size


class MyDataset_Graph(Dataset):
    def __init__(self, initial_size=1000, max_size=1000):
        self.data_graph = []
        self._initial_size = initial_size
        self._max_size = max_size
        self._idx = 0
        self._full = False

    def __len__(self):
        return len(self.data_graph)

    def __getitem__(self, idx):
        return self.data_graph[idx]

    def initialized(self):
        """
        Returns:
            Whether the replay memory has reached the number of elements that
            allows it to be used.

        """
        return self.size > self._initial_size

    def add_data(self, sample_graph):
        self.data_graph.append(sample_graph)
        self._idx += 1
        if self._idx == self._max_size:
            self._full = True
            self._idx = 0

    def add_batch(self, data_batch):
        self.data_graph.extend(data_batch.data_graph)
        len_ = len(data_batch)
        self._idx += len_
        if self._idx == self._max_size:
            self._full = True
            self._idx = 0

    def clear(self):
        self.data_graph = list()
        self._idx = 0
        self._full = False

    def size(self):
        """
        Returns:
            The number of elements contained in the replay memory.

        """
        return self._idx if not self._full else self._max_size
