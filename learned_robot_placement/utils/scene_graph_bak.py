import torch
import numpy as np
import dgl
from dgl import DGLGraph
from crowd_nav.utils.rvo_inter import rvo_inter
class SceneGraph():
    def __init__(self, data):
        #        ntypes = ['robot', 'human', 'obstacle', 'wall'], etypes = ['h_r', 'o_r', 'w_r', 'h_h', 'o_h', 'w_h', 'r_h']
        super(CrowdNavGraph, self).__init__()
        self.graph = None
        self.data = None
        self.robot_visible = False
        # 包含不同类型关系的列表，其中每个元素表示一个关系类型
        # 这样做的目的是为了使得图在后续的处理中能够区分不同类型的边，即对应于边的权重
        # 通过使用这些关系类型，代码可以正确地标识和区分从不同实体到机器人的边，从而构建了一个更加复杂的社交图
        # 'h2r'：表示人物到机器人的关系
        # 'o2r'：表示障碍物到机器人的关系
        # 'w2r'：表示墙壁到机器人的关系
        # 'o2h'：表示障碍物到人物的关系
        # 'w2h'：表示墙壁到人物的关系
        # 'h2h'：表示人物到人物的关系
        self.rels = ['h2r', 'o2r', 'w2r', 'o2h', 'w2h', 'h2h']
        self.use_rvo = True
        self.mode = 0
        if self.use_rvo is True:
            self.rvo_inter = rvo_inter(neighbor_region=6, neighbor_num=20, vxmax=1, vymax=1, acceler=1.0,
                                       env_train=True,
                                       exp_radius=0.0, ctime_threshold=3.0, ctime_line_threshold=3.0)
            rotated_data = self.config_rvo_state(data)
            self.build_up_graph_on_rvostate(rotated_data)
        else:
            rotated_data = self.rotate_state(data)
            self.initializeWithAlternativeRelational(rotated_data)

    def rotate_state(self, state):
        """
        Transform the coordinate to agent-centric.
        Input tuple include robot state tensor and human state tensor.
        robot state tensor is of size (number, state_length)(for example 1*9)
        human state tensor is of size (number, state_length)(for example 5*5)
        obstacle state tensor is of size (number, state_length)(for example 3*3)
        wall state tensor is of size (number, state_length)(for example 4*4)
        """
        # for robot
        # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta'
        #  0     1      2     3      4        5     6      7         8
        # for human
        #  'px', 'py', 'vx', 'vy', 'radius'
        #  0     1      2     3      4
        # for obstacle
        # 'px', 'py', 'radius'
        #  0     1     2
        # for wall
        # 'sx', 'sy', 'ex', 'ey', radius
        #  0     1     2     3
        assert len(state[0].shape) == 2
        assert state[2] is not None
        robot_state = state[0]
        human_state = state[1]
        obstacle_state = state[2]
        wall_state = state[3]
        dx = robot_state[:, 5] - robot_state[:, 0]
        dy = robot_state[:, 6] - robot_state[:, 1]
        dx = dx.unsqueeze(1)
        dy = dy.unsqueeze(1)

        dg = torch.norm(torch.cat([dx, dy], dim=1), 2, dim=1, keepdim=True)

        rot = torch.atan2(dy, dx)
        cos_rot = torch.cos(rot)
        sin_rot = torch.sin(rot)
        transform_matrix = torch.cat((cos_rot, -sin_rot, sin_rot, cos_rot), dim=0).reshape(2, 2)

        a = robot_state[:, 2:4]
        robot_velocities = robot_state[:, 2:4]
        robot_linear_velocity = torch.sum(robot_velocities, dim=1, keepdim=True) * 0.5
        robot_angular_velocity = (robot_velocities[:, 1] - robot_velocities[:, 0]) / (2 * 0.3)
        robot_angular_velocity = robot_angular_velocity.unsqueeze(0)
        radius_r = robot_state[:, 4].unsqueeze(1)
        v_pref = robot_state[:, 7].unsqueeze(1)
        target_heading = torch.zeros_like(radius_r)
        pos_r = torch.zeros_like(robot_velocities)
        cur_heading = (robot_state[:, 8].unsqueeze(1) - rot + np.pi) % (2 * np.pi) - np.pi
        new_robot_state = torch.cat((robot_velocities, dg, v_pref, cur_heading),
                                    dim=1)
        new_state = new_robot_state
        if state[1].shape[0] > 0:
            human_positions = human_state[:, 0:2] - robot_state[:, 0:2]
            human_positions = torch.mm(human_positions, transform_matrix)
            human_velocities = human_state[:, 2:4]
            human_velocities = torch.mm(human_velocities, transform_matrix)
            human_radius = human_state[:, 4].unsqueeze(1) + 0.3
            new_human_state = torch.cat((human_positions, human_velocities, human_radius), dim=1)
        else:
            new_human_state = None
        if state[2] is not None:
            obstacle_positions = obstacle_state[:, 0:2] - robot_state[:, 0:2]
            obstacle_positions = torch.mm(obstacle_positions, transform_matrix)
            obstacle_radius = obstacle_state[:, 2].unsqueeze(1) + 0.3
            new_obstacle_states = torch.cat((obstacle_positions, obstacle_radius), dim=1)
        else:
            new_obstacle_states = None
        if state[3] is not None:
            wall_start_positions = wall_state[:, 0:2] - robot_state[:, 0:2]
            wall_start_positions = torch.mm(wall_start_positions, transform_matrix)
            wall_end_positions = wall_state[:, 2:4] - robot_state[:, 0:2]
            wall_end_positions = torch.mm(wall_end_positions, transform_matrix)
            wall_radius = torch.zeros((wall_state.shape[0], 1)) + 0.3
            new_wall_states = torch.cat((wall_start_positions, wall_end_positions, wall_radius), dim=1)
        else:
            new_wall_states = None

        new_state = (new_robot_state, new_human_state, new_obstacle_states, new_wall_states)

        return new_state


    def rvo_state(self, state):
        return state
    def initializeWithAlternativeRelational(self, data):
        # 用于存储边的源节点和目标节点的标识
        src_id = None
        dst_id = None
        # We create a map to store the types of the nodes. We'll use it to compute edges' types
        # 是一个字典，用于存储节点类型
        self.typeMap = dict()
        # 用于存储节点的位置信息
        position_by_id = {}

        # Node Descriptor Table
        # 定义了一个包含四个元素的列表，表示节点描述符的头部，分别对应 'r'（机器人）、'h'（人类）、'o'（障碍物）和 'w'（墙）
        self.node_descriptor_header = ['r', 'h', 'o', 'w']

        # # Relations are integers
        # RelTensor = torch.LongTensor
        # # Normalization factors are floats
        # NormTensor = torch.Tensor
        # # Generate relations and number of relations integer (which is only accessed outside the class)
        # max_used_id = 0 # 0 for the robot
        # # Compute closest human distance
        # closest_human_distance = -1
        # Feature dimensions

        # 定义了一个包含四个元素的列表，表示节点的类型。每个元素对应一个节点类型，例如 'robot' 对应机器人，'human' 对应人类，以此类推
        node_types_one_hot = ['robot', 'human', 'obstacle', 'wall']
        # robot_metric_features = ['rob_pos_x', 'robot_pos_y', 'rob_vel_x', 'rob_vel_y', 'rob_radius', 'rob_goal_x',
        #                          'rob_goal_y', 'rob_vel_pre', 'rob_ori']

        # 定义了四个列表，分别表示机器人、人类、障碍物和墙的度量特征。这些特征是节点的属性，如位置、速度、半径等
        robot_metric_features = ['rob_vel_l', 'rob_vel_r', 'dis2goal', 'rob_vel_pre', 'rob_ori']
        human_metric_features = ['human_pos_x', 'human_pos_y', 'human_vel_x', 'human_vel_y', 'human_radius']
        obstacle_metric_features = ['obs_pos_x', 'obs_pos_y', 'obs_radius']
        wall_metric_features = ['start_pos_x', 'start_pos_y', 'enb_pos_x', 'end_pos_y', 'wall_radius']

        # 将这些特征列表进行合并
        all_features = node_types_one_hot + robot_metric_features + human_metric_features + obstacle_metric_features + wall_metric_features

        # Copy input data
        # 将传入的数据 data 分解成四个部分，分别赋值给 robot_state、human_state、obstacle_state 和 wall_state
        self.data = data
        robot_state, human_state, obstacle_state, wall_state = self.data

        # 计算特征维度 feature_dimensions 和各类节点的数量：robot_num、human_num、obstacle_num 和 wall_num
        feature_dimensions = len(all_features)
        robot_num = robot_state.shape[0]
        if human_state is not None:
            human_num = human_state.shape[0]
        else:
            human_num = 0
        if obstacle_state is not None:
            obstacle_num = obstacle_state.shape[0]
        else:
            obstacle_num = 0
        if wall_state is not None:
            wall_num = wall_state.shape[0]
        else:
            wall_num = 0

        # 计算所有节点的总数
        total_node_num = robot_num + human_num + obstacle_num + wall_num
        # fill data into the heterographgraph
        # data of the robot

        # 创建一个全零张量，表示机器人节点的特征，其形状为 (robot_num, feature_dimensions)
        robot_tensor = torch.zeros((robot_num, feature_dimensions))
        # 将机器人节点的类型特征设为 1
        robot_tensor[0, all_features.index('robot')] = 1
        # 将机器人节点的度量特征设为对应的数值
        robot_tensor[0, all_features.index('rob_vel_l'):all_features.index("rob_ori") + 1] = robot_state[0]
        # self.graph.nodes['robot'].data['h'] = robot_tensor
        # 将机器人节点的特征赋值给 features
        features = robot_tensor

        # 如果存在人类节点，依次创建人类节点的特征张量，并将其与之前的特征张量拼接起来
        if human_num > 0:
            human_tensor = torch.zeros((human_num, feature_dimensions))
            for i in range(human_num):
                human_tensor[i, all_features.index('human')] = 1
                human_tensor[i, all_features.index('human_pos_x'):all_features.index("human_radius") + 1] = human_state[i]
            # self.graph.nodes['human'].data['h'] = human_tensor
            features = torch.cat([features, human_tensor], dim=0)

        # 如果存在障碍物节点，依次创建障碍物节点的特征张量，并将其与之前的特征张量拼接起来
        if obstacle_num > 0:
            obstacle_tensor = torch.zeros((obstacle_num, feature_dimensions))
            for i in range(obstacle_num):
                obstacle_tensor[i, all_features.index('obstacle')] = 1
                obstacle_tensor[i, all_features.index('obs_pos_x'):all_features.index("obs_radius") + 1] = obstacle_state[i]
            # self.graph.nodes['obstacle'].data['h'] = obstacle_tensor
            features = torch.cat([features, obstacle_tensor], dim=0)

        # 如果存在墙节点，依次创建墙节点的特征张量，并将其与之前的特征张量拼接起来
        if wall_num > 0:
            for i in range(wall_num):
                wall_tensor = torch.zeros((wall_num, feature_dimensions))
                wall_tensor[i, all_features.index('wall')] = 1
                wall_tensor[i, all_features.index('start_pos_x'):all_features.index("wall_radius") + 1] = wall_state[i]
            features = torch.cat([features, wall_tensor], dim=0)
        # self.graph.nodes['wall'].data['h'] = wall_tensor
        # features = torch.cat([robot_tensor, human_tensor, obstacle_tensor, wall_tensor], dim=0)

        ### build up edges for the social graph
        # add obstacle_to_robot edges
        # 初始化边的源节点、目标节点、边的类型和规范化因子的张量，都为空张量
        src_id = torch.Tensor([])
        dst_id = torch.Tensor([])
        edge_types = torch.Tensor([])
        edge_norm = torch.Tensor([])
        # add human_to_robot edges

        # 如果存在障碍物节点，创建从障碍物到机器人的边，其中包括源节点、目标节点、边的类型和规范化因子
        if obstacle_num > 0:
            src_obstacle_id = torch.tensor(range(obstacle_num)) + robot_num + human_num
            o2r_robot_id = torch.zeros_like(src_obstacle_id)
            o2r_edge_types = torch.ones_like(o2r_robot_id) * torch.LongTensor([self.rels.index('o2r')])
            o2r_edge_norm = torch.ones_like(o2r_robot_id) * (1.0)
            src_id = src_obstacle_id
            dst_id = o2r_robot_id
            edge_types = o2r_edge_types
            edge_norm = o2r_edge_norm

        # 如果存在人类节点，创建从人类到机器人的边，其中包括源节点、目标节点、边的类型和规范化因子
        if human_num > 0:
            src_human_id = torch.tensor(range(human_num)) + robot_num
            h2r_robot_id = torch.zeros_like(src_human_id)
            h2r_edge_types = torch.ones_like(h2r_robot_id) * torch.LongTensor([self.rels.index('h2r')])
            h2r_edge_norm = torch.ones_like(h2r_robot_id) * (1.0)
            src_id = torch.cat([src_id, src_human_id], dim=0)
            dst_id = torch.cat([dst_id, h2r_robot_id], dim=0)
            edge_types = torch.cat([edge_types, h2r_edge_types], dim=0)
            edge_norm = torch.cat([edge_norm, h2r_edge_norm], dim=0)

        # add wall_to_robot edges
        # 如果存在墙节点，创建从墙到机器人的边，其中包括源节点、目标节点、边的类型和规范化因子
        if wall_num > 0:
            src_wall_id = torch.tensor(range(wall_num)) + robot_num + human_num + obstacle_num
            w2r_robot_id = torch.zeros_like(src_wall_id)
            w2r_edge_types = torch.ones_like(w2r_robot_id) * torch.LongTensor([self.rels.index('w2r')])
            w2r_edge_norm = torch.ones_like(w2r_robot_id) * (1.0)

            src_id = torch.cat([src_id, src_wall_id], dim=0)
            dst_id = torch.cat([dst_id, w2r_robot_id], dim=0)
            edge_types = torch.cat([edge_types, w2r_edge_types], dim=0)
            edge_norm = torch.cat([edge_norm, w2r_edge_norm], dim=0)

        # 如果存在多个人类节点，创建从障碍物到人类和从墙到人类的边，其中包括源节点、目标节点、边的类型和规范化因子
        if human_num > 0:
            o2h_obstacle_id = None
            o2h_human_id = None
            w2h_wall_id = None
            w2h_human_id = None
            h2h_src_id = None
            h2h_dst_id = None
            for j in range(human_num):
                if j == 0:
                    i = j + robot_num
                    if obstacle_num > 0:
                    # add obstacle_to_human edges
                        o2h_obstacle_id = torch.tensor(range(obstacle_num)) + robot_num + human_num
                        o2h_human_id = torch.ones_like(src_obstacle_id) * i
                        o2h_edge_types = torch.ones_like(o2h_human_id) * torch.LongTensor([self.rels.index('o2h')])
                        o2h_edge_norm = torch.ones_like(o2h_human_id) * (1.0)
                        src_id = torch.cat([src_id, o2h_obstacle_id], dim=0)
                        dst_id = torch.cat([dst_id, o2h_human_id], dim=0)
                        edge_types = torch.cat([edge_types, o2h_edge_types], dim=0)
                        edge_norm = torch.cat([edge_norm, o2h_edge_norm], dim=0)

                    if wall_num > 0:
                        # add wall_to_human edges
                        w2h_wall_id = torch.tensor(range(wall_num)) + robot_num + human_num + obstacle_num
                        w2h_human_id = torch.ones_like(src_wall_id) * i
                        w2h_edge_types = torch.ones_like(w2h_human_id) * torch.LongTensor([self.rels.index('w2h')])
                        w2h_edge_norm = torch.ones_like(w2h_human_id) * (1.0)
                        src_id = torch.cat([src_id, w2h_wall_id], dim=0)
                        dst_id = torch.cat([dst_id, w2h_human_id], dim=0)
                        edge_types = torch.cat([edge_types, w2h_edge_types], dim=0)
                        edge_norm = torch.cat([edge_norm, w2h_edge_norm], dim=0)
                        # self.add_edges(src_wall_id, dst_human_id, ('wall', 'human', 'w_h'))

        # 如果存在多个人类节点，创建从障碍物到人类和从墙到人类的边，其中包括源节点、目标节点、边的类型和规范化因子
        if human_num > 1:
            # add human_to_human edges
            temp_src_id = []
            temp_dst_id = []
            for i in range(human_num):
                for k in range(j + 1, human_num):
                # a = (list(range(i)) + list(range(i + 1, human_num)))
                    temp_src_id.append(i+robot_num)
                    temp_src_id.append(k + robot_num)
                    temp_dst_id.append(k + robot_num)
                    temp_dst_id.append(i+robot_num)
            temp_src_id = torch.IntTensor(temp_src_id)
            temp_dst_id = torch.IntTensor(temp_dst_id)
            h2h_src_id = torch.IntTensor(temp_src_id)
            h2h_dst_id = torch.IntTensor(temp_dst_id)
            h2h_edge_types = torch.ones_like(h2h_src_id) * torch.LongTensor([self.rels.index('h2h')])
            h2h_edge_norm = torch.ones_like(h2h_src_id) * (1.0)
            src_id = torch.cat([src_id, h2h_src_id], dim=0)
            dst_id = torch.cat([dst_id, h2h_dst_id], dim=0)
            edge_types = torch.cat([edge_types, h2h_edge_types], dim=0)
            edge_norm = torch.cat([edge_norm, h2h_edge_norm], dim=0)
        edge_norm = edge_norm.unsqueeze(dim=1)

        # 在这部分建立图
        self.graph = dgl.graph((src_id, dst_id), num_nodes=total_node_num, idtype=torch.int32)
        self.graph.ndata['h'] = features
        self.graph.edata.update({'rel_type': edge_types, 'norm': edge_norm})
        # self.graph = dgl.add_self_loop(self.graph)

    def config_rvo_state(self, state):

        """
        将输入的状态数据转换成相对于机器人的局部坐标系的状态数据
        并使用RVO（Reciprocal Velocity Obstacles）算法进行交互推理
        最终返回转换后的机器人、人类、障碍物和墙的状态数据
        Transform the coordinate to agent-centric.
        Input tuple include robot state tensor and human state tensor.
        robot state tensor is of size (number, state_length)(for example 1*9)
        human state tensor is of size (number, state_length)(for example 5*5)
        obstacle state tensor is of size (number, state_length)(for example 3*3)
        wall state tensor is of size (number, state_length)(for example 4*4)
        """
        # 这是输入坐标state的含义
        # for robot
        # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta'
        #  0     1      2     3      4        5     6      7         8
        # for human
        #  'px', 'py', 'vx', 'vy', 'radius'
        #  0     1      2     3      4
        # for obstacle
        # 'px', 'py', 'radius'
        #  0     1     2
        # for wall
        # 'sx', 'sy', 'ex', 'ey', radius
        #  0     1     2     3

        # 使用断言确保输入的第一个状态数据 state[0] 是一个二维张量，而且第三个状态数据 state[2] 不为空
        assert len(state[0].shape) == 2
        assert state[2] is not None

        # 将输入的状态数据分别赋值给机器人、人类、障碍物和墙的状态变量
        robot_state = state[0]
        human_state = state[1]
        obstacle_state = state[2]
        wall_state = state[3]

        # 将机器人、人类、障碍物和墙的状态数据转换成 NumPy 数组
        robot_state_array = robot_state.numpy()
        human_state_array = human_state.numpy()
        obstacle_state_array = obstacle_state.numpy()
        wall_state_array = wall_state.numpy()

        # 调用 self.rvo_inter.config_vo_inf 方法，使用 RVO 算法进行交互推理，计算人类、障碍物和墙的 RVO 状态。
        # 返回的结果包括人类、障碍物和墙的新状态 rvo_human_state、rvo_obstacle_state 和 rvo_wall_state，以及其他一些信息，如最小碰撞时间 min_exp_time
        rvo_human_state, rvo_obstacle_state, rvo_wall_state, _, min_exp_time, _ = self.rvo_inter.config_vo_inf(robot_state_array,human_state_array, obstacle_state_array, wall_state_array)

        # 将计算得到的人类、障碍物和墙的新状态转换成 PyTorch 张量
        rvo_human_state = torch.Tensor(rvo_human_state)
        rvo_obstacle_state = torch.Tensor(rvo_obstacle_state)
        rvo_wall_state = torch.Tensor(rvo_wall_state)

        # 调用 self.world2robotframe 方法，将世界坐标系中的人类、障碍物和墙的状态数据转换成机器人局部坐标系中的状态数据
        # 将坐标系从世界坐标系下转到机器人坐标系下
        rvo_robot_state, rvo_human_state, rvo_obstacle_state, rvo_wall_state = self.world2robotframe(robot_state, rvo_human_state, rvo_obstacle_state, rvo_wall_state)

        # 返回转换后的机器人、人类、障碍物和墙的状态数据
        return rvo_robot_state, rvo_human_state, rvo_obstacle_state, rvo_wall_state

    def world2robotframe(self, robot_state, human_state, obstacle_state, wall_state):
        """
        这个函数的作用是将物品在世界坐标系下转到机器人坐标系下
        Args:
            robot_state:
            human_state:
            obstacle_state:
            wall_state:

        Returns:

        """
        dx = robot_state[:, 5] - robot_state[:, 0]
        dy = robot_state[:, 6] - robot_state[:, 1]
        dx = dx.unsqueeze(1)
        dy = dy.unsqueeze(1)
        # 获得距离robot的距离
        dg = torch.norm(torch.cat([dx, dy], dim=1), 2, dim=1, keepdim=True)
        # 获得robot的朝向
        rot = torch.atan2(dy, dx)
        # 获取机器人的速度
        robot_velocities = robot_state[:, 2:4]
        v_pref = robot_state[:, 7].unsqueeze(1)
        robot_radius = robot_state[:, 4].unsqueeze(1)
        cur_heading = (robot_state[:, 8].unsqueeze(1) - rot + np.pi) % (2 * np.pi) - np.pi
        new_robot_state = torch.cat((robot_velocities, dg, v_pref, cur_heading), dim=1)

        cos_rot = torch.cos(rot)
        sin_rot = torch.sin(rot)
        # 构造旋转矩阵
        transform_matrix = torch.cat((cos_rot, -sin_rot, sin_rot, cos_rot), dim=0).reshape(2, 2)
        if human_state.shape[0] != 0:
            human_state = torch.index_select(human_state, 1, torch.tensor([8,9,0,1,2,3,4,5,6,7,10]))
            temp = human_state[:,:8]
            temp = temp.reshape(human_state.shape[0],-1,2)
            temp = torch.matmul(temp, transform_matrix)
            human_state[:, :8] = temp.reshape(human_state.shape[0], -1)
            human_state[:, -1] = human_state[:, -1] + 0.3
        if obstacle_state.shape[0] !=0:
            obstacle_state = torch.index_select(obstacle_state, 1, torch.tensor([8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 10]))
            temp = obstacle_state[:,:8]
            temp = temp.reshape(obstacle_state.shape[0],-1,2)  # 这部分其实是机器人的坐标部分
            temp = torch.matmul(temp, transform_matrix)
            obstacle_state[:, :8] = temp.reshape(obstacle_state.shape[0], -1)
            obstacle_state[:, -1] = obstacle_state[:, -1] + 0.3
        if wall_state.shape[0] != 0:
            wall_state = torch.index_select(wall_state, 1, torch.tensor([8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7]))
            temp = wall_state[:,:10]
            temp = temp.reshape(wall_state.shape[0],-1,2)
            temp = torch.matmul(temp, transform_matrix)
            wall_state[:, :10] = temp.reshape(wall_state.shape[0], -1)
            wall_state = torch.cat((wall_state, 0.3 * torch.ones((wall_state.shape[0],1))), dim =1)
        return new_robot_state, human_state, obstacle_state, wall_state

    def build_up_graph_on_rvostate(self, data):
        """
        根据当前的rvostate建立图
        Args:
            data:采用tuple数据类型，分别为rvo_robot_state, rvo_human_state, rvo_obstacle_state, rvo_wall_state

        Returns:

        """
        if self.mode ==0:
            src_id = torch.Tensor([])
            dst_id = torch.Tensor([])
            # We create a map to store the types of the nodes. We'll use it to compute edges' types
            self.typeMap = dict()
            position_by_id = {}

            # Node Descriptor Table
            # 建立四种类型的节点，r表示robot，h表示human，o表示obstacle，w表示wall
            self.node_descriptor_header = ['r', 'h', 'o', 'w']

            # # Relations are integers
            # RelTensor = torch.LongTensor
            # # Normalization factors are floats
            # NormTensor = torch.Tensor
            # # Generate relations and number of relations integer (which is only accessed outside the class)
            # max_used_id = 0 # 0 for the robot
            # # Compute closest human distance
            # closest_human_distance = -1
            # Feature dimensions
            node_types_one_hot = ['robot', 'human', 'obstacle', 'wall']
            # robot_metric_features = ['rob_pos_x', 'robot_pos_y', 'rob_vel_x', 'rob_vel_y', 'rob_radius', 'rob_goal_x',
            #                          'rob_goal_y', 'rob_vel_pre', 'rob_ori']

            # 机器人矩阵的特征
            robot_metric_features = ['rob_vel_l', 'rob_vel_r', 'dis2goal', 'rob_vel_pre', 'rob_ori']

            # 人类矩阵的特征
            human_metric_features = ['human_vo_px', 'human_vo_py', 'human_vo_vl_x', 'human_v0_vl_y', 'human_vo_vr_x',
                                     'human_vo_vr_y', 'human_min_dis', 'human_exp_time', 'human_pos_x', 'human_pos_y',
                                     'human_radius']

            # 障碍物矩阵的特征
            obstacle_metric_features = ['obs_vo_px', 'obs_vo_py', 'obs_vo_vl_x', 'obs_v0_vl_y', 'obs_vo_vr_x',
                                        'obs_vo_vr_y', 'obs_min_dis', 'obs_exp_time', 'obs_pos_x', 'obs_pos_y',
                                        'obs_radius']

            # 墙矩阵的特征
            wall_metric_features = ['wall_vo_px', 'wall_vo_py', 'wall_vo_vl_x', 'wall_v0_vl_y', 'wall_vo_vr_x',
                                    'wall_vo_vr_y', 'wall_min_dis', 'wall_exp_time', 'wall_sx', 'wall_sy', 'wall_ex',
                                    'wall_ey', 'wall_radius']

            # 所有的特征为特征之和
            all_features = node_types_one_hot + robot_metric_features + human_metric_features + obstacle_metric_features + wall_metric_features

            # Copy input data
            # 复制输入数据
            self.data = data

            # 输入数据分别表示的含义
            robot_state, human_state, obstacle_state, wall_state = self.data

            # 特征的维度
            feature_dimensions = len(all_features)

            # 统计各个物品的数目
            # 机器人的数目
            robot_num = robot_state.shape[0]

            # 人类的数目
            if human_state is not None:
                human_num = human_state.shape[0]
            else:
                human_num = 0

            # 障碍物的数目
            if obstacle_state is not None:
                obstacle_num = obstacle_state.shape[0]
            else:
                obstacle_num = 0

            # 墙的数目
            if wall_state is not None:
                wall_num = wall_state.shape[0]
            else:
                wall_num = 0

            # 统计总的节点的数目
            total_node_num = robot_num + human_num + obstacle_num + wall_num
            # if total_node_num == 1:
            # fill data into the heterographgraph
            # data of the robot

            # robot的tensor
            robot_tensor = torch.zeros((robot_num, feature_dimensions))

            # 记录机器人的特征
            robot_tensor[0, all_features.index('robot')] = 1
            # 记录机器人的state特征
            robot_tensor[0, all_features.index('rob_vel_l'):all_features.index("rob_ori") + 1] = robot_state[0]
            # self.graph.nodes['robot'].data['h'] = robot_tensor
            # 将features记录为robot_tensor
            features = robot_tensor

            # 记录人类特征
            if human_num > 0:
                human_tensor = torch.zeros((human_num, feature_dimensions))
                for i in range(human_num):
                    human_tensor[i, all_features.index('human')] = 1
                    human_tensor[i, all_features.index('human_vo_px'):all_features.index("human_radius") + 1] = \
                    human_state[i]
                # self.graph.nodes['human'].data['h'] = human_tensor
                features = torch.cat([features, human_tensor], dim=0)

            # 记录障碍物特征
            if obstacle_num > 0:
                obstacle_tensor = torch.zeros((obstacle_num, feature_dimensions))
                for i in range(obstacle_num):
                    obstacle_tensor[i, all_features.index('obstacle')] = 1
                    obstacle_tensor[i, all_features.index('obs_vo_px'):all_features.index("obs_radius") + 1] = \
                        obstacle_state[i]
                # self.graph.nodes['obstacle'].data['h'] = obstacle_tensor
                features = torch.cat([features, obstacle_tensor], dim=0)

            # 记录墙特征
            if wall_num > 0:
                for i in range(wall_num):
                    wall_tensor = torch.zeros((wall_num, feature_dimensions))
                    wall_tensor[i, all_features.index('wall')] = 1
                    wall_tensor[i, all_features.index('wall_vo_px'):all_features.index("wall_radius") + 1] = wall_state[
                        i]
                features = torch.cat([features, wall_tensor], dim=0)
            # self.graph.nodes['wall'].data['h'] = wall_tensor
            # features = torch.cat([robot_tensor, human_tensor, obstacle_tensor, wall_tensor], dim=0)

            ### build up edges for the social graph
            # add obstacle_to_robot edges
            # 创建了一些空的张量来存储边的信息
            src_id = torch.Tensor([])  # 源节点 ID
            dst_id = torch.Tensor([])  # 目标节点 ID (dst_id)
            edge_types = torch.Tensor([])  # 边的类型 (edge_types)
            edge_norm = torch.Tensor([])  # 边的归一化值 (edge_norm)
            # add human_to_robot edges

            # 如果存在障碍物 (obstacle_num > 0)，则创建从障碍物到机器人的边
            if obstacle_num > 0:
                # 生成障碍物的源节点 ID (src_obstacle_id)
                # 这部分相当于是对起始点索引的编号
                src_obstacle_id = torch.tensor(range(obstacle_num)) + robot_num + human_num
                # 将目标节点 ID (o2r_robot_id) 设置为零向量，表示所有这些边都指向机器人
                # 终止点索引
                o2r_robot_id = torch.zeros_like(src_obstacle_id)
                # 这行代码为边的类型创建了一个张量，这部分是乘以边的权重
                o2r_edge_types = torch.ones_like(o2r_robot_id) * torch.LongTensor([self.rels.index('o2r')])
                # 为边的归一化值创建了一个张量，将其设置为所有边的归一化值都为 1.0，无权图
                o2r_edge_norm = torch.ones_like(o2r_robot_id) * (1.0)
                src_id = src_obstacle_id
                dst_id = o2r_robot_id
                edge_types = o2r_edge_types
                edge_norm = o2r_edge_norm

            # 如果存在人物 (human_num > 0)，则创建从人物到机器人的边
            if human_num > 0:
                # 这行代码生成人物的源节点 ID，并将其与机器人的数量相加，以确保人物的 ID 与机器人的 ID 不重叠
                src_human_id = torch.tensor(range(human_num)) + robot_num
                # 创建了目标节点ID，将其设置为与人物相同长度的零向量。这意味着所有的边都指向机器人
                h2r_robot_id = torch.zeros_like(src_human_id)
                # 为边的类型创建了一个张量，与之前类似，找到了关系列表中“人物到机器人”的索引，并创建了一个与目标节点 ID 相同长度的张量，并将其填充为相应的边类型
                h2r_edge_types = torch.ones_like(h2r_robot_id) * torch.LongTensor([self.rels.index('h2r')])
                # 为边的归一化值创建了一个张量，将其设置为所有边的归一化值都为 1.0，无权图
                h2r_edge_norm = torch.ones_like(h2r_robot_id) * (1.0)
                # 记录所有的边节点
                src_id = torch.cat([src_id, src_human_id], dim=0)
                dst_id = torch.cat([dst_id, h2r_robot_id], dim=0)
                edge_types = torch.cat([edge_types, h2r_edge_types], dim=0)
                edge_norm = torch.cat([edge_norm, h2r_edge_norm], dim=0)

            # add wall_to_robot edges
            # 如果存在墙壁，则创建从墙壁到机器人的边
            if wall_num > 0:
                # 这一行创建了墙壁节点的源节点 ID，相加，以确保源节点 ID 的唯一性
                src_wall_id = torch.tensor(range(wall_num)) + robot_num + human_num + obstacle_num
                # 创建了一个与墙壁数量相同的零向量，表示所有这些边都指向机器人
                w2r_robot_id = torch.zeros_like(src_wall_id)
                # 创建了墙壁的边权
                w2r_edge_types = torch.ones_like(w2r_robot_id) * torch.LongTensor([self.rels.index('w2r')])
                # 创建了归一化后的边权
                w2r_edge_norm = torch.ones_like(w2r_robot_id) * (1.0)

                src_id = torch.cat([src_id, src_wall_id], dim=0)
                dst_id = torch.cat([dst_id, w2r_robot_id], dim=0)
                edge_types = torch.cat([edge_types, w2r_edge_types], dim=0)
                edge_norm = torch.cat([edge_norm, w2r_edge_norm], dim=0)

            # 在检查是否存在人类（human_num > 0），如果存在，则为每个人类添加边
            if human_num > 0:
                # 遍历每个人类节点
                for j in range(human_num):
                    # 处理的是第一个人类节点
                    if j == 0:
                        # 确定了当前人类节点的 ID。人类节点的 ID 是在机器人数量的基础上递增的，以确保节点 ID 的唯一性
                        i = j + robot_num
                        # 如果存在障碍物，则为当前人类节点添加从障碍物到人类的边
                        if obstacle_num > 0:
                            # add obstacle_to_human edges
                            # 创建了障碍物节点的源节点 ID
                            o2h_obstacle_id = torch.tensor(range(obstacle_num)) + robot_num + human_num
                            # 为当前人类节点创建了目标节点 ID，将其设置为当前人类节点的 ID
                            o2h_human_id = torch.ones_like(src_obstacle_id) * i
                            # 创建从obstacle到human的权重
                            o2h_edge_types = torch.ones_like(o2h_human_id) * torch.LongTensor([self.rels.index('o2h')])
                            # 创建从obstacle到human的归一化权重
                            o2h_edge_norm = torch.ones_like(o2h_human_id) * (1.0)
                            src_id = torch.cat([src_id, o2h_obstacle_id], dim=0)
                            dst_id = torch.cat([dst_id, o2h_human_id], dim=0)
                            edge_types = torch.cat([edge_types, o2h_edge_types], dim=0)
                            edge_norm = torch.cat([edge_norm, o2h_edge_norm], dim=0)

                        # 如果存在墙壁，则为当前人类节点添加从墙壁到人类的边
                        if wall_num > 0:
                            # add wall_to_human edges
                            # 创建了墙壁节点的源节点 ID
                            w2h_wall_id = torch.tensor(range(wall_num)) + robot_num + human_num + obstacle_num
                            # 为当前人类节点创建了目标节点 ID，将其设置为当前人类节点的 ID
                            w2h_human_id = torch.ones_like(src_wall_id) * i
                            # 创建从wall到human的权重
                            w2h_edge_types = torch.ones_like(w2h_human_id) * torch.LongTensor([self.rels.index('w2h')])
                            # 创建从wall到human的归一化权重
                            w2h_edge_norm = torch.ones_like(w2h_human_id) * (1.0)
                            src_id = torch.cat([src_id, w2h_wall_id], dim=0)
                            dst_id = torch.cat([dst_id, w2h_human_id], dim=0)
                            edge_types = torch.cat([edge_types, w2h_edge_types], dim=0)
                            edge_norm = torch.cat([edge_norm, w2h_edge_norm], dim=0)
                            # self.add_edges(src_wall_id, dst_human_id, ('wall', 'human', 'w_h'))

            # 检查是否存在多个人类节点，因为只有当存在多个人类节点时才需要为它们之间建立边
            if human_num > 1:
                # add human_to_human edges
                temp_src_id = []
                temp_dst_id = []
                # 这两个嵌套的循环用于遍历所有可能的人类节点对，确保每对节点只添加一条边
                for i in range(human_num):
                    for k in range(j + 1, human_num):
                        # a = (list(range(i)) + list(range(i + 1, human_num)))
                        # 添加源节点和目标节点
                        temp_src_id.append(i + robot_num)
                        temp_src_id.append(k + robot_num)
                        temp_dst_id.append(k + robot_num)
                        temp_dst_id.append(i + robot_num)
                temp_src_id = torch.IntTensor(temp_src_id)
                temp_dst_id = torch.IntTensor(temp_dst_id)
                h2h_src_id = torch.IntTensor(temp_src_id)
                h2h_dst_id = torch.IntTensor(temp_dst_id)

                # 创建human到human的节点权重
                h2h_edge_types = torch.ones_like(h2h_src_id) * torch.LongTensor([self.rels.index('h2h')])
                # 创建归一化权重
                h2h_edge_norm = torch.ones_like(h2h_src_id) * (1.0)
                src_id = torch.cat([src_id, h2h_src_id], dim=0)
                dst_id = torch.cat([dst_id, h2h_dst_id], dim=0)
                edge_types = torch.cat([edge_types, h2h_edge_types], dim=0)
                edge_norm = torch.cat([edge_norm, h2h_edge_norm], dim=0)
            edge_norm = edge_norm.unsqueeze(dim=1)
            edge_norm = edge_norm.float()
            edge_types = edge_types.float()

            # 通过dgl库创建图
            self.graph = dgl.graph((src_id, dst_id), num_nodes=total_node_num, idtype=torch.int64)
            self.graph.ndata['h'] = features
            self.graph.edata.update({'rel_type': edge_types, 'norm': edge_norm})
        elif self.mode == 1:
            src_id = torch.Tensor([])
            dst_id = torch.Tensor([])
            # We create a map to store the types of the nodes. We'll use it to compute edges' types
            self.typeMap = dict()
            position_by_id = {}

            # Node Descriptor Table
            self.node_descriptor_header = ['r', 'h', 'o', 'w']

            # # Relations are integers
            # RelTensor = torch.LongTensor
            # # Normalization factors are floats
            # NormTensor = torch.Tensor
            # # Generate relations and number of relations integer (which is only accessed outside the class)
            # max_used_id = 0 # 0 for the robot
            # # Compute closest human distance
            # closest_human_distance = -1
            # Feature dimensions
            node_types_one_hot = ['robot', 'obstacle']
            # robot_metric_features = ['rob_pos_x', 'robot_pos_y', 'rob_vel_x', 'rob_vel_y', 'rob_radius', 'rob_goal_x',
            #                          'rob_goal_y', 'rob_vel_pre', 'rob_ori']
            robot_metric_features = ['rob_vel_l', 'rob_vel_r', 'dis2goal', 'rob_vel_pre', 'rob_ori']
            obstacle_metric_features = ['vo_px', 'vo_py', 'vo_vl_x', 'vo_vl_y', 'vo_vr_x', 'vo_vr_y', 'min_dis', 'exp_time']

            all_features = node_types_one_hot + robot_metric_features + obstacle_metric_features
            # Copy input data
            self.data = data
            robot_state, human_state, obstacle_state, wall_state = self.data
            feature_dimensions = len(all_features)
            robot_num = robot_state.shape[0]
            if human_state is not None:
                human_num = human_state.shape[0]
            else:
                human_num = 0
            if obstacle_state is not None:
                obstacle_num = obstacle_state.shape[0]
            else:
                obstacle_num = 0
            if wall_state is not None:
                wall_num = wall_state.shape[0]
            else:
                wall_num = 0
            total_node_num = robot_num + human_num + obstacle_num + wall_num
            # if total_node_num == 1:
            # fill data into the heterographgraph
            # data of the robot
            robot_tensor = torch.zeros((robot_num, feature_dimensions))
            robot_tensor[0, all_features.index('robot')] = 1
            robot_tensor[0, all_features.index('rob_vel_l'):all_features.index("rob_ori") + 1] = robot_state[0]
            # self.graph.nodes['robot'].data['h'] = robot_tensor
            features = robot_tensor
            if human_num > 0:
                human_tensor = torch.zeros((human_num, feature_dimensions))
                for i in range(human_num):
                    human_tensor[i, all_features.index('obstacle')] = 1
                    human_tensor[i, all_features.index('vo_px'):all_features.index("exp_time") + 1] = \
                    human_state[i,0:8]
                # self.graph.nodes['human'].data['h'] = human_tensor
                features = torch.cat([features, human_tensor], dim=0)

            if obstacle_num > 0:
                obstacle_tensor = torch.zeros((obstacle_num, feature_dimensions))
                for i in range(obstacle_num):
                    obstacle_tensor[i, all_features.index('obstacle')] = 1
                    obstacle_tensor[i, all_features.index('vo_px'):all_features.index("exp_time") + 1] = \
                        obstacle_state[i, 0:8]
                # self.graph.nodes['obstacle'].data['h'] = obstacle_tensor
                features = torch.cat([features, obstacle_tensor], dim=0)
            if wall_num > 0:
                for i in range(wall_num):
                    wall_tensor = torch.zeros((wall_num, feature_dimensions))
                    wall_tensor[i, all_features.index('obstacle')] = 1
                    wall_tensor[i, all_features.index('vo_px'):all_features.index("exp_time") + 1] = wall_state[i, 0:8]
                features = torch.cat([features, wall_tensor], dim=0)
            # self.graph.nodes['wall'].data['h'] = wall_tensor
            # features = torch.cat([robot_tensor, human_tensor, obstacle_tensor, wall_tensor], dim=0)

            ### build up edges for the social graph
            # add obstacle_to_robot edges
            src_id = torch.Tensor([])
            dst_id = torch.Tensor([])
            edge_types = torch.Tensor([])
            edge_norm = torch.Tensor([])
            # add human_to_robot edges

            if obstacle_num > 0:
                src_obstacle_id = torch.tensor(range(obstacle_num)) + robot_num + human_num
                o2r_robot_id = torch.zeros_like(src_obstacle_id)
                o2r_edge_types = torch.ones_like(o2r_robot_id) * torch.LongTensor([self.rels.index('o2r')])
                o2r_edge_norm = torch.ones_like(o2r_robot_id) * (1.0)
                src_id = src_obstacle_id
                dst_id = o2r_robot_id
                edge_types = o2r_edge_types
                edge_norm = o2r_edge_norm

            if human_num > 0:
                src_human_id = torch.tensor(range(human_num)) + robot_num
                h2r_robot_id = torch.zeros_like(src_human_id)
                h2r_edge_types = torch.ones_like(h2r_robot_id) * torch.LongTensor([self.rels.index('h2r')])
                h2r_edge_norm = torch.ones_like(h2r_robot_id) * (1.0)
                src_id = torch.cat([src_id, src_human_id], dim=0)
                dst_id = torch.cat([dst_id, h2r_robot_id], dim=0)
                edge_types = torch.cat([edge_types, h2r_edge_types], dim=0)
                edge_norm = torch.cat([edge_norm, h2r_edge_norm], dim=0)

            # add wall_to_robot edges
            if wall_num > 0:
                src_wall_id = torch.tensor(range(wall_num)) + robot_num + human_num + obstacle_num
                w2r_robot_id = torch.zeros_like(src_wall_id)
                w2r_edge_types = torch.ones_like(w2r_robot_id) * torch.LongTensor([self.rels.index('w2r')])
                w2r_edge_norm = torch.ones_like(w2r_robot_id) * (1.0)

                src_id = torch.cat([src_id, src_wall_id], dim=0)
                dst_id = torch.cat([dst_id, w2r_robot_id], dim=0)
                edge_types = torch.cat([edge_types, w2r_edge_types], dim=0)
                edge_norm = torch.cat([edge_norm, w2r_edge_norm], dim=0)

            if human_num > 0:
                for j in range(human_num):
                    if j == 0:
                        i = j + robot_num
                        if obstacle_num > 0:
                            # add obstacle_to_human edges
                            o2h_obstacle_id = torch.tensor(range(obstacle_num)) + robot_num + human_num
                            o2h_human_id = torch.ones_like(src_obstacle_id) * i
                            o2h_edge_types = torch.ones_like(o2h_human_id) * torch.LongTensor([self.rels.index('o2h')])
                            o2h_edge_norm = torch.ones_like(o2h_human_id) * (1.0)
                            src_id = torch.cat([src_id, o2h_obstacle_id], dim=0)
                            dst_id = torch.cat([dst_id, o2h_human_id], dim=0)
                            edge_types = torch.cat([edge_types, o2h_edge_types], dim=0)
                            edge_norm = torch.cat([edge_norm, o2h_edge_norm], dim=0)

                        if wall_num > 0:
                            # add wall_to_human edges
                            w2h_wall_id = torch.tensor(range(wall_num)) + robot_num + human_num + obstacle_num
                            w2h_human_id = torch.ones_like(src_wall_id) * i
                            w2h_edge_types = torch.ones_like(w2h_human_id) * torch.LongTensor([self.rels.index('w2h')])
                            w2h_edge_norm = torch.ones_like(w2h_human_id) * (1.0)
                            src_id = torch.cat([src_id, w2h_wall_id], dim=0)
                            dst_id = torch.cat([dst_id, w2h_human_id], dim=0)
                            edge_types = torch.cat([edge_types, w2h_edge_types], dim=0)
                            edge_norm = torch.cat([edge_norm, w2h_edge_norm], dim=0)
                            # self.add_edges(src_wall_id, dst_human_id, ('wall', 'human', 'w_h'))

            if human_num > 1:
                # add human_to_human edges
                temp_src_id = []
                temp_dst_id = []
                for i in range(human_num):
                    for k in range(j + 1, human_num):
                        # a = (list(range(i)) + list(range(i + 1, human_num)))
                        temp_src_id.append(i + robot_num)
                        temp_src_id.append(k + robot_num)
                        temp_dst_id.append(k + robot_num)
                        temp_dst_id.append(i + robot_num)
                temp_src_id = torch.IntTensor(temp_src_id)
                temp_dst_id = torch.IntTensor(temp_dst_id)
                h2h_src_id = torch.IntTensor(temp_src_id)
                h2h_dst_id = torch.IntTensor(temp_dst_id)
                h2h_edge_types = torch.ones_like(h2h_src_id) * torch.LongTensor([self.rels.index('h2h')])
                h2h_edge_norm = torch.ones_like(h2h_src_id) * (1.0)
                src_id = torch.cat([src_id, h2h_src_id], dim=0)
                dst_id = torch.cat([dst_id, h2h_dst_id], dim=0)
                edge_types = torch.cat([edge_types, h2h_edge_types], dim=0)
                edge_norm = torch.cat([edge_norm, h2h_edge_norm], dim=0)
            edge_norm = edge_norm.unsqueeze(dim=1)
            edge_norm = edge_norm.float()
            edge_types = edge_types.float()
            self.graph = dgl.graph((src_id, dst_id), num_nodes=total_node_num, idtype=torch.int64)
            self.graph.ndata['h'] = features
            self.graph.edata.update({'rel_type': edge_types, 'norm': edge_norm})
        elif self.mode ==2:
            src_id = torch.Tensor([])
            dst_id = torch.Tensor([])
            # We create a map to store the types of the nodes. We'll use it to compute edges' types
            self.typeMap = dict()
            position_by_id = {}

            # Node Descriptor Table
            self.node_descriptor_header = ['r', 'h', 'o', 'w']

            # # Relations are integers
            # RelTensor = torch.LongTensor
            # # Normalization factors are floats
            # NormTensor = torch.Tensor
            # # Generate relations and number of relations integer (which is only accessed outside the class)
            # max_used_id = 0 # 0 for the robot
            # # Compute closest human distance
            # closest_human_distance = -1
            # Feature dimensions
            node_types_one_hot = ['robot', 'human', 'obstacle', 'wall']
            # robot_metric_features = ['rob_pos_x', 'robot_pos_y', 'rob_vel_x', 'rob_vel_y', 'rob_radius', 'rob_goal_x',
            #                          'rob_goal_y', 'rob_vel_pre', 'rob_ori']
            robot_metric_features = ['rob_vel_l', 'rob_vel_r', 'dis2goal', 'rob_vel_pre', 'rob_ori']
            human_metric_features = ['human_vo_px', 'human_vo_py', 'human_pos_x', 'human_pos_y',
                                     'human_radius']
            obstacle_metric_features = ['obs_pos_x', 'obs_pos_y','obs_radius']
            wall_metric_features = ['wall_sx', 'wall_sy', 'wall_ex', 'wall_ey', 'wall_radius']
            all_features = node_types_one_hot + robot_metric_features + human_metric_features + obstacle_metric_features + wall_metric_features
            # Copy input data
            self.data = data
            robot_state, human_state, obstacle_state, wall_state = self.data
            feature_dimensions = len(all_features)
            robot_num = robot_state.shape[0]
            if human_state is not None:
                human_num = human_state.shape[0]
            else:
                human_num = 0
            if obstacle_state is not None:
                obstacle_num = obstacle_state.shape[0]
            else:
                obstacle_num = 0
            if wall_state is not None:
                wall_num = wall_state.shape[0]
            else:
                wall_num = 0
            total_node_num = robot_num + human_num + obstacle_num + wall_num
            # if total_node_num == 1:
            # fill data into the heterographgraph
            # data of the robot
            robot_tensor = torch.zeros((robot_num, feature_dimensions))
            robot_tensor[0, all_features.index('robot')] = 1
            robot_tensor[0, all_features.index('rob_vel_l'):all_features.index("rob_ori") + 1] = robot_state[0]
            # self.graph.nodes['robot'].data['h'] = robot_tensor
            features = robot_tensor
            if human_num > 0:
                human_tensor = torch.zeros((human_num, feature_dimensions))
                for i in range(human_num):
                    human_tensor[i, all_features.index('human')] = 1
                    human_tensor[i, all_features.index('human_vo_px'):all_features.index("human_vo_py") + 1] = \
                    human_state[i, 0:2]
                    human_tensor[i, all_features.index('human_pos_x'):all_features.index("human_radius") + 1] = \
                    human_state[i, -3:]
                # self.graph.nodes['human'].data['h'] = human_tensor
                features = torch.cat([features, human_tensor], dim=0)

            if obstacle_num > 0:
                obstacle_tensor = torch.zeros((obstacle_num, feature_dimensions))
                for i in range(obstacle_num):
                    obstacle_tensor[i, all_features.index('obstacle')] = 1
                    obstacle_tensor[i, all_features.index('obs_pos_x'):all_features.index("obs_radius") + 1] = \
                        obstacle_state[i,-3:]
                # self.graph.nodes['obstacle'].data['h'] = obstacle_tensor
                features = torch.cat([features, obstacle_tensor], dim=0)
            if wall_num > 0:
                for i in range(wall_num):
                    wall_tensor = torch.zeros((wall_num, feature_dimensions))
                    wall_tensor[i, all_features.index('wall')] = 1
                    wall_tensor[i, all_features.index('wall_sx'):all_features.index("wall_radius") + 1] = wall_state[
                        i, -5:]
                features = torch.cat([features, wall_tensor], dim=0)
            # self.graph.nodes['wall'].data['h'] = wall_tensor
            # features = torch.cat([robot_tensor, human_tensor, obstacle_tensor, wall_tensor], dim=0)

            ### build up edges for the social graph
            # add obstacle_to_robot edges
            src_id = torch.Tensor([])
            dst_id = torch.Tensor([])
            edge_types = torch.Tensor([])
            edge_norm = torch.Tensor([])
            # add human_to_robot edges

            if obstacle_num > 0:
                src_obstacle_id = torch.tensor(range(obstacle_num)) + robot_num + human_num
                o2r_robot_id = torch.zeros_like(src_obstacle_id)
                o2r_edge_types = torch.ones_like(o2r_robot_id) * torch.LongTensor([self.rels.index('o2r')])
                o2r_edge_norm = torch.ones_like(o2r_robot_id) * (1.0)
                src_id = src_obstacle_id
                dst_id = o2r_robot_id
                edge_types = o2r_edge_types
                edge_norm = o2r_edge_norm

            if human_num > 0:
                src_human_id = torch.tensor(range(human_num)) + robot_num
                h2r_robot_id = torch.zeros_like(src_human_id)
                h2r_edge_types = torch.ones_like(h2r_robot_id) * torch.LongTensor([self.rels.index('h2r')])
                h2r_edge_norm = torch.ones_like(h2r_robot_id) * (1.0)
                src_id = torch.cat([src_id, src_human_id], dim=0)
                dst_id = torch.cat([dst_id, h2r_robot_id], dim=0)
                edge_types = torch.cat([edge_types, h2r_edge_types], dim=0)
                edge_norm = torch.cat([edge_norm, h2r_edge_norm], dim=0)

            # add wall_to_robot edges
            if wall_num > 0:
                src_wall_id = torch.tensor(range(wall_num)) + robot_num + human_num + obstacle_num
                w2r_robot_id = torch.zeros_like(src_wall_id)
                w2r_edge_types = torch.ones_like(w2r_robot_id) * torch.LongTensor([self.rels.index('w2r')])
                w2r_edge_norm = torch.ones_like(w2r_robot_id) * (1.0)

                src_id = torch.cat([src_id, src_wall_id], dim=0)
                dst_id = torch.cat([dst_id, w2r_robot_id], dim=0)
                edge_types = torch.cat([edge_types, w2r_edge_types], dim=0)
                edge_norm = torch.cat([edge_norm, w2r_edge_norm], dim=0)

            if human_num > 0:
                for j in range(human_num):
                    if j == 0:
                        i = j + robot_num
                        if obstacle_num > 0:
                            # add obstacle_to_human edges
                            o2h_obstacle_id = torch.tensor(range(obstacle_num)) + robot_num + human_num
                            o2h_human_id = torch.ones_like(src_obstacle_id) * i
                            o2h_edge_types = torch.ones_like(o2h_human_id) * torch.LongTensor([self.rels.index('o2h')])
                            o2h_edge_norm = torch.ones_like(o2h_human_id) * (1.0)
                            src_id = torch.cat([src_id, o2h_obstacle_id], dim=0)
                            dst_id = torch.cat([dst_id, o2h_human_id], dim=0)
                            edge_types = torch.cat([edge_types, o2h_edge_types], dim=0)
                            edge_norm = torch.cat([edge_norm, o2h_edge_norm], dim=0)

                        if wall_num > 0:
                            # add wall_to_human edges
                            w2h_wall_id = torch.tensor(range(wall_num)) + robot_num + human_num + obstacle_num
                            w2h_human_id = torch.ones_like(src_wall_id) * i
                            w2h_edge_types = torch.ones_like(w2h_human_id) * torch.LongTensor([self.rels.index('w2h')])
                            w2h_edge_norm = torch.ones_like(w2h_human_id) * (1.0)
                            src_id = torch.cat([src_id, w2h_wall_id], dim=0)
                            dst_id = torch.cat([dst_id, w2h_human_id], dim=0)
                            edge_types = torch.cat([edge_types, w2h_edge_types], dim=0)
                            edge_norm = torch.cat([edge_norm, w2h_edge_norm], dim=0)
                            # self.add_edges(src_wall_id, dst_human_id, ('wall', 'human', 'w_h'))

            if human_num > 1:
                # add human_to_human edges
                temp_src_id = []
                temp_dst_id = []
                for i in range(human_num):
                    for k in range(j + 1, human_num):
                        # a = (list(range(i)) + list(range(i + 1, human_num)))
                        temp_src_id.append(i + robot_num)
                        temp_src_id.append(k + robot_num)
                        temp_dst_id.append(k + robot_num)
                        temp_dst_id.append(i + robot_num)
                temp_src_id = torch.IntTensor(temp_src_id)
                temp_dst_id = torch.IntTensor(temp_dst_id)
                h2h_src_id = torch.IntTensor(temp_src_id)
                h2h_dst_id = torch.IntTensor(temp_dst_id)
                h2h_edge_types = torch.ones_like(h2h_src_id) * torch.LongTensor([self.rels.index('h2h')])
                h2h_edge_norm = torch.ones_like(h2h_src_id) * (1.0)
                src_id = torch.cat([src_id, h2h_src_id], dim=0)
                dst_id = torch.cat([dst_id, h2h_dst_id], dim=0)
                edge_types = torch.cat([edge_types, h2h_edge_types], dim=0)
                edge_norm = torch.cat([edge_norm, h2h_edge_norm], dim=0)
            edge_norm = edge_norm.unsqueeze(dim=1)
            edge_norm = edge_norm.float()
            edge_types = edge_types.float()
            self.graph = dgl.graph((src_id, dst_id), num_nodes=total_node_num, idtype=torch.int64)
            self.graph.ndata['h'] = features
            self.graph.edata.update({'rel_type': edge_types, 'norm': edge_norm})


