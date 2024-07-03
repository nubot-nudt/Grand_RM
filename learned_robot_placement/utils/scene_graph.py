import torch
import numpy as np
import dgl
from dgl import DGLGraph
from dgl.nn.pytorch.conv.relgraphconv import RelGraphConv


class SceneGraph():
    def __init__(self, cur_goal_pos, cur_goal_quat, rotated_bboxes=None, obj_bboxes_indexes=None):
        """
        建立图神经网络
        Args:
            cur_goal_pos: 当前目标位置
            cur_goal_quat: 当前目标位置
            rotated_bboxes: 转换到机器人坐标系下的bboxes
            obj_bboxes_indexes: 对应的索引
        """
        super(SceneGraph, self).__init__()
        self.graph = None
        self.data = None
        self.robot_visible = False
        self._device = 'cpu'

        # 包含不同类型关系的列表，其中每个元素表示一个关系类型
        # 这样做的目的是为了使得图在后续的处理中能够区分不同类型的边，即对应于边的权重
        # 通过使用这些关系类型，代码可以正确地标识和区分从不同实体到机器人的边，从而构建了一个更加复杂的社交图
        # 'o2r'：表示抓取物品旁的其他物品到机器人的关系，obstacle2robot，类似于circle obstacle
        # 'c2r'：容器本身作为障碍物，container2robot，类似于line obstacle
        # 's2r'：容器附近的障碍物，surrounding2robot
        self.rels = ['o2r', 'c2r', 's2r']
        self.mode = 0

        # print("cur_goal_pos=%s" % (cur_goal_pos))
        # print("rotated_bboxes=%s" % (rotated_bboxes))

        # print("cur_goal_pos=%s" % (cur_goal_pos))
        # print("cur_goal_quat=%s" % (cur_goal_quat))
        # print("rotated_bboxes=%s" % (rotated_bboxes))
        # input("输入数据")

        # ToDo:第一步获取最短距离
        cur_goal_min_dis = torch.norm(cur_goal_pos[0, :2])

        if rotated_bboxes is not None:
            objects_min_dis = torch.zeros(rotated_bboxes.shape[0], 1)

            for i in range(objects_min_dis.shape[0]):
                objects_min_dis[i, 0] = min(torch.norm(rotated_bboxes[i, 0:2]), torch.norm(rotated_bboxes[i, 2:4]))
        else:
            objects_min_dis = None

        # print("cur_goal_min_dis=%s"%(cur_goal_min_dis))
        # print("objects_min_dis=%s"%(objects_min_dis))
        # input("输入数据")

        # ToDo:第二步转换为图的数据结构
        self.build_up_graph_on_local_state(cur_goal_pos, cur_goal_quat, cur_goal_min_dis, rotated_bboxes, obj_bboxes_indexes, objects_min_dis)

    def rotate_state(self, base_tf, theta, obj_bboxes):
        """
        将其他物品的状态从世界坐标系转到机器人坐标系下
        Args:
            base_tf:当前robot的坐标系
            theta:当前robot的朝向
            obj_bboxes:在世界坐标系下的bbox，obj_bboxes的维度为[n, 5]其中0:4为bbox，4为高度z，5为朝向
            obj_bboxes_indexes:

        Returns:
        Transform the coordinate to agent-centric.
        Input tuple include robot state tensor and human state tensor.
        robot state tensor is of size (number, state_length)(for example 1*9)
        obstacle state tensor is of size (number, state_length)(for example 3*4)
        container state tensor is of size (number, state_length)(for example 5*4)
        surrounding state tensor is of size (number, state_length)(for example 4*4)
        """

        """第一步：将物体转移到机器人坐标系下"""
        inv_base_tf = torch.linalg.inv(base_tf)

        """第二步：分别获取障碍物坐标的索引值"""
        local_obj_bboxes = obj_bboxes.clone()

        total_num = len(obj_bboxes)

        for obj_num in range(total_num):
            # 获取当前对象轴对齐边界框的最小的 XY 顶点坐标
            min_xy_vertex = torch.hstack(
                (obj_bboxes[obj_num, 0:2], torch.tensor([0.0, 1.0], device=self._device))).T
            # 获取当前对象轴对齐边界框的最大的 XY 顶点坐标
            max_xy_vertex = torch.hstack(
                (obj_bboxes[obj_num, 2:4], torch.tensor([0.0, 1.0], device=self._device))).T

            # 通过矩阵乘法将最小顶点坐标转换到机器人参考框架中，并更新为新的最小顶点坐标
            new_min_xy_vertex = torch.matmul(inv_base_tf, min_xy_vertex)[0:2].T.squeeze()
            # 通过矩阵乘法将最大顶点坐标转换到机器人参考框架中，并更新为新的最大顶点坐标
            new_max_xy_vertex = torch.matmul(inv_base_tf, max_xy_vertex)[0:2].T.squeeze()

            # 记录结果
            local_obj_bboxes[obj_num, 0:4] = torch.hstack((new_min_xy_vertex, new_max_xy_vertex))
            # 设置高度差
            local_obj_bboxes[obj_num, 5] = self.limit_angle(obj_bboxes[obj_num, 5] - theta)

        return local_obj_bboxes

    def limit_angle(self, angle):
        # 将角度限制在 -π 到 π 之间
        while angle < -np.pi:
            angle += 2 * np.pi
        while angle > np.pi:
            angle -= 2 * np.pi
        return angle

    def build_up_graph_on_local_state(self, cur_goal_pos, cur_goal_quat, cur_goal_min_dis, rotated_bboxes, obj_bboxes_indexes, objects_min_dis):
        """

        Args:
            base_tf: 机器人当前的坐标系
            theta: 机器人当前的朝向
            obj_bboxes: 物品的bboxes
            obj_bboxes_indexes: 物品的索引值

        Returns:

        """

        src_id = torch.Tensor([])
        dst_id = torch.Tensor([])
        # We create a map to store the types of the nodes. We'll use it to compute edges' types
        self.typeMap = dict()
        position_by_id = {}

        # Node Descriptor Table
        # 建立四种类型的节点，r表示robot，h表示human，o表示obstacle，w表示wall
        self.node_descriptor_header = ['g', 'o', 'c', 's']

        # Relations are integers
        node_types_one_hot = ['goal', 'obstacle', 'container', 'surrounding']

        # 机器人矩阵的特征，可以选择的变量，机器人的速度
        goal_metric_features = ['goal_x', 'goal_y', 'goal_z', 'goal_quat1', 'goal_quat2', 'goal_quat3', 'goal_quat4', 'goal_min_dis']

        # 桌面上其他障碍物矩阵的特征
        obstacle_metric_features = ['obs_min_x', 'obs_min_y', 'obs_max_x', 'obs_max_y', 'obs_z', 'obs_ori', 'obs_min_dis']

        # 障碍物矩阵的特征
        container_metric_features = ['con_min_x', 'con_min_y', 'con_max_x', 'con_max_y', 'con_z', 'con_ori', 'con_min_dis']

        # 墙矩阵的特征
        surrounding_features = ['surr_min_x', 'surr_min_y', 'surr_max_x', 'surr_max_y', 'surr_z', 'surr_ori', 'surr_min_dis']

        # 所有的特征为特征之和
        all_features = node_types_one_hot + goal_metric_features + obstacle_metric_features + container_metric_features + surrounding_features

        # Copy input data
        # 复制输入数据
        self.obj_bboxes = rotated_bboxes

        # 输入数据分别表示的含义
        # print("base_tf[:3, 3]=%s, theta=%s, dist2goal=%s"%(
        #     base_tf[:3, 3], theta, dist2goal
        # ))
        # print("cur_goal_pos=%s, cur_goal_quat=%s, cur_goal_min_dis=%s"%(
        #     cur_goal_pos, cur_goal_quat, cur_goal_min_dis
        # ))
        # print("self.obj_bboxes[obj_bboxes_indexes[0], :]=%s, objects_min_dis[obj_bboxes_indexes[0]]=%s" % (
        #     self.obj_bboxes[obj_bboxes_indexes[0], :], objects_min_dis[obj_bboxes_indexes[0]]
        # ))
        # print("self.obj_bboxes[obj_bboxes_indexes[1], :]=%s, objects_min_dis[obj_bboxes_indexes[1]]=%s" % (
        #     self.obj_bboxes[obj_bboxes_indexes[1], :], objects_min_dis[obj_bboxes_indexes[1]]
        # ))

        goal_state = torch.hstack([cur_goal_pos[0], cur_goal_quat[0], cur_goal_min_dis])

        if self.obj_bboxes is not None:
            obstacle_state = torch.hstack([self.obj_bboxes[obj_bboxes_indexes[0], :], objects_min_dis[obj_bboxes_indexes[0]]])
            container_state = torch.hstack([self.obj_bboxes[obj_bboxes_indexes[1], :], objects_min_dis[obj_bboxes_indexes[1]]])
            surrounding_state = torch.hstack([self.obj_bboxes[obj_bboxes_indexes[2], :], objects_min_dis[obj_bboxes_indexes[2]]])
        else:
            obstacle_state = None
            container_state = None
            surrounding_state = None

            # print("Debug2:self.obj_bboxes=%s"%(self.obj_bboxes))
        # print("Debug2:goal_state=%s, obstacle_state=%s, container_state=%s,"
        #       "surrounding_state=%s"%(goal_state, obstacle_state, container_state, surrounding_state))
        # input("测试")

        # 特征的维度
        feature_dimensions = len(all_features)

        # 统计各个物品的数目
        # 机器人的数目
        goal_num = 1

        # 障碍物的数目
        if obstacle_state is not None:
            obstacle_num = obstacle_state.shape[0]
        else:
            obstacle_num = 0

        # 容器的数目，这是一个广义的容器，因为对于桌子只有一个长方体，对于沙发有三个长方体，对于冰箱有四个长方体
        if container_state is not None:
            container_num = container_state.shape[0]
        else:
            container_num = 0

        # 容器周围障碍物的数目
        if surrounding_state is not None:
            surrounding_num = surrounding_state.shape[0]
        else:
            surrounding_num = 0

        # print('obstacle_num=%d, container_num=%d, surrounding_num=%d'%(
        #     obstacle_num, container_num, surrounding_num
        # ))

        # 统计总的节点的数目
        total_node_num = goal_num + obstacle_num + container_num + surrounding_num

        # robot的tensor
        goal_tensor = torch.zeros((goal_num, feature_dimensions))

        # 记录机器人的特征
        goal_tensor[0, all_features.index('goal')] = 1
        # 记录机器人的state特征
        goal_tensor[0, all_features.index('goal_x'):all_features.index('goal_min_dis') + 1] = goal_state
        # 将features记录为goal_tensor
        features = goal_tensor

        # input("goal_tensor=%s"%(goal_tensor))

        # 记录障碍物特征
        if obstacle_num > 0:
            obstacle_tensor = torch.zeros((obstacle_num, feature_dimensions))
            for i in range(obstacle_num):
                obstacle_tensor[i, all_features.index('obstacle')] = 1
                obstacle_tensor[i, all_features.index('obs_min_x'):all_features.index("obs_min_dis") + 1] = \
                    obstacle_state[i, :]

            # self.graph.nodes['human'].data['h'] = human_tensor
            features = torch.cat([features, obstacle_tensor], dim=0)
        # print("obstacle_tensor=%s" % (obstacle_tensor))

        # 记录障碍物特征
        if container_num > 0:
            container_tensor = torch.zeros((container_num, feature_dimensions))
            for i in range(container_num):
                container_tensor[i, all_features.index('container')] = 1
                container_tensor[i, all_features.index('con_min_x'):all_features.index("con_min_dis") + 1] = \
                    container_state[i, :]
            # self.graph.nodes['obstacle'].data['h'] = obstacle_tensor
            features = torch.cat([features, container_tensor], dim=0)
        # print("container_tensor=%s" % (container_tensor))

        # 记录墙特征
        if surrounding_num > 0:
            surrounding_tensor = torch.zeros((surrounding_num, feature_dimensions))
            for i in range(surrounding_num):
                surrounding_tensor[i, all_features.index('surrounding')] = 1
                surrounding_tensor[i, all_features.index('surr_min_x'):all_features.index("surr_min_dis") + 1] = \
                    surrounding_state[i, :]
            features = torch.cat([features, surrounding_tensor], dim=0)
        # print("surrounding_tensor=%s" % (surrounding_tensor))

        # input("测试节点特征")
        # Todo:features测试通过
        # print("features=%s" % (features))

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
            src_obstacle_id = torch.tensor(range(obstacle_num)) + goal_num
            # 将目标节点 ID (o2r_robot_id) 设置为零向量，表示所有这些边都指向机器人
            # 终止点索引
            o2r_robot_id = torch.zeros_like(src_obstacle_id)
            # 这行代码为边的类型创建了一个张量，这部分是乘以边的权重
            o2r_edge_types = torch.ones_like(o2r_robot_id) * torch.LongTensor([self.rels.index('o2r')])
            # 为边的归一化值创建了一个张量，将其设置为所有边的归一化值都为 1.0，无权图
            o2r_edge_norm = torch.ones_like(o2r_robot_id) * (1.0)

            # print("obstacle:src_obstacle_id=%s, o2r_robot_id=%s, o2r_edge_types=%s, o2r_edge_norm=%s" % (
            #     src_obstacle_id, o2r_robot_id, o2r_edge_types, o2r_edge_norm
            # ))

            src_id = src_obstacle_id
            dst_id = o2r_robot_id
            edge_types = o2r_edge_types
            edge_norm = o2r_edge_norm

        # 如果存在人物 (human_num > 0)，则创建从人物到机器人的边
        if container_num > 0:
            # 这行代码生成人物的源节点 ID，并将其与机器人的数量相加，以确保人物的 ID 与机器人的 ID 不重叠
            src_container_id = torch.tensor(range(container_num)) + goal_num + obstacle_num
            # 创建了目标节点ID，将其设置为与人物相同长度的零向量。这意味着所有的边都指向机器人
            c2r_robot_id = torch.zeros_like(src_container_id)
            # 为边的类型创建了一个张量，与之前类似，找到了关系列表中“人物到机器人”的索引，并创建了一个与目标节点 ID 相同长度的张量，并将其填充为相应的边类型
            c2r_edge_types = torch.ones_like(c2r_robot_id) * torch.LongTensor([self.rels.index('c2r')])
            # 为边的归一化值创建了一个张量，将其设置为所有边的归一化值都为 1.0，无权图
            c2r_edge_norms = torch.ones_like(c2r_robot_id) * (1.0)

            # print("container:src_container_id=%s, c2r_robot_id=%s, c2r_edge_types=%s, c2r_edge_norms=%s"%(
            #     src_container_id, c2r_robot_id, c2r_edge_types, c2r_edge_norms
            # ))

            # 记录所有的边节点
            src_id = torch.cat([src_id, src_container_id], dim=0)
            dst_id = torch.cat([dst_id, c2r_robot_id], dim=0)
            edge_types = torch.cat([edge_types, c2r_edge_types], dim=0)
            edge_norm = torch.cat([edge_norm, c2r_edge_norms], dim=0)

        # add wall_to_robot edges
        # 如果存在墙壁，则创建从墙壁到机器人的边
        if surrounding_num > 0:
            # 这一行创建了墙壁节点的源节点 ID，相加，以确保源节点 ID 的唯一性
            src_surrounding_id = torch.tensor(range(surrounding_num)) + goal_num + obstacle_num + container_num
            # 创建了一个与墙壁数量相同的零向量，表示所有这些边都指向机器人
            s2r_robot_id = torch.zeros_like(src_surrounding_id)
            # 创建了墙壁的边权
            s2r_edge_types = torch.ones_like(s2r_robot_id) * torch.LongTensor([self.rels.index('s2r')])
            # 创建了归一化后的边权
            s2r_edge_norm = torch.ones_like(s2r_robot_id) * (1.0)

            # print("obstacle:src_container_id=%s, c2r_robot_id=%s, c2r_edge_types=%s, c2r_edge_norms=%s"%(
            #     src_container_id, c2r_robot_id, c2r_edge_types, c2r_edge_norms
            # ))

            src_id = torch.cat([src_id, src_surrounding_id], dim=0)
            dst_id = torch.cat([dst_id, s2r_robot_id], dim=0)
            edge_types = torch.cat([edge_types, s2r_edge_types], dim=0)
            edge_norm = torch.cat([edge_norm, s2r_edge_norm], dim=0)

        edge_norm = edge_norm.unsqueeze(dim=1)
        edge_norm = edge_norm.float()
        edge_types = edge_types.float()

        # print("edge_types=%s, edge_norm=%s, src_id=%s, dst_id=%s, total_node_num=%s"%(
        #     edge_types, edge_norm, src_id, dst_id, total_node_num))

        # 通过dgl库创建图
        self.graph = dgl.graph((src_id, dst_id), num_nodes=total_node_num, idtype=torch.int64)
        # input("type(self.graph)=%s"%(type(self.graph)))
        self.graph.ndata['h'] = features
        self.graph.edata.update({'rel_type': edge_types, 'norm': edge_norm})

    def print_graph_info(self, state_graph):
        # 打印图的信息
        # 打印节点特征信息
        # print("\nNode data schemes:")
        # for key, value in state_graph.ndata.items():
        #     print(f"Key: {key}, Shape: {value.shape}, Data type: {value.dtype}")

        # 打印节点的具体特征信息
        print("\nNode feature 'h':")
        print(state_graph.ndata['h'])

        # 打印边特征信息
        print("\nEdge data schemes:")
        for key, value in state_graph.edata.items():
            print(f"Key: {key}, Shape: {value.shape}, Data type: {value.dtype}")
        input("测试graph的输入")

        # 如果有边，打印边的具体特征信息
        # if state_graph.number_of_edges() > 0:
        #     print("\nEdge feature 'rel_type':")
        #     print(state_graph.edata['rel_type'])
        #     print("\nEdge feature 'norm':")
        #     print(state_graph.edata['norm'])


