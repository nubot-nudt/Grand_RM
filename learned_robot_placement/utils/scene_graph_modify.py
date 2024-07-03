import torch
import numpy as np
import dgl
from dgl import DGLGraph
from dgl.nn.pytorch.conv.relgraphconv import RelGraphConv
from scipy.spatial.transform import Rotation


class SceneGraph():
    def __init__(self, curr_goal_pos, curr_goal_quat, rotated_bboxes, ori_bboxes, obj_bboxes_indexes):
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
        self.obstacle_len = rotated_bboxes.shape[0]
        self.obj_bboxes_indexes = obj_bboxes_indexes

        robot_pos_rad_bbox = np.array([0, 0, 0, 0, 0.5, 0.5, 1])
        goal_rad_bboxes, pos_rad_bboxes = self.get_pos_rad(curr_goal_pos, curr_goal_quat, rotated_bboxes, ori_bboxes)

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

        # ToDo:第一步获取最短距离
        # cur_goal_min_dis = torch.norm(cur_goal_pos[0, :2])
        # objects_min_dis = torch.zeros(rotated_bboxes.shape[0], 1)
        #
        # for i in range(objects_min_dis.shape[0]):
        #     objects_min_dis[i, 0] = min(torch.norm(rotated_bboxes[i, 0:2]), torch.norm(rotated_bboxes[i, 2:4]))

        # ToDo:第二步转换为图的数据结构
        self.build_graph(robot_pos_rad_bbox, goal_rad_bboxes, pos_rad_bboxes, obj_bboxes_indexes)

    def get_pos_rad(self, curr_goal_pos, curr_goal_quat, cur_bboxes, ori_bboxes):
        len_ = cur_bboxes.shape[0]
        goal_rad_bboxes = np.zeros([1, 7])
        goal_rad_bboxes[:, :3] = curr_goal_pos
        goal_rad_bboxes[:, 3] = self.quaternion_to_angle(curr_goal_quat[0])
        goal_rad_bboxes[:, 4:] = ori_bboxes[self.obj_bboxes_indexes[0][0]]

        ori_bboxes = np.delete(ori_bboxes, self.obj_bboxes_indexes[0][0], axis=0)

        pos_rad_bboxes = np.zeros([len_, 7])

        pos_rad_bboxes[:, :2] = (cur_bboxes[:, :2] + cur_bboxes[:, 2:4]) / 2
        pos_rad_bboxes[:, 2] = cur_bboxes[:, 4]
        pos_rad_bboxes[:, 3] = cur_bboxes[:, 5]
        pos_rad_bboxes[:, 4:] = ori_bboxes[:, :3]

        return goal_rad_bboxes, pos_rad_bboxes

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

    def quaternion_to_angle(self, quaternion):
        rotation = Rotation.from_quat(quaternion)
        euler_angles = rotation.as_euler('xzy')
        angle_rad = euler_angles[0]
        angle_rad = -angle_rad + np.pi
        angle_rad = angle_rad % (2 * np.pi)
        # print("quaternion=%s, angle_rad=%s"%(quaternion, angle_rad))
        if angle_rad > np.pi:
            angle_rad -= 2 * np.pi
        elif angle_rad < -np.pi:
            angle_rad += 2 * np.pi
        return angle_rad

    def build_graph(self, robot_pos_rad_bbox, goal_rad_bboxes, pos_rad_bboxes, obj_bboxes_indexes):
        # 定义节点特征（使用机器人坐标系）
        robot_feature = self.create_node_feature(robot_pos_rad_bbox.tolist(), 'robot')
        table_feature = self.create_node_feature(pos_rad_bboxes[-1, :].tolist(), 'table')
        target_feature = self.create_node_feature(goal_rad_bboxes[0].tolist(), 'target_object')
        distractor_features = [self.create_node_feature(d.tolist(), 'distractor_object') for d in
                               pos_rad_bboxes[:-1, :]]

        # 汇总所有节点特征
        features = []

        # 添加 robot_feature
        features.append(robot_feature)

        # 添加 table_feature
        features.append(table_feature)

        # 添加 target_feature
        features.append(target_feature)

        # 添加 distractor_features
        features.extend(distractor_features)

        # print("robot_feature=%s" % (robot_feature))
        # print("table_feature=%s" % (table_feature))
        # print("target_feature=%s" % (target_feature))
        # print("distractor_features=%s" % (distractor_features))
        #
        # print("features=")
        # print(features)
        # input("==================")

        # 创建图并添加节点
        g = dgl.graph(([], []))
        g.add_nodes(len(features))

        # 添加节点特征
        g.ndata['feat'] = torch.tensor(features, dtype=torch.float)

        # 定义边
        edge_list = []
        edge_features = []

        # 机器人与所有物品之间的边
        for i in range(1, len(features)):
            edge_list.append((0, i))  # 机器人到物品
            edge_features.append(self.create_edge_feature(features[0], features[i]))

        # 物品与物品之间的边，物品与桌子之间的边
        for i in range(1, len(features)):
            for j in range(i + 1, len(features)):
                edge_list.append((i, j))
                edge_features.append(self.create_edge_feature(features[i], features[j]))

        # 添加边
        src, dst = zip(*edge_list)
        g.add_edges(src, dst)

        # print("edge_features=")
        # print(edge_features)

        # 添加边特征
        g.edata['feat'] = torch.tensor(edge_features, dtype=torch.float)

        # print("g=")
        # print(g)
        # input("测试图")
        self.graph = g

    def create_node_feature(self, entity, entity_type):
        if entity_type == 'robot':
            feature = entity + [0]  # 0表示机器人
        elif entity_type == 'table':
            feature = entity + [1]  # 1表示桌子
        elif entity_type == 'target_object':
            feature = entity + [2]  # 2表示目标物品
        else:  # distractor_object
            feature = entity + [3]  # 3表示干扰物品
        return feature

    def create_edge_feature(self, node1, node2):
        x1, y1 = node1[:2]
        x2, y2 = node2[:2]
        distance = np.linalg.norm([x2 - x1, y2 - y1])
        return [distance]



