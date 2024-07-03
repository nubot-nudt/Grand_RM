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

import torch
import numpy as np
from learned_robot_placement.tasks.base.rl_task import RLTask
from learned_robot_placement.handlers.fetchWBhandler import FetchWBHandler
from omni.isaac.core.objects.cone import VisualCone
from omni.isaac.core.prims import GeometryPrimView
from learned_robot_placement.tasks.utils.pinoc_fetch_utils import PinFetchIKSolver  # For IK
from learned_robot_placement.tasks.utils import scene_utils
from omni.isaac.isaac_sensor import _isaac_sensor
from mushroom_rl.core.logger.console_logger import ConsoleLogger
from PIL import Image
from scipy.spatial.transform import Rotation

# from omni.isaac.core.utils.prims import get_prim_at_path
# from omni.isaac.core.utils.prims import create_prim
# from omni.isaac.core.utils.stage import add_reference_to_stage

from omni.isaac.core.utils.torch.maths import torch_rand_float, tensor_clamp
from omni.isaac.core.utils.torch.rotations import euler_angles_to_quats, quat_diff_rad
from omni.isaac.debug_draw import _debug_draw
from scipy.spatial.transform import Rotation
from learned_robot_placement.tasks.utils.transformations import *
from learned_robot_placement.tasks.utils.transformations import _AXES2TUPLE
from learned_robot_placement.utils.scene_graph import SceneGraph
from learned_robot_placement.policy.graph_model import DGL_RGCN_RL
import math
import time

# Base placement environment for fetching a target object among clutter
class FetchMultiObjFetchingTask(RLTask):
    def __init__(
            self,
            name,
            sim_config,
            env
    ) -> None:
        self.last_q = None
        self.last_states = None
        self.next_action = None
        self.robot_state = None
        self.round_loc = None
        self.states = None
        self.draw = _debug_draw.acquire_debug_draw_interface()
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._device = self._cfg["sim_device"]
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

        self._gamma = self._task_cfg["env"]["gamma"]
        self._max_episode_length = self._task_cfg["env"]["horizon"]

        self._randomize_robot_on_reset = self._task_cfg["env"]["randomize_robot_on_reset"]

        # Get dt for integrating velocity commands and checking limit violations
        self._dt = torch.tensor(
            self._sim_config.task_config["sim"]["dt"] * self._sim_config.task_config["env"]["controlFrequencyInv"],
            device=self._device)

        # Environment object settings: (reset() randomizes the environment)
        # self._obstacle_names = ["larger_x_y_mammut", "godishus", "godishus2", "godishus3"]  # ShapeNet policy in usd format
        # self._tabular_obstacle_mask = [True,
        #                                False,
        #                                False,
        #                                False]  # Mask to denote which objects are tabular (i.e. grasp objects can be placed on them)

        self._obstacle_names = ["larger_x_y_mammut"]  # ShapeNet policy in usd format
        self._tabular_obstacle_mask = [True]  # Mask to denote which objects are tabular (i.e. grasp objects can be placed on them)
        #
        # self._obstacle_names = ["larger_x_y_mammut", "godishus"]  # ShapeNet policy in usd format
        # self._tabular_obstacle_mask = [
        #     True, False]  # Mask to denote which objects are tabular (i.e. grasp objects can be placed on them)

        # self._obstacle_names = ["larger_x_y_mammut", "godishus", "godishus2"]  # ShapeNet policy in usd format
        # self._tabular_obstacle_mask = [
        #     True, False, False]  # Mask to denote which objects are tabular (i.e. grasp objects can be placed on them)

        # 定义的物体的名称
        self._grasp_obj_names = ["004_sugar_box", "008_pudding_box", "010_potted_meat_can",
                                 "061_foam_brick"]  # YCB policy in usd format

        # 全封闭
        # self._grasp_obj_names = ["obstacle1", "obstacle2", "obstacle3", "obstacle4", "004_sugar_box"]

        # 半封闭
        # self._grasp_obj_names = ["obstacle1", "obstacle2", "obstacle3", "004_sugar_box"]
        # 可用的障碍物数量
        self._num_obstacles = min(self._task_cfg["env"]["num_obstacles"], len(self._obstacle_names))
        self.obj_bboxes_indexes = self._task_cfg["env"]["obj_bboxes_indexes"]

        # 可供抓取的对象数量
        self._num_grasp_objs = min(self._task_cfg["env"]["num_grasp_objects"], len(self._grasp_obj_names))
        # 初始化了表示对象状态的张量，因为每个对象状态需要6个值来表示其位置和旋转：3个用于位置（x、y、z坐标），3个用于旋转（通常是roll、pitch、yaw）
        self._obj_states = torch.zeros((6 * (self._num_obstacles + self._num_grasp_objs - 1), self._num_envs),
                                       device=self._device)  # All grasp objs except the target object will be used in obj state (BBox)
        self._obstacles = []
        self._obstacles_dimensions = []
        self._grasp_objs = []
        self._grasp_objs_dimensions = []
        #  Contact sensor interface for collision detection:
        # 用于碰撞检测的sensor_interface
        self._contact_sensor_interface = _isaac_sensor.acquire_contact_sensor_interface()

        # Choose num_obs and num_actions based on task
        # 6D goal/target object grasp pose + 6D bbox for each obstacle in the room. All grasp objs except the target object will be used in obj state
        # (3 pos + 4 quat + 6*(n-1)= 7 + ) ，观测空间的维度为以及物体的状态
        # self._num_observations = 7 + len(self._obj_states)
        # Todo:修改图结构
        self._num_observations = 64
        self._move_group = self._task_cfg["env"]["move_group"]
        self._use_torso = self._task_cfg["env"]["use_torso"]
        # Position control. Actions are base SE2 pose (3) and discrete arm activation (2)
        self._num_actions = self._task_cfg["env"]["continous_actions"] + self._task_cfg["env"]["discrete_actions"]
        # env specific limits
        self._world_xy_radius = self._task_cfg["env"]["world_xy_radius"]
        self._action_xy_radius = self._task_cfg["env"]["action_xy_radius"]
        self._action_ang_lim = self._task_cfg["env"]["action_ang_lim"]
        # self.max_arm_vel = torch.tensor(self._task_cfg["env"]["max_rot_vel"], device=self._device)
        # self.max_rot_vel = torch.tensor(self._task_cfg["env"]["max_rot_vel"], device=self._device)
        # self.max_base_xy_vel = torch.tensor(self._task_cfg["env"]["max_base_xy_vel"], device=self._device)

        # End-effector reaching settings
        self._goal_pos_threshold = self._task_cfg["env"]["goal_pos_thresh"]
        self._goal_ang_threshold = self._task_cfg["env"]["goal_ang_thresh"]
        # For now, setting dummy goal:
        self._goals = torch.hstack((torch.tensor([[0.8, 0.0, 0.4 + 0.15]]),
                                    euler_angles_to_quats(torch.tensor([[0.19635, 1.375, 0.19635]]),
                                                          device=self._device)))[0].repeat(self.num_envs, 1)
        self._goal_tf = torch.zeros((4, 4), device=self._device)
        self._goal_tf[:3, :3] = torch.tensor(Rotation.from_quat(np.array(
            [self._goals[0, 3 + 1], self._goals[0, 3 + 2], self._goals[0, 3 + 3], self._goals[0, 3]])).as_matrix(),
                                             dtype=float, device=self._device)  # Quaternion in scalar last format!!!
        self._goal_tf[:, -1] = torch.tensor([self._goals[0, 0], self._goals[0, 1], self._goals[0, 2], 1.0],
                                            device=self._device)  # x,y,z,1
        self._curr_goal_tf = self._goal_tf.clone()
        self._goals_xy_dist = torch.linalg.norm(self._goals[:, 0:2], dim=1)  # distance from origin

        # Reward settings，奖励设置
        self._reward_success = self._task_cfg["env"]["reward_success"]
        self._reward_dist_weight = self._task_cfg["env"]["reward_dist_weight"]
        self._reward_noIK = self._task_cfg["env"]["reward_noIK"]
        # self._reward_timeout = self._task_cfg["env"]["reward_timeout"]
        self._reward_collision = self._task_cfg["env"]["reward_collision"]
        self._penalty_slack = self._task_cfg["env"]["penalty_slack"]
        self.using_function = self._task_cfg["env"]["using_function"]

        self._collided = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
        self._ik_fails = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
        self._is_success = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)

        # IK solver，逆运动学求解器
        self._ik_solver = PinFetchIKSolver(move_group=self._move_group, include_torso=self._use_torso,
                                           include_base=False, max_rot_vel=100.0)  # No max rot vel

        # Handler for Fetch
        self.fetch_handler = FetchWBHandler(move_group=self._move_group, use_torso=self._use_torso,
                                            sim_config=self._sim_config, num_envs=self._num_envs, device=self._device)


        self.path = []

        RLTask.__init__(self, name, env)

        # 用于调试打印备份信息
        self.console_logger = ConsoleLogger(log_name='')
        """这是一段调试代码"""
        self.env = env

    # 配置场景，在这一部分中将场景设置得更加复杂
    def set_up_scene(self, scene) -> None:
        import omni
        self.fetch_handler.get_robot()
        # Spawn obstacles (from ShapeNet usd policy):
        for i in range(self._num_obstacles):
            obst = scene_utils.spawn_obstacle(name=self._obstacle_names[i],
                                              prim_path=self.fetch_handler.default_zero_env_path, device=self._device)
            self._obstacles.append(obst)  # Add to list of obstacles (Geometry Prims)
            # 在这一部分给obst添加了碰撞传感器所以可以检测到碰撞
            # Optional: Add contact sensors for collision detection. Covers whole body by default
            omni.kit.commands.execute("IsaacSensorCreateContactSensor", path="/Contact_Sensor",
                                      sensor_period=float(self._sim_config.task_config["sim"]["dt"]),
                                      parent=obst.prim_path)
        # Spawn grasp objs (from YCB usd policy):
        for i in range(self._num_grasp_objs):
            grasp_obj = scene_utils.spawn_grasp_object(name=self._grasp_obj_names[i],
                                                       prim_path=self.fetch_handler.default_zero_env_path,
                                                       device=self._device)
            self._grasp_objs.append(grasp_obj)  # Add to list of grasp objects (Rigid Prims)
            # 在这一部分给_grasp_objs添加了碰撞传感器所以可以检测到碰撞
            # Optional: Add contact sensors for collision detection. Covers whole body by default
            omni.kit.commands.execute("IsaacSensorCreateContactSensor", path="/Contact_Sensor",
                                      sensor_period=float(self._sim_config.task_config["sim"]["dt"]),
                                      parent=grasp_obj.prim_path)
        # Goal visualizer
        goal_viz = VisualCone(prim_path=self.fetch_handler.default_zero_env_path + "/goal",
                              radius=0.05, height=0.05, color=np.array([1.0, 0.0, 0.0]))
        super().set_up_scene(scene)
        self._robots = self.fetch_handler.create_articulation_view()
        scene.add(self._robots)
        self._goal_vizs = GeometryPrimView(prim_paths_expr="/World/envs/.*/goal", name="goal_viz")
        scene.add(self._goal_vizs)
        # Enable object axis-aligned bounding box computations
        # 允许计算bbox
        scene.enable_bounding_boxes_computations()
        # Add spawned objects to scene registry and store their bounding boxes:
        for obst in self._obstacles:
            scene.add(obst)
            self._obstacles_dimensions.append(
                scene.compute_object_AABB(obst.name))  # Axis aligned bounding box used as dimensions
        for grasp_obj in self._grasp_objs:
            scene.add(grasp_obj)
            self._grasp_objs_dimensions.append(
                scene.compute_object_AABB(grasp_obj.name))  # Axis aligned bounding box used as dimensions

        print("self._obstacles_dimensions=%s, obstacles_name=%s"%(self._obstacles_dimensions, self._obstacle_names[0]))
        # Optional viewport for rendering in a separate viewer
        from omni.isaac.synthetic_utils import SyntheticDataHelper
        self.viewport_window = omni.kit.viewport_legacy.get_default_viewport_window()
        self.sd_helper = SyntheticDataHelper()
        self.sd_helper.initialize(sensor_names=["rgb"], viewport=self.viewport_window)

    # 当仿真世界被重置时执行的操作
    def post_reset(self):
        # reset that takes place when the isaac world is reset (typically happens only once)
        self.fetch_handler.post_reset()

    def limit_angle(self, angle):
        # 将角度限制在 -π 到 π 之间
        while angle < -math.pi:
            angle += 2 * math.pi
        while angle > math.pi:
            angle -= 2 * math.pi
        return angle

    def refresh(self):
        for _ in range(3):
            self.env._world.step(render=True)
            self.env.rate.sleep()

    # 获取观测数据，包括当前目标的位置和姿态以及目标物体的bboxes
    def get_observations(self, curr_goal_tf=None, curr_bboxes_flattened=None, flag_save=True):
        if flag_save:
            # Handle any pending resets
            reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            if len(reset_env_ids) > 0:
                self.reset_idx(reset_env_ids)
            curr_goal_tf = self._curr_goal_tf
            # 获取待抓取物品的bbox的oriented bounding boxes of objects
            curr_bboxes_flattened = self._curr_obj_bboxes

        # curr_bboxes_flattened = curr_bboxes_flattened.flatten().unsqueeze(dim=0)
        # Fill observation buffer
        # Goal: 3D pos + rot_quaternion (3+4=7)

        # 获取目标物体位置
        curr_goal_pos = curr_goal_tf[0:3, 3].unsqueeze(dim=0)

        # 按理来说这个角度应该也是准确的，TODO：4.27调试通过
        curr_goal_quat = torch.tensor(Rotation.from_matrix(curr_goal_tf[:3, :3]).as_quat()[[3, 0, 1, 2]],
                                          dtype=torch.float, device=self._device).unsqueeze(dim=0)

        # JL:修改标准化四元数
        # curr_goal_quat /= torch.norm(curr_goal_quat)
        # if curr_goal_quat[0, 0] < 0:
        #     curr_goal_quat *= -1

        # state = torch.hstack((curr_goal_pos, curr_goal_quat, curr_bboxes_flattened))

        """测试绘制桌子等障碍物的位置"""
        """
        测试结果索引0，1，2为抓取物品附近的障碍物
        3为摆放的桌子
        4为障碍物的桌子
        """
        # def connect_points2(obj_bbox):
        #     # 这么计算是不正确的，因为告诉了min_xy_vertex和max_xy_vertex其实只能确定矩形的两个点
        #     # 不能直接通过组合得到另外的两个点，相当于一条连线确定了但矩形仍然是可以任意旋转的
        #     obj_bbox = obj_bbox.numpy()
        #     min_xy_vertex, max_xy_vertex = obj_bbox[:2], obj_bbox[2:4]
        #     points = np.ones([2, 3])
        #     points[:, 2] = obj_bbox[-2]
        #     points[0, 0:2] = np.array([min_xy_vertex[0], min_xy_vertex[1]])
        #     points[1, 0:2] = np.array([max_xy_vertex[0], max_xy_vertex[1]])
        #     return points
        # #
        # self.draw.clear_points()
        # # len_objects = len(curr_bboxes_flattened)
        # i = 3
        # points = connect_points2(self._obj_bboxes[i])
        # self.draw.draw_points(points, [(0, 0, 1, 1)] * 2, [8] * 2)
        """测试绘制桌子等障碍物的位置"""

        """在这里测试图神经网络部分代码"""
        # obj_bboxes_indexes = [[0, 1, 2], [3], [4]]
        # obj_bboxes_indexes = [[0, 1, 2], [3], [4]]

        # 当前的self._obj_bboxes应该不会出现问题
        state_graph = SceneGraph(curr_goal_pos, curr_goal_quat,
                                 curr_bboxes_flattened, self.obj_bboxes_indexes).graph


        # g = SceneGraph(curr_goal_pos, curr_goal_quat,
        #                curr_bboxes_flattened, self.obj_bboxes_indexes)
        # state_graph = g.graph
        #
        # g.print_graph_info(state_graph)

        # hstack 并转换为 NumPy 数组
        curr_goal_pos_quat = torch.hstack((curr_goal_pos, curr_goal_quat)).numpy()

        # 组合数据
        arr = [curr_goal_pos_quat[0], state_graph]

        # 创建包含不同类型对象的NumPy数组
        combined_arr = np.array([np.hstack(arr)], dtype=object)
        """在这里测试图神经网络部分代码"""

        if flag_save:
            self.obs_buf = combined_arr

        # TODO: Scale or normalize robot observations as per env
        return combined_arr


    """JL修改：用于在仿真环境中显示可达性，这里的动作一定要和生成的部分对应"""
    def show_reachability(self, q_values, size=1, resolution_dis=0.1, resolution_ang=1):
        reshaped_q_values = q_values.reshape((-1, 18))
        # 沿着第二个轴计算平均值
        average_values = np.mean(reshaped_q_values, axis=1)

        _len = len(average_values)

        # 将q值转为颜色
        min_val = np.min(average_values)
        max_val = np.max(average_values)
        normalized_q = (average_values - min_val) / (max_val - min_val)

        """这是一段调试q值大小的代码"""
        # 靠近的过程中正的值也大负的值也大
        avg_val = np.average(average_values)
        print("第%d轮，q值中：min_val=%s, max_val=%s, avg_val=%s"%(self.progress_buf, min_val, max_val, avg_val))
        """这是一段调试q值大小的代码"""

        # r = np.minimum(1.0, 2.0 - 2.0 * normalized_q)
        # g = np.minimum(1.0, 2.0 * normalized_q)
        # b = np.zeros_like(q_values)
        # alpha = np.ones_like(q_values)
        # colors = np.column_stack((r, g, b, alpha))

        dark_blue = (0, 1/255, 249/255, 1)
        green = (2/255, 247/255, 2/255, 1)
        yellow = (248/255, 250/255, 1/255, 1)
        cyan = (0, 251/255, 247/255, 1)
        ori = (247/255, 204/255, 239/255, 1)
        # red1 = (255 / 255, 140 / 255, 0, 1)  # 橙色
        # red2 = (242 / 255, 90 / 255, 96 / 255, 1)
        # red3 = (255 / 255, 99/255, 71/255, 1)  # 番茄
        # red4 = (255 / 255, 0, 0, 1)  # 赤红
        # red5 = (178 / 255, 34 / 255, 34 / 255, 1)  # 耐火砖

        # red = (242 / 255, 90 / 255, 96 / 255, 1)
        red = (247 / 255, 0, 0, 1)
        colors = np.zeros([_len, 4])
        for i in range(_len):
            if self.point_inside_rectangle(self.table_corners, self.round_loc[i, :2]):
                colors[i, :] = ori
            elif normalized_q[i] > 0.85:
                colors[i, :] = dark_blue
            elif normalized_q[i] > 0.8:
                colors[i, :] = cyan
            elif normalized_q[i] > 0.7:
                colors[i, :] = green
            elif normalized_q[i] > 0.6:
                colors[i, :] = yellow
            # elif normalized_q[i] > 0.5:
            #     colors[i, :] = red1
            # elif normalized_q[i] > 0.4:
            #     colors[i, :] = red2
            # elif normalized_q[i] > 0.3:
            #     colors[i, :] = red3
            # elif normalized_q[i] > 0.2:
            #     colors[i, :] = red4
            else:
                colors[i, :] = red

        # 采用画点的方式，仅仅画点效果也挺好的
        self.draw.clear_points()
        self.draw.draw_points(self.round_loc, colors, [10]*_len)

        # 采用画线的方式
        # self.draw.clear_lines()
        # # self.round_loc2 = np.roll(self.round_loc, -1, axis=0)
        # # # self.console_logger.info("self.round_loc=%s \n self.round_loc2=%s"%(self.round_loc, self.round_loc2))
        # # self.draw.draw_lines(self.round_loc, self.round_loc2, colors, [size]*_len)
        #
        # # 画线包括两部分：从圆心向外画线，在Jupyter文件中从测试过索引值的正确性
        # start = 0
        # line_len = int((1.3 - 0.3) / resolution_dis)
        # # self.console_logger.info("line_len=%d"%(line_len))
        # for phi_cnt in range(0, 360, resolution_ang):
        #     # self.console_logger.info("起始点坐标为(%d, %d)"%(start, start+line_len))
        #     cur_line = self.round_loc[start:start+line_len-1, :]
        #     cur_color = colors[start:start+line_len-1, :]
        #     next_line = self.round_loc[start+1:start+line_len, :]
        #     self.draw.draw_lines(cur_line, next_line, cur_color, [size] * (line_len - 1))
        #     start += line_len
        #
        # # 一圈一圈画线如同涟漪一般
        # start = 0
        # round_len = int((360 - 0) / resolution_ang)
        # for r in np.arange(0.3, 1.3, resolution_dis):
        #     cur_line = self.round_loc[start:-line_len:line_len, :]
        #     cur_color = colors[start:-line_len:line_len, :]
        #     next_line = self.round_loc[start+line_len::line_len, :]
        #     self.draw.draw_lines(cur_line, next_line, cur_color, [size] * (round_len - 1))
        #     start += 1

        """用于调试桌子部分的坐标是否正确"""
        # self.draw.clear_points()
        # self.draw.draw_points(self.table_corners, [(0, 1, 0, 1)] * 4, [10] * 4)

        # self.console_logger.info("self.round_loc=%s" % (self.round_loc))
        """这是一段调试代码"""
        # cnt = 0
        # loc = []
        # for r in np.arange(0.3, 0.8, 0.05):
        #     for phi_cnt in range(0, 720, 1):
        #         # phi = phi_cnt * math.pi / 180
        #         phi = math.radians(phi_cnt)
        #         x = r * math.cos(phi)
        #         y = r * math.sin(phi)
        #
        #         loc.append((x, y, 0))
        #
        #         cnt += 1
        # self.draw.draw_points(loc, [(0, 1, 0, 1)] * cnt, [7] * cnt)

        # 画线会比画点效果好一点
        # cnt = 0
        # point_list_1 = []
        # point_list_2 = []
        # for r in np.arange(0.3, 0.8, 0.05):
        #     for phi_cnt in range(0, 360, 1):
        #         phi = phi_cnt * math.pi / 180
        #         x = r * math.cos(phi)
        #         y = r * math.sin(phi)
        #
        #         if phi_cnt == 0:
        #             point_list_1.append((x, y, 0))
        #         elif phi_cnt == 359:
        #             point_list_2.append((x, y, 0))
        #         else:
        #             point_list_2.append((x, y, 0))
        #             point_list_1.append((x, y, 0))
        #
        #         cnt += 1
        #     self.draw.draw_lines(point_list_1, point_list_2, [(1, 0, 0, 1)] * 359, [3] * 359)
        #     print("len(point_list_1)=%d" % (len(point_list_1)))
        #     point_list_1 = []
        #     point_list_2 = []
        """这是一段调试代码"""

    """JL修改：用于显示桌子周围的可达性"""
    def show_surrounding_reachablity(self, q_values, size=20):
        # 首先进行归一化
        reshaped_q_values = q_values.reshape((-1, 18))
        # 沿着第二个轴计算平均值
        average_values = np.mean(reshaped_q_values, axis=1)

        _len = len(average_values)

        # 将q值转为颜色
        min_val = np.min(average_values)
        max_val = np.max(average_values)
        normalized_q = (average_values - min_val) / (max_val - min_val)

        """这是一段调试q值大小的代码"""
        # 靠近的过程中正的值也大负的值也大
        avg_val = np.average(average_values)
        print("第%d轮，q值中：min_val=%s, max_val=%s, avg_val=%s" % (self.progress_buf, min_val, max_val, avg_val))
        """这是一段调试q值大小的代码"""

        # 设置颜色
        dark_blue = (0, 1 / 255, 249 / 255, 1)
        green = (2 / 255, 247 / 255, 2 / 255, 1)
        yellow = (248 / 255, 250 / 255, 1 / 255, 1)
        cyan = (0, 251 / 255, 247 / 255, 1)
        red = (247 / 255, 0, 0, 1)
        colors = np.zeros([_len, 4])

        for i in range(_len):
            if normalized_q[i] > 0.8:
                colors[i, :] = dark_blue
            elif normalized_q[i] > 0.7:
                colors[i, :] = cyan
            elif normalized_q[i] > 0.5:
                colors[i, :] = green
            elif normalized_q[i] > 0.3:
                colors[i, :] = yellow
            else:
                colors[i, :] = red

        self.draw.clear_points()
        self.draw.draw_points(self.surrounding_area, colors, [size]*_len)


        global_goal_pos = [self._goals[0, :3].numpy()]
        # self.console_logger.info("global_goal_pos=%s"%(global_goal_pos))
        self.draw.draw_points(global_goal_pos, [(0, 1, 0, 1)], [30])

        self.draw.draw_points(self.object_positions, [(0, 0, 1, 1)]*4, [30]*4)



    """JL修改：用于计算桌子的四角坐标"""
    def compute_table_corners(self, center, dimensions, yaw):
        # print("center=%s, dimensions=%s, yaw=%s" % (center, dimensions, yaw))
        center[2] = 0

        # 解包长宽高
        length, width, height = dimensions

        # 计算桌子顶点的相对位置
        corners_relative = np.array([
            [-length / 2, -width / 2, 0],
            [length / 2, -width / 2, 0],
            [length / 2, width / 2, 0],
            [-length / 2, width / 2, 0]
        ])

        # 构建旋转矩阵
        rotation_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        # 应用旋转矩阵
        rotated_corners = np.dot(corners_relative, rotation_matrix.T)

        # 平移到桌子中心
        corners = rotated_corners + np.array(center)
        # self.console_logger.info("corners=%s, type(corners)=%s"%(corners, type(corners)))
        return corners

    def point_inside_rectangle(self, corners, location):
        # 长方形的最小 x 坐标和最大 x 坐标
        min_x = np.min(corners[:, 0])
        max_x = np.max(corners[:, 0])

        # 长方形的最小 y 坐标和最大 y 坐标
        min_y = np.min(corners[:, 1])
        max_y = np.max(corners[:, 1])

        # self.console_logger.info("location=%s"%(location))
        a, b = location[0], location[1]

        # 检查点 (a, b) 是否在长方形内部
        if a >= min_x and a <= max_x and b >= min_y and b <= max_y:
            return True
        else:
            return False

    # 获取渲染图像
    def get_render(self):
        # Get ground truth viewport rgb image
        gt = self.sd_helper.get_groundtruth(
            ["rgb"], self.viewport_window, verify_sensor_init=False, wait_for_sensor_data=0
        )
        # 获取当前时间戳
        timestamp = int(time.time())
        # 生成文件名，包含当前时间戳
        image = Image.fromarray(np.array(gt["rgb"][:, :, :3]))

        image.save("./reachability_paths/output_image_%d.png"%(timestamp))
        return np.array(gt["rgb"][:, :, :3])

    """JL修改：将坐标转换到机器人相对坐标系下"""
    def transform_to_relative_coordinate(self, robot_x, robot_y, robot_rad, object_x, object_y):
        # 计算相对位置
        relative_x = object_x - robot_x
        relative_y = object_y - robot_y

        # 将相对位置旋转，使得机器人的朝向与 x 轴重合
        rotated_x = relative_x * math.cos(robot_rad) + relative_y * math.sin(robot_rad)
        rotated_y = -relative_x * math.sin(robot_rad) + relative_y * math.cos(robot_rad)

        return rotated_x, rotated_y

    """JL修改：将位移转换到机器人局部坐标系下"""
    def transform_to_robot_coordinate(self, robot_x, robot_y, robot_rad, x_scaled, y_scaled, theta_scaled):
        new_x = x_scaled * math.cos(robot_rad) - y_scaled * math.sin(robot_rad) + robot_x
        new_y = x_scaled * math.sin(robot_rad) + y_scaled * math.cos(robot_rad) + robot_y
        return new_x, new_y

    def connect_points(self, min_xy_vertex, max_xy_vertex, flag=True, height=1):
        # 先转numpy
        min_xy_vertex = min_xy_vertex.numpy()
        max_xy_vertex = max_xy_vertex.numpy()

        points = np.ones([2, 3])
        points[:, 2] = height
        points[0, 0:2] = np.array([min_xy_vertex[0], min_xy_vertex[1]])
        points[1, 0:2] = np.array([max_xy_vertex[0], max_xy_vertex[1]])

        if flag:
            points = np.hstack((points, np.ones((points.shape[0], 1))))
            # 获取机器人的基准坐标系
            base_joint_pos = self.fetch_handler.get_robot_obs()[:, :3]  # First three are always base positions

            # 构建基座的变换矩阵base_tf。根据基座的旋转角度（base_joint_pos[0,2]），设置旋转矩阵的旋转部分，并将基座的位置设置为变换矩阵的最后一列
            base_tf = torch.zeros((4, 4), device=self._device)
            base_tf[:2, :2] = torch.tensor([[torch.cos(base_joint_pos[0, 2]), -torch.sin(base_joint_pos[0, 2])],
                                            [torch.sin(base_joint_pos[0, 2]),
                                             torch.cos(base_joint_pos[0, 2])]])  # rotation about z axis
            base_tf[2, 2] = 1.0  # No rotation here
            base_tf[:, -1] = torch.tensor([base_joint_pos[0, 0], base_joint_pos[0, 1], 0.0, 1.0])
            base_tf = base_tf.numpy()

            # 将点的坐标从机器人坐标系转换到全局坐标系
            points = np.dot(base_tf, points.T).T[:, :3]

        return points

    """这个函数是下面三个函数都要用到的基础函数"""
    def get_normal_state(self, base_tf, theta):
        total = self._num_obstacles + self._num_grasp_objs - 1
        min_xy_vertex = torch.zeros([total, 4], dtype=torch.float32)
        max_xy_vertex = torch.zeros([total, 4], dtype=torch.float32)

        # 这里的六个维度分别对应于x方向最小值，y方向最小值，x方向最大值和y方向最大值，到地面的距离，以及object_yaw物品的朝向
        for obj_num in range(total):
            # 获取当前对象轴对齐边界框的最小的 XY 顶点坐标
            min_xy_vertex[obj_num] = torch.hstack(
                (self._obj_bboxes[obj_num, 0:2], torch.tensor([0.0, 1.0], device=self._device))).T
            # 获取当前对象轴对齐边界框的最大的 XY 顶点坐标
            max_xy_vertex[obj_num] = torch.hstack(
                (self._obj_bboxes[obj_num, 2:4], torch.tensor([0.0, 1.0], device=self._device))).T

        # Transform goal to robot frame
        # 将目标转换到机器人坐标系中。首先计算基座到世界坐标系的逆变换矩阵，然后将目标的变换矩阵与逆变换矩阵相乘，得到目标在机器人坐标系中的新位置
        # self._goal_tf其实与self._goal有关，已更新
        inv_base_tf = torch.linalg.inv(base_tf)
        # 这里inv_base_tf中对应的sin和cos的值应该是相同的，因为sin和cos的周期是2*pi不会改变
        curr_goal_tf = torch.matmul(inv_base_tf, self._goal_tf)

        # 每次都需要更新数据，这个视角
        # self._obj_bboxes中记录了所有的除了goal之外物体的bbox
        curr_obj_bboxes = self._obj_bboxes.clone()

        """注意：这部分在调试完成后一定要删除"""
        # self.draw.clear_points()

        # Transform all other object-oriented bounding boxes to robot frame
        # 将所有其他对象的轴对齐边界框转换到机器人的参考框架
        """对于每个函数而言都需要加入curr_obj_bboxes的观测，但是其实这个不涉及到要抓取物体，这个只涉及到其他物体"""
        # 循环遍历所有对象，包括障碍物和可供抓取的对象，但排除了目标对象本身
        for obj_num in range(total):
            # 通过矩阵乘法将最小顶点坐标转换到机器人参考框架中，并更新为新的最小顶点坐标
            new_min_xy_vertex = torch.matmul(inv_base_tf, min_xy_vertex[obj_num])[0:2].T.squeeze()
            # 通过矩阵乘法将最大顶点坐标转换到机器人参考框架中，并更新为新的最大顶点坐标
            new_max_xy_vertex = torch.matmul(inv_base_tf, max_xy_vertex[obj_num])[0:2].T.squeeze()
            # 将新的最小和最大顶点坐标组合成一个新的轴对齐边界框
            curr_obj_bboxes[obj_num, 0:4] = torch.hstack((new_min_xy_vertex, new_max_xy_vertex))
            # 将对象的角度调整为机器人当前的角度
            # 这个角度的计算要么就是当前坐标系下直接减去在世界坐标系下的角度curr_obj_bboxes = self._obj_bboxes.clone()，-expected_theta
            # 要么由记录的上一次角度curr_obj_bboxes = self._curr_obj_bboxes.clone()，-theta_scaled
            curr_obj_bboxes[obj_num, 5] = self.limit_angle(curr_obj_bboxes[obj_num, 5] - theta)  # new theta

            """这是一段检查以机器人为局部坐标系结果是否正确的代码：绘制转换后的点"""
            # points = self.connect_points(self._obj_bboxes[obj_num, 0:2], self._obj_bboxes[obj_num, 2:4], False, self._obj_bboxes[obj_num, -2])
            # points = self.connect_points(new_min_xy_vertex, new_max_xy_vertex, height=self._obj_bboxes[obj_num, -2])
            # self.draw.draw_points(points, [(0, 0, 1, 1)] * 4, [8] * 4)
            """绘制转换后的点"""

        return curr_goal_tf, curr_obj_bboxes

    """JL修改：将更新观测以及动作变为一个函数"""
    def update_observation(self, actions):
        # print("当前第%d步的动作为%s"%(self.progress_buf, actions))
        # Scale actions and convert polar co-ordinates r-phi to x-y
        # NOTE: actions shape has to match the move_group selected
        # 缩放动作，将极坐标（r-phi）转换为笛卡尔坐标（x-y）。根据动作中的极坐标参数（actions[:,0]和actions[:,1]），分别乘以动作的xy半径和角度限制来获得x和y方向的缩放值，然后使用三角函数将极坐标转换为笛卡尔坐标
        """测试一下机器人原地转圈的情况下，可达性结果是否正确"""
        r_scaled = actions[:, 0] * self._action_xy_radius
        phi_scaled = actions[:, 1] * self._action_ang_lim
        x_scaled = r_scaled * torch.cos(phi_scaled)
        y_scaled = r_scaled * torch.sin(phi_scaled)
        theta_scaled = actions[:, 2] * self._action_ang_lim

        # NOTE: Actions are in robot frame but the handler is in world frame!
        # Get current base positions
        # 动作是在机器人坐标系中定义的，但处理程序是在世界坐标系中操作的！获取当前base的位置，这些位置是机器人的关节位置中的前三个
        base_joint_pos = self.fetch_handler.get_robot_obs()[:, :3]  # First three are always base positions

        # 构建基座的变换矩阵base_tf。根据基座的旋转角度（base_joint_pos[0,2]），设置旋转矩阵的旋转部分，并将基座的位置设置为变换矩阵的最后一列
        base_tf = torch.zeros((4, 4), device=self._device)
        base_tf[:2, :2] = torch.tensor([[torch.cos(base_joint_pos[0, 2]), -torch.sin(base_joint_pos[0, 2])],
                                        [torch.sin(base_joint_pos[0, 2]),
                                         torch.cos(base_joint_pos[0, 2])]])  # rotation about z axis
        base_tf[2, 2] = 1.0  # No rotation here
        base_tf[:, -1] = torch.tensor([base_joint_pos[0, 0], base_joint_pos[0, 1], 0.0, 1.0])  # x,y,z,1

        # Transform actions to world frame and apply to base
        # 将动作转换为世界坐标系并应用到基座上。根据动作的旋转角度（theta_scaled[0]）设置旋转矩阵的旋转部分，并将动作的位移设置为变换矩阵的最后一列
        action_tf = torch.zeros((4, 4), device=self._device)
        action_tf[:2, :2] = torch.tensor([[torch.cos(theta_scaled[0]), -torch.sin(theta_scaled[0])],
                                          [torch.sin(theta_scaled[0]), torch.cos(theta_scaled[0])]])
        action_tf[2, 2] = 1.0  # No rotation here
        action_tf[:, -1] = torch.tensor([x_scaled[0], y_scaled[0], 0.0, 1.0])  # x,y,z,1

        # 计算应用动作后的新基座的变换矩阵。将基座的位移提取出来作为新的基座XY坐标，并计算新的基座角度
        new_base_tf = torch.matmul(base_tf, action_tf)
        new_base_xy = new_base_tf[0:2, 3].unsqueeze(dim=0)
        new_base_theta = torch.arctan2(new_base_tf[1, 0], new_base_tf[0, 0]).unsqueeze(dim=0).unsqueeze(dim=0)

        # Move base
        # 移动基座。将新的基座位置（XY坐标和角度）设置到机器人处理程序中，用于更新机器人的位置
        self.fetch_handler.set_base_positions(torch.hstack((new_base_xy, new_base_theta)))
        # print("当前机器人动作的new_base_xy = %s, new_base_theta=%s, x_scaled=%s, y_scaled=%s, theta_scaled=%s"
        #       %(new_base_xy, new_base_theta, x_scaled, y_scaled, theta_scaled))

        self._curr_goal_tf, self._curr_obj_bboxes = self.get_normal_state(new_base_tf, new_base_theta[0][0])
        state = self.get_observations(curr_goal_tf=self._curr_goal_tf, curr_bboxes_flattened=self._curr_obj_bboxes,
                                               flag_save=False)

        """绘制机器人路径"""
        # print("base_joint_pos[0, :]=")
        # print(base_joint_pos[0, :])
        # input("测试输入")
        self.path.append(new_base_tf[0:3, 3].numpy())

        # input("测试输入")
        """绘制机器人路径"""

        self.robot_state = np.tile(state, (3240, 1))


    """JL修改：角度采用朝向目标物体的角度，用于获取以机器人为中心的状态，局部坐标点的位置以及对应的动作"""
    def get_round_states(self, resolution_dis=0.2, resolution_ang=10, point_loop=18):
        """数据利用部分"""
        # 1.在真正计算前的函数
        base_joint_pos = self.fetch_handler.get_robot_obs()[:, :3]  # First three are always base positions

        base_tf = torch.zeros((4, 4), device=self._device)
        base_tf[:2, :2] = torch.tensor([[torch.cos(base_joint_pos[0, 2]), -torch.sin(base_joint_pos[0, 2])],
                                        [torch.sin(base_joint_pos[0, 2]),
                                         torch.cos(base_joint_pos[0, 2])]])  # rotation about z axis
        base_tf[2, 2] = 1.0  # No rotation here
        base_tf[:, -1] = torch.tensor([base_joint_pos[0, 0], base_joint_pos[0, 1], 0.0, 1.0])  # x,y,z,1

        global_goal_pos = self._goals[0, :3]

        """开始循环"""
        total_len = int(360 / resolution_ang * (1.3 - 0.3) / resolution_dis)
        # total_len = 3600 * 18

        # 定义数组形状
        shape = (total_len * point_loop, 8)

        # 创建一个空数组，数据类型为对象类型
        states = np.empty(shape, dtype=object)

        # states = np.zeros([total_len * point_loop, 31])  # 当前对应的观测状态
        round_actions = np.zeros([total_len * point_loop, 5])  # 当前对应的动作
        round_actions[:, 3] = 1
        curr_round_loc = np.zeros([total_len, 3])  # 当前实际的位姿
        curr_round_loc[:, 2] = 0.1
        angle_gap = math.pi / 180  # 角度转弧度
        cnt = 0
        cnt_single = 0

        for phi_cnt in range(0, 360, resolution_ang):
            for r in np.arange(0.3, 1.3, resolution_dis):
                phi = phi_cnt * angle_gap

                round_actions[cnt:cnt+point_loop, 0] = r / self._action_xy_radius
                round_actions[cnt:cnt+point_loop, 1] = phi / self._action_ang_lim

                x_scaled = torch.tensor([r * np.cos(phi)])
                y_scaled = torch.tensor([r * np.sin(phi)])

                # Transform actions to world frame and apply to base
                # 将动作转换为世界坐标系并应用到基座上。根据动作的旋转角度（theta_scaled[0]）设置旋转矩阵的旋转部分，并将动作的位移设置为变换矩阵的最后一列
                action_tf = torch.zeros((4, 4), device=self._device)

                action_tf[2, 2] = 1.0  # No rotation here
                action_tf[:, -1] = torch.tensor([x_scaled[0], y_scaled[0], 0.0, 1.0])  # x,y,z,1
                # 计算应用动作后的新基座的变换矩阵。将基座的位移提取出来作为新的基座XY坐标，并计算新的基座角度
                new_base_tf = torch.matmul(base_tf, action_tf)
                new_base_xy = new_base_tf[0:2, 3].unsqueeze(dim=0)

                curr_round_loc[cnt_single, :2] = new_base_xy[0, :].numpy()

                # 在世界坐标系下目前想要的角度为
                expected_theta = math.atan2(global_goal_pos[1] - new_base_xy[0, 1], global_goal_pos[0] - new_base_xy[0, 0])

                curr_round_loc[cnt_single, :2] = new_base_xy[0, :].numpy()

                for ang in range(-27, 27, 3):
                    cur_theta = expected_theta + ang * angle_gap

                    theta_scaled = torch.tensor([cur_theta - base_joint_pos[0, 2]])
                    round_actions[cnt, 2] = theta_scaled / self._action_ang_lim

                    action_tf[:2, :2] = torch.tensor([[torch.cos(theta_scaled[0]), -torch.sin(theta_scaled[0])],
                                                      [torch.sin(theta_scaled[0]), torch.cos(theta_scaled[0])]])
                    new_base_tf = torch.matmul(base_tf, action_tf)

                    curr_goal_tf, curr_obj_bboxes = self.get_normal_state(new_base_tf, cur_theta)
                    states[cnt, :] = self.get_observations(curr_goal_tf=curr_goal_tf, curr_bboxes_flattened=curr_obj_bboxes,
                                                           flag_save=False)

                    """绘图检查机器人状态更新"""
                    # new_base_theta = torch.arctan2(new_base_tf[1, 0], new_base_tf[0, 0]).unsqueeze(dim=0).unsqueeze(dim=0)
                    # self.fetch_handler.set_base_positions(torch.hstack((new_base_xy, new_base_theta)))
                    # self.draw.draw_points([curr_round_loc[cnt, :]], [(1, 0, 0, 1)], [5])
                    # self.refresh()
                    """绘图检查机器人状态更新"""

                    cnt += 1

                cnt_single += 1

        return states, curr_round_loc, round_actions


    """JL修改：获得以物体为中心的可操作区域"""
    def get_object_states(self, center, dimensions, yaw, r_max=0.8, r_min=0.3, resolution=0.1, point_loop=18):
        center[2] = 0
        # 解包长宽高
        length, width, height = dimensions

        half_length = length / 2
        half_width = width / 2

        area_points = list()

        # 排除不需要的区域
        for i in np.arange(-half_width - r_max, half_width + r_max, resolution):
            for j in np.arange(-half_length - r_max, half_length + r_max, resolution):
                if -half_width - r_min < i < half_width + r_min and -half_length - r_min < j < half_length + r_min:
                    continue
                area_points.append([j, i, 0])

        # 构建旋转矩阵
        rotation_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        # 应用旋转矩阵
        rotated_corners = np.dot(area_points, rotation_matrix.T)

        # 平移到桌子中心
        surrounding_area = rotated_corners + np.array(center)

        # 获取当前的状态
        len_surrounding_area = len(surrounding_area)

        cnt = 0
        # states = np.zeros([len_surrounding_area*point_loop, 31])  # 当前对应的观测状态

        # 定义数组形状
        shape = (len_surrounding_area*point_loop, 8)

        # 创建一个空数组，数据类型为对象类型
        states = np.empty(shape, dtype=object)

        surrounding_actions = np.zeros([len_surrounding_area*point_loop, 5])

        surrounding_actions[:, 3] = 1

        angle_gap = math.pi / 180

        # 获取世界坐标系下的绝对值
        global_goal_pos = self._goals[0, :3]
        # base_joint_pos = self.fetch_handler.get_robot_obs()[:, :3]  # First three are always base positions

        for i in range(len_surrounding_area):
            # 获取当前物体周围的坐标
            cur_loc = surrounding_area[i, :]

            # 获取当前期望的角度
            expected_theta = torch.atan2(global_goal_pos[1] - cur_loc[1], global_goal_pos[0] - cur_loc[0])

            # self.console_logger.info("当前角度expected_theta=%s, theta_scaled=%s"%(expected_theta, theta_scaled))

            for ang in range(-27, 27, 3):
                cur_theta = expected_theta + ang * angle_gap

                # 获取当前的base_tf
                base_tf = torch.zeros((4, 4), device=self._device)
                base_tf[:2, :2] = torch.tensor([[torch.cos(cur_theta), -torch.sin(cur_theta)],
                                                [torch.sin(cur_theta),
                                                 torch.cos(cur_theta)]])  # rotation about z axis
                base_tf[2, 2] = 1.0  # No rotation here
                base_tf[:, -1] = torch.tensor([cur_loc[0], cur_loc[1], 0.0, 1.0])  # x,y,z,1

                curr_goal_tf, curr_obj_bboxes = self.get_normal_state(base_tf, cur_theta)
                states[cnt, :] = self.get_observations(curr_goal_tf=curr_goal_tf, curr_bboxes_flattened=curr_obj_bboxes, flag_save=False)

                """绘图检查机器人状态更新"""
                # new_base_xy = torch.unsqueeze(torch.tensor(cur_loc[:2]), dim=0)
                # new_base_theta = torch.arctan2(base_tf[1, 0], base_tf[0, 0]).unsqueeze(dim=0).unsqueeze(dim=0)
                # self.fetch_handler.set_base_positions(torch.hstack((new_base_xy, new_base_theta)))
                # self.draw.draw_points([cur_loc], [(1, 0, 0, 1)], [5])
                # self.refresh()
                """绘图检查机器人状态更新"""

                cnt += 1

        return surrounding_area, states, surrounding_actions


    # 在物理步骤之前执行的操作，主要是处理动作、重置以及坐标变换等
    def pre_physics_step(self, actions) -> None:
        # actions (num_envs, num_action)
        # Handle resets，对环境进行重置
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        """修改为一个函数"""
        self.update_observation(actions)

        """更新状态"""
        # self.states, self.round_loc = self.get_round_states1()
        if self.using_function == 2:
            self.states, self.round_loc, self.next_action = self.get_round_states()
        # self.next_action, self.round_loc = self.get_round_actions()

        # self.console_logger.info('Step1:第%d步，设置机器人基座当前位置new_base_xy=%s, new_base_theta=%s， %.4f' % (
        # self.progress_buf, str(new_base_xy), str(new_base_theta), math.degrees(new_base_theta)))
        """修改为一个函数"""

        # Discrete Arm action:
        # 处理离散臂动作
        self._ik_fails[0] = 0
        # 检查动作中的第四个和第五个动作参数，并比较它们的值。这个动作参数可以被认为是控制机器人臂运动的决策变量
        if actions[0, 3] > actions[0, 4]:  # This is the arm decision variable TODO: Parallelize
            # Compute IK to self._curr_goal_tf
            curr_goal_pos = self._curr_goal_tf[0:3, 3]
            curr_goal_quat = Rotation.from_matrix(self._curr_goal_tf[:3, :3]).as_quat()[[3, 0, 1, 2]]

            # self.console_logger.info('Step2:当前距离障碍物curr_goal_pos=%s, curr_goal_quat=%s' % (str(curr_goal_pos), str(curr_goal_quat)))

            # 求解逆运动学，判断是否能够成功到达
            success, ik_positions = self._ik_solver.solve_ik_pos_fetch(des_pos=curr_goal_pos.cpu().numpy(),
                                                                       des_quat=curr_goal_quat,
                                                                       pos_threshold=self._goal_pos_threshold,
                                                                       angle_threshold=self._goal_ang_threshold,
                                                                       verbose=False)
            if success:
                self._is_success[0] = 1  # Can be used for reward, termination
                # set upper body positions 将找到的关节位置应用于机器人的上半身
                self.fetch_handler.set_upper_body_positions(
                    jnt_positions=torch.tensor(np.array([ik_positions]), dtype=torch.float, device=self._device))
                # 打印成功率等信息
                # self.console_logger.info(
                #     'Step3:成功获得逆解success=%s, ik_positions=%s' % (str(success), str(ik_positions)))
            else:
                self._ik_fails[0] = 1  # Can be used for reward
                # self.console_logger.info('Debug4:未找到可行动作')

    def reset_idx(self, env_ids):
        # apply resets
        indices = env_ids.to(dtype=torch.int32)
        # reset dof values，这里重新配置机械臂的关节位置
        self.fetch_handler.reset(indices, randomize=self._randomize_robot_on_reset)

        # reset the scene objects (randomize), get target end-effector goal/grasp as well as oriented bounding boxes of all other objects
        # get target end-effector goal/grasp as well as oriented bounding boxes of all other objectsscene_utils.setup_tabular_scene(
        # self._obstacles障碍物的列表，包含了场景中的所有障碍物
        # self._tabular_obstacle_mask[0:self._num_obstacles]用于标识哪些障碍物是用于标签的
        # self._grasp_objs: 可供抓取的对象列表，包含了场景中的所有可供抓取的对象
        # self._obstacles_dimensions: 障碍物的轴对齐边界框的尺寸列表
        # self._grasp_objs_dimensions: 可供抓取对象的轴对齐边界框的尺寸列表
        # self._world_xy_radius: 世界的XY半径，用于限制对象的位置
        # self._device: 运行设备，指定在哪个设备上进行计算
        (self._curr_grasp_obj, self._goals[env_ids], self._obj_bboxes, tab_xyz_size, tab_position, tab_yaw, self.object_positions) = scene_utils.setup_tabular_scene(
            self, self._obstacles, self._tabular_obstacle_mask[0:self._num_obstacles], self._grasp_objs,
            self._obstacles_dimensions, self._grasp_objs_dimensions, self._world_xy_radius, self._device)

        self._curr_obj_bboxes = self._obj_bboxes.clone()

        # print("在配置场景时：self._curr_obj_bboxes=%s"%(self._curr_obj_bboxes))
        # self._goals[env_ids] = torch.hstack((goals_sample[:,:3],euler_angles_to_quats(goals_sample[:,3:6],device=self._device)))

        self._goal_tf = torch.zeros((4, 4), device=self._device)
        goal_rot = Rotation.from_quat(np.array([self._goals[0, 3 + 1], self._goals[0, 3 + 2], self._goals[0, 3 + 3],
                                                self._goals[0, 3]]))  # Quaternion in scalar last format!!!
        self._goal_tf[:3, :3] = torch.tensor(goal_rot.as_matrix(), dtype=float, device=self._device)
        self._goal_tf[:, -1] = torch.tensor([self._goals[0, 0], self._goals[0, 1], self._goals[0, 2], 1.0],
                                            device=self._device)  # x,y,z,1



        self._curr_goal_tf = self._goal_tf.clone()
        self._goals_xy_dist = torch.linalg.norm(self._goals[:, 0:2], dim=1)  # distance from origin

        # Pitch visualizer by 90 degrees for aesthetics
        goal_viz_rot = goal_rot * Rotation.from_euler("xyz", [0, np.pi / 2.0, 0])
        self._goal_vizs.set_world_poses(indices=indices, positions=self._goals[env_ids, :3],
                                        orientations=torch.tensor(goal_viz_rot.as_quat()[[3, 0, 1, 2]],
                                                                  device=self._device).unsqueeze(dim=0))

        # 获取robot当前状态
        curr_bboxes_flattened = self._curr_obj_bboxes.flatten().unsqueeze(dim=0)
        curr_goal_pos = self._curr_goal_tf[0:3, 3].unsqueeze(dim=0)
        curr_goal_quat = torch.tensor(Rotation.from_matrix(self._curr_goal_tf[:3, :3]).as_quat()[[3, 0, 1, 2]],
                                      dtype=torch.float, device=self._device).unsqueeze(dim=0)

        state = torch.hstack((curr_goal_pos, curr_goal_quat, curr_bboxes_flattened))
        self.robot_state = state.repeat(3240, 1)

        # 获取局部坐标系下当前状态
        self.round_loc = None
        if self.using_function == 2:
            self.states, self.round_loc, self.next_action = self.get_round_states()
        # self.next_action, self.round_loc = self.get_round_actions()

        # JL修改：用于计算桌子的四角
        if self.using_function == 2:
            self.table_corners = self.compute_table_corners(tab_position, tab_xyz_size, tab_yaw)

        # # JL修改：获取桌子周围的状态
        if self.using_function == 3:
            self.surrounding_area, self.surrounding_area_states, self.surrounding_actions = self.get_object_states(
                tab_position, tab_xyz_size, tab_yaw)

        """调试代码"""
        # self.console_logger.info("goal_viz_rot.as_quat()=%s"%(goal_viz_rot.as_quat()))
        # self.console_logger.info("indices=%s, self._goals[env_ids, :3]=%s, ori=%s"%(
        #     indices, self._goals[env_ids, :3], torch.tensor(goal_viz_rot.as_quat()[[3, 0, 1, 2]],
        #                                                           device=self._device).unsqueeze(dim=0)
        # ))

        """调试代码"""

        """清除已经绘制的路径"""
        self.draw.clear_points()
        self.draw.clear_lines()
        self.path = []
        """清除已经绘制的路径"""

        # self.console_logger.info('Start:设置目标物体在世界中的位置，positions[env_ids, :3]=%s，orientations=%s' % (
        # str(self._goals[env_ids, :3]), str(self._goals[env_ids, 3:])))

        # bookkeeping
        self._is_success[env_ids] = 0
        self._ik_fails[env_ids] = 0
        self._collided[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.extras[env_ids] = 0

    # 判断机器人是否与场景中的物品发生了碰撞
    def check_robot_collisions(self):
        # Check if the robot collided with an object
        # TODO: Parallelize
        # 遍历所有的障碍物，是否与robot间发生碰撞
        for obst in self._obstacles:
            # 获取与指定障碍物obst.prim_path相关的接触传感器的原始数据
            raw_readings = self._contact_sensor_interface.get_contact_sensor_raw_data(
                obst.prim_path + "/Contact_Sensor")
            # 检查是否有接触传感器的原始数据
            if raw_readings.shape[0]:
                # 遍历接触传感器的原始数据
                for reading in raw_readings:
                    # self.console_logger.info("reading_body0=%s"%(str(reading["body0"])))
                    # 检查机器人的某个部分是否与障碍物发生碰撞
                    if "Fetch" in str(self._contact_sensor_interface.decode_body_name(reading["body1"])):
                        # self.console_logger.info(
                        #     "self._contact_sensor_interface.decode_body_name(reading[\"body1\"])=%s" % (
                        #         str(self._contact_sensor_interface.decode_body_name(reading["body1"]))))
                        return True  # Collision detected with some part of the robot
                    if "Fetch" in str(self._contact_sensor_interface.decode_body_name(reading["body0"])):
                        # self.console_logger.info(
                        #     "self._contact_sensor_interface.decode_body_name(reading[\"body0\"])=%s" % (
                        #         str(self._contact_sensor_interface.decode_body_name(reading["body0"]))))
                        return True  # Collision detected with some part of the robot

        # 遍历所有的抓取对象，是否与robot间发生碰撞
        for grasp_obj in self._grasp_objs:
            # 如果当前抓取对象与检查对象相同，则跳过
            if grasp_obj == self._curr_grasp_obj:
                continue  # Important. Exclude current target object for collision checking，重要必须要排除目标物品本身
            # 获取与指定抓取对象grasp_obj相关的接触传感器的原始数据
            raw_readings = self._contact_sensor_interface.get_contact_sensor_raw_data(
                grasp_obj.prim_path + "/Contact_Sensor")
            if raw_readings.shape[0]:
                for reading in raw_readings:
                    if "Fetch" in str(self._contact_sensor_interface.decode_body_name(reading["body1"])):
                        return True  # Collision detected with some part of the robot
                    if "Fetch" in str(self._contact_sensor_interface.decode_body_name(reading["body0"])):
                        return True  # Collision detected with some part of the robot
        return False

    # 计算度量指标，包括奖励值的计算
    def calculate_metrics(self) -> None:
        # assuming data from obs buffer is available (get_observations() called before this function)
        # 如果发生碰撞直接给self._reward_collision=-0.1的惩罚
        if self.check_robot_collisions():  # TODO: Parallelize
            # Collision detected. Give penalty and no other rewards，第一部分是碰撞检测
            self._collided[0] = 1
            self._is_success[0] = 0  # Success isn't considered in this case
            reward = torch.tensor(self._reward_collision, device=self._device)
        else:
            # Distance reward，第二部分是距离奖励
            prev_goal_xy_dist = self._goals_xy_dist

            # 计算范数（距离）
            norm = np.linalg.norm(self.obs_buf[:, :2])
            # 转换为torch.tensor并传输到CUDA设备（如果需要）
            curr_goal_xy_dist = torch.tensor(norm, dtype=torch.float32).unsqueeze(0)

            # curr_goal_xy_dist = torch.linalg.norm(self.obs_buf[:, :2], dim=1)
            goal_xy_dist_reduction = (prev_goal_xy_dist - curr_goal_xy_dist).clone()
            reward = self._reward_dist_weight * goal_xy_dist_reduction

            # self.console_logger.info(
            #     "调试距离：prev_goal_xy_dist = %s, curr_goal_xy_dist=%s, goal_xy_dist_reduction=%s" % (
            #     str(prev_goal_xy_dist), str(curr_goal_xy_dist), str(goal_xy_dist_reduction)))
            #
            # self.console_logger.info('Debug5:距离障碍物的距离goal_xy_dist_reduction=%s' % goal_xy_dist_reduction)

            # print(f"Goal Dist reward: {reward}")
            self._goals_xy_dist = curr_goal_xy_dist

            # IK fail reward (penalty)，IK运动学查询失败惩罚
            reward += self._reward_noIK * self._ik_fails

            # Success reward，任务完成的成功奖励
            reward += self._reward_success * self._is_success

            # self.console_logger.info('reward1:%s, reward2:%s, reward3:%s, reward4:%s'%(str(self._reward_dist_weight * goal_xy_dist_reduction),
            #                                                                str(self._reward_noIK * self._ik_fails),
            #                                                                str(self._reward_success * self._is_success),
            #                                                                str(self._penalty_slack)))
        # print(f"Total reward: {reward}")
        self.rew_buf[:] = reward
        self.extras[:] = self._is_success.clone()  # Track success

    def is_done(self) -> None:
        # resets = torch.where(torch.abs(cart_pos) > self._reset_dist, 1, 0)
        # resets = torch.where(torch.abs(pole_pos) > np.pi / 2, 1, resets)
        # resets = torch.zeros(self._num_envs, dtype=int, device=self._device)

        # reset if success OR collided OR if reached max episode length
        # 如果成功到达或发生碰撞或到达最大轮数则进行重置
        resets = self._is_success.clone()
        resets = torch.where(self._collided.bool(), 1, resets)
        resets = torch.where(self.progress_buf >= self._max_episode_length, 1, resets)
        self.reset_buf[:] = resets

        # if resets:
        #     len_ = len(self.path)
        #     self.draw.draw_points(self.path, [(1, 0, 0, 1)] * len_, [5] * len_)
        #     self.refresh()
        #     self.draw.draw_lines(self.path[0:len_-1], self.path[1:len_], [(0, 0, 1, 1)] * (len_ - 1), [5] * (len_ - 1))
        #     self.refresh()
        #     self.get_render()
            # input("已经完成")
        # if resets:
        #     if self.progress_buf >= self._max_episode_length:
        #         self.console_logger.info('Debug6:结束，运行轮数超过最大限制%d'%(self._max_episode_length))
        #     else:
        #         self.console_logger.info('Debug7:抓到物体，成功完成')
        #     self.console_logger.info('====================================================================================')

