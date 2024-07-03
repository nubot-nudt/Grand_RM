import torch
import math

from learned_robot_placement.handlers.base.fetchhandler import FetchBaseHandler
from learned_robot_placement.robots.articulations.fetch import FetchRobot
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.torch.maths import torch_rand_float, tensor_clamp
from omni.isaac.core.utils.stage import get_current_stage
from mushroom_rl.core.logger.console_logger import ConsoleLogger
from scipy.spatial.transform import Rotation

# Whole Body robot handler for the dual-armed Fetch robot
class FetchWBHandler(FetchBaseHandler):
    def __init__(self, move_group, use_torso, sim_config, num_envs, device):
        # self.task = task_class
        self._move_group = move_group
        self._use_torso = use_torso
        self._sim_config = sim_config
        self._num_envs = num_envs
        self._robot_positions = torch.tensor([0, 0, 0])  # placement of the robot in the world，机器人在当前世界坐标系下的位置
        self._device = device

        # Custom default values of arm joints
        # middle of joint ranges
        # Start at 'home' positions 设置起始位置
        self.arm_start = torch.tensor([-1.44, -1.06, -2.60,
                                            -1.00, -0.27, -1.62, 0.23], device=self._device)
        # Opened gripper by default，默认打开机械臂
        self.gripper_start = torch.tensor([0.05, 0.05], device=self._device)
        self.torso_fixed_state = torch.tensor([0.15], device=self._device)

        self.default_zero_env_path = "/World/envs/env_0"

        # self.max_arm_vel = torch.tensor(self._sim_config.task_config["env"]["max_rot_vel"], device=self._device)
        # self.max_base_rot_vel = torch.tensor(self._sim_config.task_config["env"]["max_rot_vel"], device=self._device)
        # self.max_base_xy_vel = torch.tensor(self._sim_config.task_config["env"]["max_base_xy_vel"], device=self._device)
        # Get dt for integrating velocity commands
        self.dt = torch.tensor(
            self._sim_config.task_config["sim"]["dt"] * self._sim_config.task_config["env"]["controlFrequencyInv"],
            device=self._device)

        # articulation View will be created later
        self.robots = None

        # joint names
        self._base_joint_names = ["l_wheel_joint",
                                  "r_wheel_joint"]
        self._torso_joint_name = ["torso_lift_joint"]
        self._arm_names = ["shoulder_pan_joint",
                           "shoulder_lift_joint",
                           "upperarm_roll_joint",
                           "elbow_flex_joint",
                           "forearm_roll_joint",
                           "wrist_flex_joint",
                           "wrist_roll_joint"]

        # Future: Use end-effector link names and get their poses and velocities from Isaac
        self.ee_prim = ["gripper_grasping_frame"]
        # self.ee_left_tf =  torch.tensor([[ 0., 0.,  1.,  0.      ], # left_7_link to ee tf
        #                                  [ 0., 1.,  0.,  0.      ],
        #                                  [-1., 0.,  0., -0.196575],
        #                                  [ 0., 0.,  0.,  1.      ]])
        # self.ee_right_tf = torch.tensor([[ 0., 0., -1.,  0.      ], # right_7_link to ee tf
        #                                  [ 0., 1.,  0.,  0.      ],
        #                                  [ 1., 0.,  0.,  0.196575],
        #                                  [ 0., 0.,  0.,  1.      ]])
        self._gripper_names = ["l_gripper_finger_joint", "r_gripper_finger_joint"]

        # values are set in post_reset after model is loaded，设置位置索引值
        # self.base_dof_idxs = []
        self.torso_dof_idx = []
        self.arm_dof_idxs = []
        self.gripper_dof_idxs = []
        self.upper_body_dof_idxs = []
        self.combined_dof_idxs = []

        # dof joint position limits
        self.torso_dof_lower = []
        self.torso_dof_upper = []
        self.arm_dof_lower = []
        self.arm_dof_upper = []

        # 用于调试打印备份信息
        self.console_logger = ConsoleLogger(log_name='')

    # "将弧度转换为四元数的函数"
    def rad_to_quaternion(self, angle_rad):
        angle_rad = angle_rad % (2 * torch.pi)
        if angle_rad > torch.pi:
            angle_rad -= 2 * torch.pi
        elif angle_rad < -torch.pi:
            angle_rad += 2 * torch.pi
        angle_rad2 = -angle_rad + torch.pi
        rotation = Rotation.from_euler('x', angle_rad2, degrees=False)
        quaternion = rotation.as_quat()[0, :]  # 将旋转矩阵转换为四元数
        quaternion = torch.tensor(quaternion)
        return quaternion

    def quaternion_to_angle(self, quaternion):
        rotation = Rotation.from_quat(quaternion)
        euler_angles = rotation.as_euler('xzy')
        angle_rad = torch.tensor(euler_angles[0])
        angle_rad = -angle_rad + torch.pi
        angle_rad = angle_rad % (2 * torch.pi)
        # print("quaternion=%s, angle_rad=%s"%(quaternion, angle_rad))
        if angle_rad > torch.pi:
            angle_rad -= 2 * torch.pi
        elif angle_rad < -torch.pi:
            angle_rad += 2 * torch.pi
        return angle_rad

    def get_robot(self):
        # make it in task and use handler as getter for path
        self.fetch_prim = FetchRobot(prim_path=self.default_zero_env_path + "/Fetch", name="Fetch",
                              translation=self._robot_positions)
        # Optional: Apply additional articulation settings
        self._sim_config.apply_articulation_settings("Fetch", get_prim_at_path(self.fetch_prim.prim_path),
                                                     self._sim_config.parse_actor_config("Fetch"))

    # call it in setup_up_scene in Task
    def create_articulation_view(self):
        self.robots = ArticulationView(prim_paths_expr="/World/envs/.*/Fetch", name="fetch_view")
        return self.robots

    def post_reset(self):
        # reset that takes place when the isaac world is reset (typically happens only once)
        # add dof indexes
        self._set_dof_idxs()
        # set dof limits
        self._set_dof_limits()
        # set new default state for reset
        self._set_default_state()
        # get stage
        self._stage = get_current_stage()

    # 设置索引值对应到时候的运动
    def _set_dof_idxs(self):
        # [self.base_dof_idxs.append(self.robots.get_dof_index(name)) for name in self._base_joint_names]
        [self.torso_dof_idx.append(self.robots.get_dof_index(name)) for name in self._torso_joint_name]
        [self.arm_dof_idxs.append(self.robots.get_dof_index(name)) for name in self._arm_names]
        [self.gripper_dof_idxs.append(self.robots.get_dof_index(name)) for name in self._gripper_names]
        self.upper_body_dof_idxs = [] # 可能包含torse也可能不包含
        if self._use_torso:
            self.upper_body_dof_idxs += self.torso_dof_idx

        if self._move_group == "arm":
            self.upper_body_dof_idxs += self.arm_dof_idxs
        else:
            raise ValueError('move_group not defined')
        # Future: Add end-effector prim paths
        self.combined_dof_idxs = self.upper_body_dof_idxs

    # 设置关节的位置限制
    def _set_dof_limits(self):  # dof position limits
        # (num_envs, num_dofs, 2)
        dof_limits = self.robots.get_dof_limits()
        dof_limits_lower = dof_limits[0, :, 0].to(self._device)
        dof_limits_upper = dof_limits[0, :, 1].to(self._device)

        # set relevant joint position limit values
        self.torso_dof_lower = dof_limits_upper[self.torso_dof_idx]
        self.torso_dof_upper = dof_limits_upper[self.torso_dof_idx]
        self.arm_dof_lower = dof_limits_lower[self.arm_dof_idxs]
        self.arm_dof_upper = dof_limits_upper[self.arm_dof_idxs]
        # self.gripper_dof_lower = dof_limits_lower[self.gripper_idxs]
        # Holo base has no limits

    # 设置默认动作
    def _set_default_state(self):
        joint_states = self.robots.get_joints_default_state()
        jt_pos = joint_states.positions
        jt_pos[:, self.torso_dof_idx] = self.torso_fixed_state
        jt_pos[:, self.arm_dof_idxs] = self.arm_start
        jt_pos[:, self.gripper_dof_idxs] = self.gripper_start
        # 用于设置默认值
        self.robots.set_joints_default_state(positions=jt_pos)
        self.set_base_positions(torch.tensor([[0.0, 0.0, 0.0]]))

    # 运动部分
    def apply_actions(self, actions):
        # Actions are velocity commands
        # The first three actions are the base velocities
        self.apply_base_actions(actions=actions[:, :2])
        self.apply_upper_body_actions(actions=actions[:, 2:])

    # 执行对应的上身动作
    def apply_upper_body_actions(self, actions):
        # Apply actions as per the selected upper_body_dof_idxs (move_group)
        # Velocity commands (rad/s) will be converted to next-position (rad) targets
        jt_pos = self.robots.get_joint_positions(joint_indices=self.upper_body_dof_idxs, clone=True)
        jt_pos += actions * self.dt  # create new position targets
        # self.robots.set_joint_position_targets(positions=jt_pos, joint_indices=self.upper_body_dof_idxs)
        # TEMP: Use direct joint positions
        self.robots.set_joint_positions(positions=jt_pos, joint_indices=self.upper_body_dof_idxs)
        if not self._use_torso:
            # Hack to avoid torso falling when it isn't controlled
            pos = self.torso_fixed_state.unsqueeze(dim=0)
            self.robots.set_joint_positions(positions=pos, joint_indices=self.torso_dof_idx)
        # if self._move_group == "arm":  # Hack to avoid arm falling when it isn't controlled
        #     pos = self.arm_start.unsqueeze(dim=0)
        #     self.robots.set_joint_positions(positions=pos, joint_indices=self.arm_right_dof_idxs)
        # elif self._move_group == "both_arms":
        #     pass

    # 执行对应的底盘动作
    def apply_base_actions(self, actions):
        base_actions = actions.clone()

        # self.robots.set_joint_velocity_targets(velocities=base_actions, joint_indices=self.base_dof_idxs)
        # TEMP: Use direct joint positions
        position, orientation = self.fetch_prim.get_world_pose()

        position += base_actions[:, :2] * self.dt
        delta = base_actions[:, :3] * self.dt
        orientation += self.rad_to_quaternion(delta)

        self.robots.set_base_positions(position, orientation)

    # 设置上半身位置
    def set_upper_body_positions(self, jnt_positions):
        # Set upper body joints to specific positions
        self.robots.set_joint_positions(positions=jnt_positions, joint_indices=self.upper_body_dof_idxs)
        # if not self._use_torso:
        #     # Hack to avoid torso falling when it isn't controlled
        #     pos = self.torso_fixed_state.unsqueeze(dim=0)
        #     self.robots.set_joint_positions(positions=pos, joint_indices=self.torso_dof_idx)
        # if self._move_group == "arm_left": # Hack to avoid arm falling when it isn't controlled
        #     pos = self.arm_right_start.unsqueeze(dim=0)
        #     self.robots.set_joint_positions(positions=pos, joint_indices=self.arm_right_dof_idxs)
        # elif self._move_group == "arm_right": # Hack to avoid arm falling when it isn't controlled
        #     pos = self.arm_left_start.unsqueeze(dim=0)
        #     self.robots.set_joint_positions(positions=pos, joint_indices=self.arm_left_dof_idxs)
        # elif self._move_group == "both_arms":
        #     pass

    # 设置底盘位置
    def set_base_positions(self, base_positions):
        # Set base joints to specific positions
        base_xy, base_theta = base_positions[:, :2], base_positions[:, 2]
        _orientation = self.rad_to_quaternion(base_theta)
        self.fetch_prim.set_world_pose(position=torch.hstack((base_xy, torch.zeros(1, 1))), orientation=_orientation)

    # 获取机器人当前位置
    def get_robot_obs(self):
        # 给定张量的形状
        _shape = (1, 11)  # 与给定张量的形状相同

        # 创建一个与给定张量相同维度且值全为0的张量
        combined_pos = torch.zeros(_shape)
        combined_vel = torch.zeros(_shape)

        # return positions and velocities of upper body and base joints
        # 获取所有的关节和位置
        base_pos, base_orientation = self.fetch_prim.get_world_pose()
        base_rad = self.quaternion_to_angle(base_orientation)
        upper_pos = self.robots.get_joint_positions(joint_indices=self.upper_body_dof_idxs, clone=True)

        # print("Debug:get_robot_obs:base_rad=%s, base_orientation=%s" % (str(base_rad), str(base_orientation)))
        combined_pos[0, :2] = base_pos[:2]
        combined_pos[0, 2] = base_rad
        combined_pos[0, 3:] = upper_pos

        # NOTE: Velocities here will only be correct if the correct control mode is used!!

        # base_xy_velocity = self.fetch_prim.get_linear_velocity()
        # print("Debug:base_xy_velocity=%s"%(str(base_xy_velocity)))
        # base_xy_velocity = torch.tensor(base_xy_velocity)
        # base_theta_angular = self.fetch_prim.get_angular_velocity()
        upper_velocity = self.robots.get_joint_velocities(joint_indices=self.upper_body_dof_idxs, clone=True)

        # combined_vel[0, :2] = base_xy_velocity[:, :2]
        combined_vel[0, 3:] = upper_velocity
        # Future: Add pose and velocity of end-effector from Isaac prims
        return torch.hstack((combined_pos, combined_vel))

    # 获取机械臂当前角度
    def get_arms_dof_pos(self):
        # (num_envs, num_dof)
        arm_pos = self.robots.get_joint_positions(joint_indices=self.arm_dof_idxs, clone=False)
        return arm_pos

    # 获取机械臂当前速度
    def get_arms_dof_vel(self):
        # (num_envs, num_dof)
        arm_vel = self.robots.get_joint_velocities(joint_indices=self.arm_dof_idxs, clone=False)
        return arm_vel

    # 获取底盘当前位置和速度
    def get_base_dof_values(self):
        # 给定张量的形状
        _shape = (1, 11)  # 与给定张量的形状相同

        # 创建一个与给定张量相同维度且值全为0的张量
        base_pos = torch.zeros(_shape)
        base_vel = torch.zeros(_shape)

        _pos, _orientation = self.fetch_prim.get_world_pose()
        _rad = self.quaternion_to_angle(_orientation)
        # _xy_velocity = self.fetch_prim.get_linear_velocity()
        # _theta_angular = self.fetch_prim.get_angular_velocity()

        base_pos[0, 0:2] = _pos[:2]
        base_pos[0, 3] = _rad
        # base_vel[0, 0:2] = _xy_velocity[:, :2]
        return base_pos, base_vel

    # 获取torso躯干位置和速度
    def get_torso_dof_values(self):
        torso_pos = self.robots.get_joint_positions(joint_indices=self.torso_dof_idx, clone=False)
        torso_vel = self.robots.get_joint_velocities(joint_indices=self.torso_dof_idx, clone=False)
        return torso_pos, torso_vel

    # 进行重置
    def reset(self, indices, randomize=False, loc=None, theta=None):
        num_resets = len(indices)
        joint_states = self.robots.get_joints_default_state()
        jt_pos = joint_states.positions.clone()
        # self.console_logger.info("jt_pos=%s"%(jt_pos))
        jt_pos = jt_pos[0:num_resets]  # we need only num_resets rows
        ### 先关闭randomize否则会出现尤其是对torse而言-0.75过大
        randomize = False
        # self.console_logger.info("jt_pos=%s"%(jt_pos))
        if randomize:
            noise = torch_rand_float(-0.75, 0.75, jt_pos[:, self.upper_body_dof_idxs].shape, device=self._device)
            # Clip needed? dof_pos[:] = tensor_clamp(self.initial_dof_pos[env_ids] + dof_pos, self.dof_limits_lower, self.dof_limits_upper)
            # jt_pos[:, self.upper_body_dof_idxs] = noise
            jt_pos[:, self.upper_body_dof_idxs] += noise  # Optional: Add to default instead
        self.robots.set_joint_positions(jt_pos, indices=indices)
        # 在这里需要重新设置robot的位置
        if loc is None:
            self.set_base_positions(torch.tensor([[0.0, 0.0, 0.0]]))
            # self.set_base_positions(torch.tensor([[0.62719, 1.6400, 0.0]]))
            # input("设置机器人位置")
        else:
            self.set_base_positions(torch.hstack((loc, theta)))