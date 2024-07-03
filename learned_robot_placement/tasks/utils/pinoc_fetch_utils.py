from __future__ import print_function
import numpy as np
import os
# pin是一个基于线性代数的特征值和碰撞检测的FCL
import pinocchio as pin
from scipy.spatial.transform import Rotation
from mushroom_rl.core.logger.console_logger import ConsoleLogger

def get_se3_err(pos_first, quat_first, pos_second, quat_second):
    # Retruns 6 dimensional log.SE3 error between two poses expressed as position and quaternion rotation
    
    rot_first = Rotation.from_quat(np.array([quat_first[1],quat_first[2],quat_first[3],quat_first[0]])).as_matrix() # Quaternion in scalar last format!!!
    rot_second = Rotation.from_quat(np.array([quat_second[1],quat_second[2],quat_second[3],quat_second[0]])).as_matrix() # Quaternion in scalar last format!!!
    
    oMfirst = pin.SE3(rot_first, pos_first)
    oMsecond = pin.SE3(rot_second, pos_second)
    firstMsecond = oMfirst.actInv(oMsecond)
    
    return pin.log(firstMsecond).vector # log gives us a spatial vector (exp co-ords)


class PinFetchIKSolver(object):
    def __init__(
        self,
        urdf_name: str = "fetch.urdf",
        move_group: str = "arm", # Can be 'arm_right' or 'arm_left'
        include_torso: bool = False, # Use torso in th IK solution
        include_base: bool = False, # Use base in th IK solution
        max_rot_vel: float = 1.0472
    ) -> None:
        # 用于调试打印备份信息
        self.console_logger = ConsoleLogger(log_name='')

        # Settings
        self.damp = 1e-10 # Damping co-efficient for linalg solve (to avoid singularities)
        self._include_torso = include_torso
        self._include_base = include_base
        self.max_rot_vel = max_rot_vel # Maximum rotational velocity of all joints
        
        ## Load urdf
        urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/" + urdf_name
        # self.console_logger.info('urdf_file=%s'%urdf_file)
        self.model = pin.buildModelFromUrdf(urdf_file)
        # Choose joints
        # name_end_effector = "gripper_grasping_frame"

        # JL修改
        name_end_effector = 'gripper_link'

        # left_7_link to ee tf = [[0., 0.,  1.,  0.      ],
        #                    [ 0., 1.,  0.,  0.      ],
        #                    [-1., 0.,  0., -0.196575],
        #                    [ 0., 0.,  0.,  1.      ]]
        # right_7_link to ee tf = [[0., 0., -1.,  0.      ],
        #                    [0., 1.,  0.,  0.      ],
        #                    [1., 0.,  0.,  0.196575],
        #                    [0., 0.,  0.,  1.      ]]
        # # name_base_link = "base_footprint"#"world"

        # JL修改关节的名称
        jointsOfInterest = ["shoulder_pan_joint",
                            "shoulder_lift_joint",
                            "upperarm_roll_joint",
                            "elbow_flex_joint",
                            "forearm_roll_joint",
                            "wrist_flex_joint",
                            "wrist_roll_joint"]

        if self._include_torso:
            # Add torso joint
            jointsOfInterest = ['torso_lift_joint'] + jointsOfInterest
        if self._include_base:
            # Add base joints
            jointsOfInterest = ['base_joint1',
                                'base_joint2',] + jointsOfInterest  # 10 DOF with holo base joints included (11 with torso)

        remove_ids = list()
        for jnt in jointsOfInterest:
            if self.model.existJointName(jnt):
                remove_ids.append(self.model.getJointId(jnt))
            else:
                print('[IK WARNING]: joint ' + str(jnt) + ' does not belong to the model!')

        jointIdsToExclude = np.delete(np.arange(0,self.model.njoints), remove_ids)
        # Lock extra joints except joint 0 (root)
        reference_configuration=pin.neutral(self.model)

        # JL修改torse的索引值
        if not self._include_torso:
            reference_configuration[11] = 0.25 # lock torso_lift_joint at 0.25

        self.model = pin.buildReducedModel(self.model, jointIdsToExclude[1:].tolist(), reference_configuration=reference_configuration)
        assert (len(self.model.joints)==(len(jointsOfInterest)+1)), "[IK Error]: Joints != nDoFs"
        self.model_data = self.model.createData()

        # Define Joint-Limits，JL修改
        self.joint_pos_min = np.array(
            [-1.6055999, -1.2209998, -3.1415926, -2.2509999, -3.1415926, -2.1599998, -3.1415926])
        self.joint_pos_max = np.array(
            [+1.6055999, +1.5179999, +3.1415926, +2.2509999, +3.1415926, +2.1599998, +3.1415926])

        if self._include_torso:
            self.joint_pos_min = np.hstack((np.array([0.0]), self.joint_pos_min))  # hstack是水平方向的连接，放在水平方向
            self.joint_pos_max = np.hstack((np.array([3.8615000e-01]), self.joint_pos_max))
        if self._include_base:
            self.joint_pos_min = np.hstack((np.array([-100.0, -100.0]), self.joint_pos_min))
            self.joint_pos_max = np.hstack((np.array([+100.0, +100.0]), self.joint_pos_max))
        self.joint_pos_mid = (self.joint_pos_max + self.joint_pos_min) / 2.0

        # Get End Effector Frame ID
        self.id_EE = self.model.getFrameId(name_end_effector)


    def solve_fk_fetch(self, curr_joints):
        pin.framesForwardKinematics(self.model,self.model_data,curr_joints)
        oMf = self.model_data.oMf[self.id_EE]
        ee_pos = oMf.translation
        ee_quat = pin.Quaternion(oMf.rotation)

        return ee_pos, np.array([ee_quat.w,ee_quat.x,ee_quat.y,ee_quat.z])

    def solve_ik_pos_fetch(self, des_pos, des_quat, curr_joints=None, n_trials=7, dt=0.1, pos_threshold=0.05, angle_threshold=15.*np.pi/180, verbose=False):
        """

            :param des_pos: 目标位置
            :param des_quat: 目标姿态
            :param curr_joints: 当前关节角度
            :param n_trials: 默认为7步
            :param dt: 时间步长
            :param pos_threshold:位置阈值
            :param angle_threshold: 角度阈值
            :param verbose: 是否输出详细信息
            :return:success: 表示成功的标志
            :return:best_q：表示最佳采取的动作，包括底盘和手臂
            """

        # self.console_logger.info('des_pos=%s, des_quat=%s'%(str(des_pos), str(des_quat)))
        # self.console_logger.info('type(des_pos)=%s, type(des_quat)=%s'%(str(type(des_pos)), str(type(des_quat))))

        # Get IK positions for fetch robot，设置了一个微小的阻尼项，用于数值的稳定性
        damp = 1e-10
        success = False
        dex = 0

        if des_quat is not None:
            # quaternion to rot matrix，将目标四元数转换为旋转矩阵，创建了oMdes目标的位姿
            des_rot = Rotation.from_quat(np.array([des_quat[1],des_quat[2],des_quat[3],des_quat[0]])).as_matrix() # Quaternion in scalar last format!!!
            oMdes = pin.SE3(des_rot, des_pos)
        else:
            # 3D position error only，仅计算位置误差
            des_rot = None

        if curr_joints is None:
            # 随机成成关节角度
            q = np.random.uniform(self.joint_pos_min, self.joint_pos_max)

        # self.console_logger.info('=====q=%s======' % (str(q)))

        # 多次尝试，在每次尝试中通过迭代优化逼近目标姿态和位置
        for n in range(n_trials):
            for i in range(800):
                # self.console_logger.info('debug:model_data=%s'%(str(self.model_data.oMf[self.id_EE])))

                # self.console_logger.info('2debug_pinoc_utils:self.model=%s, self.model_data=%s, q=%s' % (
                # str(self.model), str(self.model_data), str(q)))
                # First calls the forwardKinematics on the model, then computes the placement of each frame.
                # Update the joint placements according to the current joint configuration.
                pin.framesForwardKinematics(self.model,self.model_data, q)
                # 计算当前关节角度对应的末端执行器位姿oMf
                oMf = self.model_data.oMf[self.id_EE]
                if des_rot is None:
                    oMdes = pin.SE3(oMf.rotation, des_pos) # Set rotation equal to current rotation to exclude this error
                dMf = oMdes.actInv(oMf)
                # 计算目标姿态和当前位姿的误差err
                err = pin.log(dMf).vector
                # 如果误差满足设定的阈值条件，则认为成功找到解，并跳出迭代
                if (np.linalg.norm(err[0:3]) < pos_threshold) and (np.linalg.norm(err[3:6]) < angle_threshold):
                    success = True
                    J = pin.computeFrameJacobian(self.model, self.model_data, q, self.id_EE)
                    dex = self.compute_dexterity(J)
                    break
                # 如果误差不满足条件，则计算雅各比矩阵`J`并求解关节速度`v`，通过数值积分更新关节角度
                J = pin.computeFrameJacobian(self.model,self.model_data, q, self.id_EE)

                if des_rot is None:
                    J = J[:3,:] # Only pos errors
                    err = err[:3]
                v = - J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))
                q = pin.integrate(self.model,q, v*dt)
                # Clip q to within joint limits，同时将q限制在关节角度的范围内
                q = np.clip(q, self.joint_pos_min, self.joint_pos_max)

                # 如果设置了verboase，则在迭代过程中输出迭代次数、误差等信息
                if verbose:
                    if not i % 100:
                        print('Trial %d: iter %d: error = %s' % (n+1, i, err.T))
                    i += 1

            # 如果成功找到解，则返回`success`为True，以及最终的关节角度`best_q`
            if success:
                best_q = np.array(q)
                break
            else: # 否则返回False，并输出警告信息
                # Save current solution
                best_q = np.array(q)
                # Reset q to random configuration
                q = np.random.uniform(self.joint_pos_min, self.joint_pos_max)
        if verbose:
            if success:
                print("[[[[IK: Convergence achieved!]]]")
            else:
                print("[Warning: the IK iterative algorithm has not reached convergence to the desired precision]")
        
        return success, best_q, dex


    def compute_dexterity(self, jacobian):
        condition_number = np.linalg.cond(jacobian)
        dexterity = 1.0 / condition_number
        return dexterity
