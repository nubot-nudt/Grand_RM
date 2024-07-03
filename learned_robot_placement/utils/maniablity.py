import numpy as np
from urdfpy import URDF


class ManipulabilityCalculator:
    def __init__(self, urdf_path):
        try:
            self.robot = URDF.load(urdf_path)
        except Exception as e:
            print(f"Error loading URDF file: {e}")
            raise
        self.num_joints = None
        self.joint_axes = None
        self.end_effector_index = None
        self.joint_names = ["shoulder_pan_joint",
                            "shoulder_lift_joint",
                            "upperarm_roll_joint",
                            "elbow_flex_joint",
                            "forearm_roll_joint",
                            "wrist_flex_joint",
                            "wrist_roll_joint"]
        self.set_joint_information()

    def set_joint_information(self):
        self.num_joints = len(self.joint_names)
        self.joint_axes = np.zeros((self.num_joints, 3))
        i = 0

        for joint in self.robot.joints:
            if joint.name in self.joint_names:
                self.joint_axes[i] = np.array(joint.axis)
                i += 1

        if i != self.num_joints:
            print("Warning: Not all joint axes were found in the URDF file.")

        self.end_effector_index = self.num_joints - 1

    def unit_vector(self, data, axis=None, out=None):
        if out is None:
            data = np.array(data, dtype=np.float64, copy=True)
            if data.ndim == 1:
                data /= np.sqrt(np.dot(data, data))
                return data
        else:
            if out is not data:
                out[:] = np.array(data, copy=False)
            data = out
        length = np.atleast_1d(np.sum(data * data, axis))
        np.sqrt(length, length)
        if axis is not None:
            length = np.expand_dims(length, axis)
        data /= length
        if out is None:
            return data

    def rotation_matrix(self, angle, direction, point=None):
        sina = np.sin(angle)
        cosa = np.cos(angle)
        direction = self.unit_vector(direction[:3])
        R = np.array(((cosa, 0.0, 0.0),
                      (0.0, cosa, 0.0),
                      (0.0, 0.0, cosa)), dtype=np.float64)
        R += np.outer(direction, direction) * (1.0 - cosa)
        direction *= sina
        R += np.array(((0.0, -direction[2], direction[1]),
                       (direction[2], 0.0, -direction[0]),
                       (-direction[1], direction[0], 0.0)),
                      dtype=np.float64)
        M = np.identity(4)
        M[:3, :3] = R
        if point is not None:
            point = np.array(point[:3], dtype=np.float64, copy=False)
            M[:3, 3] = point - np.dot(R, point)
        return M

    def compute_jacobian(self, joint_positions):
        jacobian = np.zeros((6, self.num_joints))
        current_transform = np.eye(4)

        for i in range(self.num_joints):
            joint_axis = self.joint_axes[i]
            joint_angle = joint_positions[i]
            joint_transform = self.rotation_matrix(joint_angle, joint_axis)
            current_transform = np.dot(current_transform, joint_transform)
            z_axis = current_transform[:3, 2]
            position = current_transform[:3, 3]
            jacobian[:3, i] = np.cross(z_axis, position)
            jacobian[3:, i] = z_axis

        return jacobian[:, :self.num_joints]

    def compute_dexterity(self, jacobian):
        condition_number = np.linalg.cond(jacobian)
        dexterity = 1.0 / condition_number
        return dexterity

    def compute_manipulability(self, jacobian):
        JTJ = np.dot(jacobian.T, jacobian)
        manipulability = np.sqrt(np.linalg.det(JTJ))
        return manipulability

    def compute_coi(self, joint_positions, w1=0.5, w2=0.5):
        jacobian = self.compute_jacobian(joint_positions)
        dexterity = self.compute_dexterity(jacobian)
        manipulability = self.compute_manipulability(jacobian)
        COI = w1 * dexterity + w2 * manipulability
        return COI, dexterity, manipulability


if __name__ == "__main__":
    urdf_path = "/home/lu/Desktop/embodied_ai/rlmmbp/learned_robot_placement/urdf/fetch.urdf"  # Adjust this to the path of your URDF file
    joint_positions = [0.0, -1.0, 1.0, 0.5, 0.5, 0.0, 0.0]  # Example joint positions
    w1 = 0.5  # 设置灵巧性的权重
    w2 = 0.5  # 设置可操作性的权重

    try:
        calculator = ManipulabilityCalculator(urdf_path)
        COI, dexterity, manipulability = calculator.compute_coi(joint_positions, w1, w2)
        print(f"综合指标COI: {COI}")
        print(f"灵巧性指标: {dexterity}")
        print(f"可操作性指标: {manipulability}")
    except Exception as e:
        print(f"An error occurred: {e}")
