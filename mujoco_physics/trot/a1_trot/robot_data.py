import numpy as np
import utility as ram

class Robot:
    class Body:
        """Defines the body (links) of the robot."""
        def __init__(self, parent, name, pos, quat, ipos, iquat, mass, inertia, joint_axis, joint_range):
            self.parent = parent
            self.name = name
            self.pos = np.array(pos)
            self.quat = np.array(quat)
            self.ipos = np.array(ipos)
            self.iquat = np.array(iquat)
            self.mass = mass
            self.inertia = np.array(inertia)
            self.joint_axis = np.array(joint_axis)
            self.joint_range = np.array(joint_range)

    class Params:
        """Class to hold heterogeneous robot-level parameters."""
        def __init__(self):
            self.end_eff_pos_local = np.array([0, 0, -0.2])
            self.q_base = np.array([0, 0, 0.27, 1, 0, 0, 0])  # Orientation: [1, 0, 0, 0] (body frame)


    def __init__(self):
        self.body = {}
        self.params = Robot.Params()  # Initialize robot parameters

    def add_body(self, body_id, parent, name, pos, quat, ipos, iquat, mass, inertia, joint_axis, joint_range):
        self.body[body_id] = Robot.Body(parent, name, pos, quat, ipos, iquat, mass, inertia, joint_axis, joint_range)


# Initialize the robot
robot = Robot()

# Add body parts
robot.add_body(
    1, parent='ground', name='trunk', pos=[0, 0, 0.43],
    quat=[1, 0, 0, 0], ipos=[0, 0.0041, -0.0005], iquat=[1, 0, 0, 0],
    mass=4.713, inertia=[0.0158533, 0.0377999, 0.0456542, -3.66e-05, -6.11e-05, -2.75e-05],
    joint_axis=[0, 0, 0, 1, 0, 0, 0], joint_range=[-6.28319, 6.28319]
)

robot.add_body(
    2, parent='trunk', name='FR_hip_joint', pos=[0.183, -0.047, 0],
    quat=[1, 0, 0, 0], ipos=[-0.003311, -0.000635, 3.1e-05], iquat=[0.507528, 0.506268, 0.491507, 0.494499],
    mass=0.696, inertia=[0.000807752, 0.00055293, 0.000468983],
    joint_axis=[1, 0, 0], joint_range=[-0.802851, 0.802851]
)

robot.add_body(
    3, parent='FR_hip_joint', name='FR_thigh_joint', pos=[0, -0.08505, 0],
    quat=[1, 0, 0, 0], ipos=[-0.003237, 0.022327, -0.027326], iquat=[0.999125, -0.00256393, -0.0409531, -0.00806091],
    mass=1.013, inertia=[0.00555739, 0.00513936, 0.00133944],
    joint_axis=[0, 1, 0], joint_range=[-1.0472, 4.18879]
)

robot.add_body(
    4, parent='FR_thigh_joint', name='FR_calf_joint', pos=[0, 0, -0.2],
    quat=[1, 0, 0, 0], ipos=[0.00472659, 0, -0.131975], iquat=[0.706886, 0.017653, 0.017653, 0.706886],
    mass=0.226, inertia=[0.00340344, 0.00339393, 3.54834e-05],
    joint_axis=[0, 1, 0], joint_range=[-2.69653, -0.916298]
)

# Adding FL (Front Left) joints
robot.add_body(
    5, parent='trunk', name='FL_hip_joint', pos=[0.183, 0.047, 0],
    quat=[1, 0, 0, 0], ipos=[-0.003311, 0.000635, 3.1e-05], iquat=[0.494499, 0.491507, 0.506268, 0.507528],
    mass=0.696, inertia=[0.000807752, 0.00055293, 0.000468983],
    joint_axis=[1, 0, 0], joint_range=[-0.802851, 0.802851]
)


robot.add_body(
    6, parent='FL_hip_joint', name='FL_thigh_joint', pos=[0, 0.08505, 0],
    quat=[1, 0, 0, 0], ipos=[-0.003237, -0.022327, -0.027326], iquat=[0.999125, 0.00256393, -0.0409531, 0.00806091],
    mass=1.013, inertia=[0.00555739, 0.00513936, 0.00133944],
    joint_axis=[0, 1, 0], joint_range=[-1.0472, 4.18879]
)

robot.add_body(
    7, parent='FL_thigh_joint', name='FL_calf_joint', pos=[0, 0, -0.2],
    quat=[1, 0, 0, 0], ipos=[0.00472659, 0, -0.131975], iquat=[0.706886, 0.017653, 0.017653, 0.706886],
    mass=0.226, inertia=[0.00340344, 0.00339393, 3.54834e-05],
    joint_axis=[0, 1, 0], joint_range=[-2.69653, -0.916298]
)

# Adding RR (Rear Right) joints
robot.add_body(
    8, parent='trunk', name='RR_hip_joint', pos=[-0.183, -0.047, 0],
    quat=[1, 0, 0, 0], ipos=[0.003311, -0.000635, 3.1e-05], iquat=[0.494499, 0.491507, 0.506268, 0.507528],
    mass=0.696, inertia=[0.000807752, 0.00055293, 0.000468983],
    joint_axis=[1, 0, 0], joint_range=[-0.802851, 0.802851]
)

robot.add_body(
    9, parent='RR_hip_joint', name='RR_thigh_joint', pos=[0, -0.08505, 0],
    quat=[1, 0, 0, 0], ipos=[-0.003237, 0.022327, -0.027326], iquat=[0.999125, -0.00256393, -0.0409531, -0.00806091],
    mass=1.013, inertia=[0.00555739, 0.00513936, 0.00133944],
    joint_axis=[0, 1, 0], joint_range=[-1.0472, 4.18879]
)

robot.add_body(
    10, parent='RR_thigh_joint', name='RR_calf_joint', pos=[0, 0, -0.2],
    quat=[1, 0, 0, 0], ipos=[0.00472659, 0, -0.131975], iquat=[0.706886, 0.017653, 0.017653, 0.706886],
    mass=0.226, inertia=[0.00340344, 0.00339393, 3.54834e-05],
    joint_axis=[0, 1, 0], joint_range=[-2.69653, -0.916298]
)

# Adding RL (Rear Left) joints
robot.add_body(
    11, parent='trunk', name='RL_hip_joint', pos=[-0.183, 0.047, 0],
    quat=[1, 0, 0, 0], ipos=[0.003311, 0.000635, 3.1e-05], iquat=[0.507528, 0.506268, 0.491507, 0.494499],
    mass=0.696, inertia=[0.000807752, 0.00055293, 0.000468983],
    joint_axis=[1, 0, 0], joint_range=[-0.802851, 0.802851]
)


robot.add_body(
    12, parent='RL_hip_joint', name='RL_thigh_joint', pos=[0, 0.08505, 0],
    quat=[1, 0, 0, 0], ipos=[-0.003237, -0.022327, -0.027326], iquat=[0.999125, 0.00256393, -0.0409531, 0.00806091],
    mass=1.013, inertia=[0.00555739, 0.00513936, 0.00133944],
    joint_axis=[0, 1, 0], joint_range=[-1.0472, 4.18879]
)

robot.add_body(
    13, parent='RL_thigh_joint', name='RL_calf_joint', pos=[0, 0, -0.2],
    quat=[1, 0, 0, 0], ipos=[0.00472659, 0, -0.131975], iquat=[0.706886, 0.017653, 0.017653, 0.706886],
    mass=0.226, inertia=[0.00340344, 0.00339393, 3.54834e-05],
    joint_axis=[0, 1, 0], joint_range=[-2.69653, -0.916298]
)

# Normalize quaternions
for body_id, body in robot.body.items():
    body.quat = ram.quat_normalize(body.quat)
    body.iquat = ram.quat_normalize(body.iquat)

#Example of parameter usage
# print(robot.params.end_eff_pos_local)
# print(robot[1])
