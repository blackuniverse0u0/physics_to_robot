import numpy as np
from types import SimpleNamespace
from robot_data import robot
import utility as ram

def forward_kinematics_robot(q):

    end_eff_pos_local = robot.params.end_eff_pos_local

    # Initialize variables for leg joint angles and robot body
    q_trunk = q[:7].copy()
    q_legs = q[7:].copy()
    j = 0  # Counter for the legs


    # Step 1: Get the local homogeneous transformation matrices for each joint
    for i in range(1,len(robot.body)+1):
        if i == 1:  # Skip trunk (body 1) as it's handled separately
            continue

        quat = robot.body[i].quat
        joint_axis = robot.body[i].joint_axis
        axis_id = np.argmax(np.abs(joint_axis))
        angle = q_legs[j]  # Joint angle for this specific leg part
        j += 1

        # Compute rotation based on joint angle
        R_q = ram.rotation(angle, axis_id)
        robot.body[i].R_local = ram.quat2rotation(quat) @ R_q
        robot.body[i].o_local = robot.body[i].pos

        # Compute the local homogeneous transformation matrix
        robot.body[i].H_local = np.block([
            [robot.body[i].R_local, robot.body[i].o_local.reshape(-1, 1)],
            [np.zeros((1, 3)), 1]
        ])

    # Step 2: Compute global homogeneous transformation matrices
    # Base transformation (trunk)
    pos_trunk = q_trunk[:3]
    quat_trunk = q_trunk[3:]
    R_trunk = ram.quat2rotation(quat_trunk)

    robot.body[1].H_global = np.block([
        [R_trunk, pos_trunk.reshape(-1, 1)],
        [np.zeros((1, 3)), 1]
    ])

    # Apply transformations for each leg
    i = 2  # Start from the first leg joint
    for _ in range(4):  # Loop over the four legs
        temp = robot.body[1].H_global  # Start with the base (trunk)
        for _ in range(3):  # Loop over each joint in the leg
            robot.body[i].H_global = temp @ robot.body[i].H_local
            temp = robot.body[i].H_global
            i += 1

    # Step 3: Calculates various positions for each leg
    in_shoulder_pos = []
    for i in range(1, 5):  # For each leg
        in_shoulder_pos_temp = robot.body[1 + i * 3 - 2].H_global @ np.array([0,0,0,1])  # Homogeneous coordinates
        in_shoulder_pos.append(in_shoulder_pos_temp[:3])  # Extract only position


    out_shoulder_pos = []
    for i in range(1, 5):  # For each leg
        out_shoulder_pos_temp = robot.body[1 + i * 3 - 1].H_global @ np.array([0,0,0,1])  # Homogeneous coordinates
        out_shoulder_pos.append(out_shoulder_pos_temp[:3])  # Extract only position

    elbow_pos = []
    for i in range(1, 5):  # For each leg
        elbow_pos_temp = robot.body[1 + i * 3].H_global @ np.array([0,0,0,1])  # Homogeneous coordinates
        elbow_pos.append(elbow_pos_temp[:3])  # Extract only position

    end_eff_pos = []
    for i in range(1, 5):  # For each leg
        end_eff_pos_temp = robot.body[1 + i * 3].H_global @ np.append(end_eff_pos_local, 1)  # Homogeneous coordinates
        end_eff_pos.append(end_eff_pos_temp[:3])  # Extract only position

    trunk_com_pos_local = np.array(robot.body[1].ipos)
    trunk_com_pos_temp = robot.body[1].H_global @ np.append(trunk_com_pos_local, 1)
    trunk_com_pos = trunk_com_pos_temp[:3].copy()

    # Store in a SimpleNamespace
    sol = SimpleNamespace(
        in_shoulder_pos=np.array(in_shoulder_pos),
        out_shoulder_pos=np.array(out_shoulder_pos),
        elbow_pos=np.array(elbow_pos),
        end_eff_pos=np.array(end_eff_pos),
        trunk_com_pos = np.array(trunk_com_pos)
    )

    return robot, sol
