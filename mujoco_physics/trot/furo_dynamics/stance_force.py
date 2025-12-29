import globals
import numpy as np
from parameters import parms
from forward_kinematics_robot import forward_kinematics_robot
import utility as ram


def stance_force(leg_no):
    """
    지지 다리가 몸통을 안정화시키기 위한 가상 지면 반력 계산

    Parameters:
    leg_no: 0 or 1 (어느 대각선 쌍이 지지 중인지)

    Returns:
    F0, F1, F2, F3: 각 다리의 지면 반력 (3차원 벡터)
    """

    q_act = globals.q_act.copy()
    pos_quat_trunk = globals.pos_quat_trunk.copy()
    vel_angvel_trunk = globals.vel_angvel_trunk.copy()

    quat = pos_quat_trunk[3:]
    euler = ram.quat2bryant(quat)

    # Yaw 각도만 제거한 body frame 사용
    Rz = ram.rotation(euler[2], 2)
    R = ram.quat2mat(quat)
    R_body = Rz.T @ R
    quat_body = ram.mat2quat(R_body)
    pos_quat_trunk_ = np.concatenate((pos_quat_trunk[:3], quat_body))
    vel = vel_angvel_trunk[:3]
    vel_body = Rz.T @ vel

    q = np.concatenate((pos_quat_trunk_, q_act))

    # 전체 로봇 순기구학
    _, sol = forward_kinematics_robot(q)
    end_eff_pos = sol.end_eff_pos
    trunk_com_pos = sol.trunk_com_pos

    # A 행렬 설정 (지지 다리에 따라)
    I = np.identity(3)

    if (leg_no == 0):  # FR + RL 쌍
        r0 = end_eff_pos[0, :] - trunk_com_pos
        r3 = end_eff_pos[3, :] - trunk_com_pos
        R0 = ram.vec2skew(r0)
        R3 = ram.vec2skew(r3)
        A = np.block([
            [I, I],      # Top row
            [R0, R3]     # Bottom row
        ])

    if (leg_no == 1):  # FL + RR 쌍
        r1 = end_eff_pos[1, :] - trunk_com_pos
        r2 = end_eff_pos[2, :] - trunk_com_pos
        R1 = ram.vec2skew(r1)
        R2 = ram.vec2skew(r2)
        A = np.block([
            [I, I],      # Top row
            [R1, R2]     # Bottom row
        ])

    # 목표 wrench (b 벡터) 계산
    z = pos_quat_trunk[2]
    z_ref = -parms.lz0  # 목표 높이
    zdot = vel_angvel_trunk[2]
    omega = vel_angvel_trunk[3:]
    xdot = vel_body[0]
    ydot = vel_body[1]
    psidot = vel_angvel_trunk[5]

    # 제어 게인 (Furo는 더 무거워서 게인 조정)
    fx0 = 80 * (globals.xdot_ref - xdot)
    fy0 = 80 * (globals.ydot_ref - ydot)
    fz0 = 40 * (-10*(z - z_ref) - 1*zdot) + parms.mass * parms.gravity
    Mx0 = 40 * (-10*euler[0] - 0.5*omega[0])
    My0 = 40 * (-10*euler[1] - 0.5*omega[1])
    Mz0 = 8 * (globals.psidot_ref - psidot)

    b = np.array([fx0, fy0, fz0, Mx0, My0, Mz0])

    # F = inv(A)*b
    Ainv = np.linalg.pinv(A, rcond=1e-10, hermitian=False)
    F = Ainv.dot(b)

    # 각 다리의 힘 추출
    if (leg_no == 0):  # FR + RL
        F0 = np.array([F[0], F[1], F[2]])
        F3 = np.array([F[3], F[4], F[5]])
        F1 = np.zeros(3)
        F2 = np.zeros(3)

    if (leg_no == 1):  # FL + RR
        F1 = np.array([F[0], F[1], F[2]])
        F2 = np.array([F[3], F[4], F[5]])
        F0 = np.zeros(3)
        F3 = np.zeros(3)

    return F0, F1, F2, F3
