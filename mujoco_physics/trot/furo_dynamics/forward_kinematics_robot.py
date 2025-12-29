import numpy as np
from types import SimpleNamespace
from forward_kinematics_leg import forward_kinematics_leg


def forward_kinematics_robot(q):
    """
    Furo 로봇 전체 순기구학 (간소화 버전)

    Parameters:
    q: [x, y, z, q0, qx, qy, qz, q_leg0, q_leg1, q_leg2, q_leg3] (19차원)

    Returns:
    end_eff_pos: (4, 3) 배열 - 각 다리의 발 끝 위치 (world frame)
    trunk_com_pos: 몸통 질량 중심 위치
    """

    # 몸통 자세
    pos_trunk = q[:3]
    quat_trunk = q[3:7]

    # 쿼터니언 → 회전 행렬
    import utility as ram
    R_trunk = ram.quat2mat(quat_trunk)

    # 각 다리의 관절 각도
    q_legs = q[7:]

    # Furo 로봇 고관절 위치 (base frame)
    # XML에서: FL=(0.3985, 0.07, 0), FR=(0.3985, -0.07, 0)
    #          RL=(-0.3985, 0.07, 0), RR=(-0.3985, -0.07, 0)
    hip_positions_base = np.array([
        [0.3985, -0.07, 0],    # FR (leg 0)
        [0.3985, 0.07, 0],     # FL (leg 1)
        [-0.3985, -0.07, 0],   # RR (leg 2)
        [-0.3985, 0.07, 0]     # RL (leg 3)
    ])

    end_eff_pos = []

    for leg_no in range(4):
        # 이 다리의 관절 각도
        q_leg = q_legs[3*leg_no:3*leg_no+3]

        # 다리 프레임에서 발 끝 위치 계산
        sol_leg = forward_kinematics_leg(q_leg, leg_no)
        foot_pos_leg = sol_leg.end_eff_pos

        # World frame으로 변환
        hip_pos_world = pos_trunk + R_trunk @ hip_positions_base[leg_no]
        foot_pos_world = hip_pos_world + R_trunk @ foot_pos_leg

        end_eff_pos.append(foot_pos_world)

    # 몸통 COM (간단히 base 위치로 가정)
    trunk_com_pos = pos_trunk.copy()

    sol = SimpleNamespace(
        end_eff_pos=np.array(end_eff_pos),
        trunk_com_pos=trunk_com_pos
    )

    return None, sol  # A1 호환성을 위해 robot 객체는 None 반환
