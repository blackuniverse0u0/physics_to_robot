import numpy as np
import utility as ram
from forward_kinematics_leg import forward_kinematics_leg


def jac_end_effector_leg(q, leg_no):
    """
    Furo 로봇 다리 야코비안 계산

    Parameters:
    q: [hip_roll, hip_pitch, knee] 관절 각도
    leg_no: 다리 번호

    Returns:
    Jv_E: 3x3 야코비안 행렬
    """
    sol = forward_kinematics_leg(q, leg_no)

    # Get the output
    end_eff_pos = sol.end_eff_pos
    H01 = sol.H01
    H02 = sol.H02
    H03 = sol.H03

    # End-effector position
    e0 = end_eff_pos

    # Frame origins
    o01 = H01[0:3, 3]
    o02 = H02[0:3, 3]
    o03 = H03[0:3, 3]

    # Joint axes
    n1 = np.array([1, 0, 0])  # Hip roll: X-axis
    n2 = np.array([0, 1, 0])  # Hip pitch: Y-axis
    n3 = np.array([0, 1, 0])  # Knee: Y-axis

    # Rotation matrices
    R00 = np.eye(3)
    R01 = H01[0:3, 0:3]
    R02 = H02[0:3, 0:3]
    R03 = H03[0:3, 0:3]

    # Jacobian columns: J_i = n_i × (end_eff - origin_i)
    Jv_E = np.column_stack([
        ram.vec2skew(R00 @ n1) @ (e0 - o01),
        ram.vec2skew(R01 @ n2) @ (e0 - o02),
        ram.vec2skew(R02 @ n3) @ (e0 - o03),
    ])

    return Jv_E
