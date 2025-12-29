import numpy as np
from types import SimpleNamespace


def forward_kinematics_leg(q, leg_no):
    """
    Furo 로봇 다리 순기구학

    Parameters:
    q: [hip_roll, hip_pitch, knee] 관절 각도
    leg_no: 0=FR, 1=FL, 2=RR, 3=RL

    Returns:
    sol: SimpleNamespace with end_eff_pos, H01, H02, H03
    """

    L1 = 0.35  # Thigh length (hip to knee)
    L2 = 0.35  # Calf length (knee to foot)

    # Hip lateral offset (from XML: pos="0 ±0.11125 0")
    if leg_no == 1 or leg_no == 3:  # FL or RL (left legs)
        w = 0.11125
    else:  # FR or RR (right legs)
        w = -0.11125

    c1 = np.cos(q[0])
    s1 = np.sin(q[0])
    c2 = np.cos(q[1])
    s2 = np.sin(q[1])
    c3 = np.cos(q[2])
    s3 = np.sin(q[2])

    # Joint positions
    o01 = [0, 0, 0]      # Hip roll joint
    o12 = [0, w, 0]      # Hip pitch joint (lateral offset)
    o23 = [0, 0, -L1]    # Knee joint

    # Homogeneous transformation matrices
    # H01: Hip roll (X-axis rotation)
    H01 = np.array([[1, 0, 0, o01[0]],
                    [0, c1, -s1, o01[1]],
                    [0, s1, c1, o01[2]],
                    [0, 0, 0, 1]])

    # H12: Hip pitch (Y-axis rotation)
    H12 = np.array([[c2, 0, s2, o12[0]],
                    [0, 1, 0, o12[1]],
                    [-s2, 0, c2, o12[2]],
                    [0, 0, 0, 1]])

    # H23: Knee pitch (Y-axis rotation)
    H23 = np.array([[c3, 0, s3, o23[0]],
                    [0, 1, 0, o23[1]],
                    [-s3, 0, c3, o23[2]],
                    [0, 0, 0, 1]])

    # Composite transformations
    H02 = H01 @ H12
    H03 = H02 @ H23

    # End effector position (foot)
    end_eff_pos_local = np.array([0, 0, -L2, 1])
    end_eff_pos = H03 @ end_eff_pos_local
    end_eff_pos = end_eff_pos[0:3]

    sol = SimpleNamespace(
        end_eff_pos=end_eff_pos,
        H01=H01,
        H02=H02,
        H03=H03
    )

    return sol
