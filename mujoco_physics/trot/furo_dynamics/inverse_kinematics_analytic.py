import numpy as np


def inverse_kinematics_analytic(X_ref):
    """
    Furo 로봇 다리 역기구학 (해석적 해)

    Parameters:
    X_ref: [lx, ly, lz] 다리 프레임에서 발 끝 위치

    Returns:
    q_leg: [hip_roll, hip_pitch, knee] 관절 각도
    """

    L1 = 0.35  # Thigh length
    L2 = 0.35  # Calf length

    lx = X_ref[0]
    ly = X_ref[1]
    lz = X_ref[2]

    # 발 끝까지 총 거리
    l = np.sqrt(lx**2 + ly**2 + lz**2)

    # 거리 제한 (도달 가능한 범위)
    l_max = L1 + L2 - 0.01  # 완전히 펴진 상태보다 조금 짧게
    l_min = 0.05  # 최소 거리
    l = np.clip(l, l_min, l_max)

    # Solution (삼각법 사용)
    # Hip roll: 측면 성분
    ly_ratio = np.clip(ly / l, -1.0, 1.0)  # arcsin 범위 제한
    q_hip_roll = np.arcsin(ly_ratio)

    # Knee: 코사인 법칙
    # cos_knee = (l^2 - L1^2 - L2^2) / (2*L1*L2)
    cos_knee = (l**2 - L1**2 - L2**2) / (2 * L1 * L2)
    cos_knee = np.clip(cos_knee, -1.0, 1.0)  # arccos 범위 제한
    q_knee = -np.pi + np.arccos(cos_knee)

    # Hip pitch: 수직/수평 성분
    lx_ratio = np.clip(-lx / l, -1.0, 1.0)  # arcsin 범위 제한
    q_hip_pitch = -0.5 * q_knee + np.arcsin(lx_ratio)

    q_leg = np.array([q_hip_roll, q_hip_pitch, q_knee])

    return q_leg
