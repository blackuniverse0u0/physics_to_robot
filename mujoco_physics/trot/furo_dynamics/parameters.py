

class parameters:
    def __init__(self):

        self.fsm_stand = 1
        self.fsm_stance = 2
        self.fsm_swing = 3
        self.t_stand = 0.1
        self.t_step = 0.2  # Furo는 A1보다 크고 무거워서 스텝 시간을 더 길게

        # Furo 로봇 기하학적 파라미터
        # Keyframe 기준 발 높이 (hip_pitch=0.6, knee=-1.4)
        self.lz0 = -0.533  # 초기 발 위치 (다리 프레임)
        self.hcl = 0.10    # 스윙 시 발 들어올리는 높이

        # Furo 로봇 물리적 특성
        self.mass = 15.702  # XML에서 base inertial mass
        self.gravity = 9.81

        # 속도 제한
        self.vx_min = -1.5
        self.vx_max = 1.5
        self.dvx = 0.08  # 더 무거워서 가속도 제한
        self.vy_min = -0.8
        self.vy_max = 0.8
        self.dvy = 0.04
        self.omega_min = -1.5
        self.omega_max = 1.5
        self.domega = 0.08


parms = parameters()
