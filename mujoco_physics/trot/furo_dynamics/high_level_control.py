import globals
import numpy as np
from parameters import parms
from set_command_step import set_command_step


def high_level_control():
    """
    상위 레벨 속도 명령 생성 및 업데이트
    """
    if (globals.prev_step < globals.step):
        globals.prev_step = globals.step

        # 전진 속도
        vx = globals.xdot_ref
        vx_ = 0.5  # 목표 속도 (m/s) - Furo는 더 무거워서 천천히
        globals.xdot_ref = set_command_step(vx_, vx, parms.vx_min, parms.vx_max, parms.dvx)

        # 측면 속도 (필요시 활성화)
        # vy = globals.ydot_ref
        # vy_ = 0.0
        # globals.ydot_ref = set_command_step(vy_, vy, parms.vy_min, parms.vy_max, parms.dvy)

        # 요 각속도
        omega = globals.psidot_ref
        omega_ = 0.5  # 목표 각속도 (rad/s)
        globals.psidot_ref = set_command_step(omega_, omega, parms.omega_min, parms.omega_max, parms.domega)
