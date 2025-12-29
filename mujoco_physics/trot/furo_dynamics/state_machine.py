import globals
from parameters import parms


def state_machine():
    """
    Furo 로봇 Trot 걸음걸이를 위한 유한 상태 기계
    """

    time = globals.time

    fsm_stand = parms.fsm_stand
    fsm_stance = parms.fsm_stance
    fsm_swing = parms.fsm_swing

    t_stand = parms.t_stand
    t_step = parms.t_step

    # Furo의 몸통 길이 (XML에서 front hip x=0.3985, rear hip x=-0.3985)
    c = 0.3985  # 앞/뒤 고관절 거리의 절반

    for leg_no in range(4):
        # Stand → Swing/Stance (초기 시작)
        if (time >= globals.t_fsm[leg_no] + t_stand and globals.fsm[leg_no] == fsm_stand):

            if (leg_no == 0 or leg_no == 3):  # FR + RL (대각선 쌍)
                globals.fsm[leg_no] = fsm_swing
                globals.t_fsm[leg_no] = time
                globals.t_i[leg_no] = 0
                globals.t_f[leg_no] = t_step
                globals.lz_i[leg_no] = parms.lz0
                globals.lz_f[leg_no] = parms.lz0 + parms.hcl

            if (leg_no == 1 or leg_no == 2):  # FL + RR (반대 쌍)
                globals.fsm[leg_no] = fsm_stance
                globals.t_fsm[leg_no] = time
                globals.t_i[leg_no] = 0
                globals.t_f[leg_no] = t_step
                globals.lz_i[leg_no] = parms.lz0
                globals.lz_f[leg_no] = parms.lz0

        # Stance → Swing
        if (time >= globals.t_fsm[leg_no] + t_step and globals.fsm[leg_no] == fsm_stance):
            globals.fsm[leg_no] = fsm_swing
            globals.t_fsm[leg_no] = time
            globals.t_i[leg_no] = 0
            globals.t_f[leg_no] = t_step
            globals.lz_i[leg_no] = parms.lz0
            globals.lz_f[leg_no] = parms.lz0 + parms.hcl

            # Raibert heuristic: 발 착지 위치
            globals.lx_i[leg_no] = -0.5 * globals.xdot_ref * t_step
            globals.lx_f[leg_no] = 0.5 * globals.xdot_ref * t_step
            globals.ly_i[leg_no] = -0.5 * globals.ydot_ref * t_step
            globals.ly_f[leg_no] = 0.5 * globals.ydot_ref * t_step

            # 요(yaw) 회전을 위한 보정
            if (leg_no == 0 or leg_no == 1):  # Front legs
                globals.ly_i[leg_no] -= 0.5 * c * globals.psidot_ref * t_step
                globals.ly_f[leg_no] += 0.5 * c * globals.psidot_ref * t_step
            else:  # Rear legs
                globals.ly_i[leg_no] += 0.5 * c * globals.psidot_ref * t_step
                globals.ly_f[leg_no] -= 0.5 * c * globals.psidot_ref * t_step

        # Swing → Stance
        if (time >= globals.t_fsm[leg_no] + t_step and globals.fsm[leg_no] == fsm_swing):
            if (leg_no == 0 or leg_no == 1):  # Front legs
                globals.step += 1

            globals.fsm[leg_no] = fsm_stance
            globals.t_fsm[leg_no] = time
            globals.t_i[leg_no] = 0
            globals.t_f[leg_no] = t_step
            globals.lz_i[leg_no] = parms.lz0
            globals.lz_f[leg_no] = parms.lz0

            # 뒤로 쓸어내리는 궤적
            globals.lx_i[leg_no] = 0.5 * globals.xdot_ref * t_step
            globals.lx_f[leg_no] = -0.5 * globals.xdot_ref * t_step
            globals.ly_i[leg_no] = 0.5 * globals.ydot_ref * t_step
            globals.ly_f[leg_no] = -0.5 * globals.ydot_ref * t_step

            if (leg_no == 0 or leg_no == 1):  # Front legs
                globals.ly_i[leg_no] += 0.5 * c * globals.psidot_ref * t_step
                globals.ly_f[leg_no] -= 0.5 * c * globals.psidot_ref * t_step
            else:  # Rear legs
                globals.ly_i[leg_no] -= 0.5 * c * globals.psidot_ref * t_step
                globals.ly_f[leg_no] += 0.5 * c * globals.psidot_ref * t_step
