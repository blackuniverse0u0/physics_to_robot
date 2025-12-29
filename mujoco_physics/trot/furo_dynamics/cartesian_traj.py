import globals
from quintic_poly import quintic_poly
from parameters import parms
import numpy as np


def cartesian_traj():
    """
    5차 다항식을 사용하여 데카르트 공간에서 발 궤적 생성
    """

    time = globals.time

    t_i = globals.t_i
    t_f = globals.t_f
    lz_i = globals.lz_i
    lz_f = globals.lz_f
    lx_i = globals.lx_i
    lx_f = globals.lx_f
    ly_i = globals.ly_i
    ly_f = globals.ly_f

    t_fsm = globals.t_fsm
    fsm = globals.fsm

    fsm_stance = parms.fsm_stance
    fsm_swing = parms.fsm_swing
    fsm_stand = parms.fsm_stand

    t_step = parms.t_step

    lx_ref, ly_ref, lz_ref, lxdot_ref, lydot_ref, lzdot_ref = (np.zeros(4) for _ in range(6))

    for leg_no in range(4):

        if (fsm[leg_no] == fsm_stand or fsm[leg_no] == fsm_stance):
            # Stand/Stance: 수직 방향만 제어
            lz_ref[leg_no], lzdot_ref[leg_no], _ = quintic_poly(
                time - t_fsm[leg_no], t_i[leg_no], t_f[leg_no], lz_i[leg_no], lz_f[leg_no]
            )

        if (fsm[leg_no] == fsm_swing):
            # Swing: 수평 + 수직 궤적
            lx_ref[leg_no], lxdot_ref[leg_no], _ = quintic_poly(
                time - t_fsm[leg_no], t_i[leg_no], t_f[leg_no], lx_i[leg_no], lx_f[leg_no]
            )
            ly_ref[leg_no], lydot_ref[leg_no], _ = quintic_poly(
                time - t_fsm[leg_no], t_i[leg_no], t_f[leg_no], ly_i[leg_no], ly_f[leg_no]
            )

            # 포물선 궤적 (2단계)
            if (time - t_fsm[leg_no] <= 0.5 * t_step):  # 전반부: 올라가기
                lz_ref[leg_no], lzdot_ref[leg_no], _ = quintic_poly(
                    time - t_fsm[leg_no], t_i[leg_no], t_f[leg_no]/2, lz_i[leg_no], lz_f[leg_no]
                )
            else:  # 후반부: 내려가기
                lz_ref[leg_no], lzdot_ref[leg_no], _ = quintic_poly(
                    time - t_fsm[leg_no], t_f[leg_no]/2, t_f[leg_no], lz_f[leg_no], lz_i[leg_no]
                )

        # 전역 변수에 저장
        globals.lz_ref[leg_no] = lz_ref[leg_no]
        globals.lzdot_ref[leg_no] = lzdot_ref[leg_no]

        globals.lx_ref[leg_no] = lx_ref[leg_no]
        globals.lxdot_ref[leg_no] = lxdot_ref[leg_no]

        globals.ly_ref[leg_no] = ly_ref[leg_no]
        globals.lydot_ref[leg_no] = lydot_ref[leg_no]
