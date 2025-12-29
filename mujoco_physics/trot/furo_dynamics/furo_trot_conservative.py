"""
Furo Robot Trot Gait Controller (Conservative - 작은 움직임)
매우 작은 관절 변화로 안정적인 trot 구현
"""
import mujoco
import mujoco.viewer
import numpy as np
import time

MODEL_XML_PATH = "../Furo/scene.xml"

def deg2rad(deg):
    return deg * (np.pi / 180.0)

# Keyframe 기본 자세
q_stand = np.array([
    0,  0.6, -1.4,   # FL
    0, -0.6,  1.4,   # FR
    0,  0.6, -1.4,   # RL
    0, -0.6,  1.4    # RR
])

# Trot Phase 1: FL + RR swing
# 변화를 최소화 (±0.1 rad = ±5.7도만 변화)
q_trot_phase1 = np.array([
    0,  0.7, -1.5,   # FL (swing - 약간 앞으로)
    0, -0.5,  1.3,   # FR (stance - 약간 뒤로)
    0,  0.5, -1.3,   # RL (stance - 약간 뒤로)
    0, -0.7,  1.5    # RR (swing - 약간 앞으로)
])

# Trot Phase 2: FR + RL swing
q_trot_phase2 = np.array([
    0,  0.5, -1.3,   # FL (stance - 약간 뒤로)
    0, -0.7,  1.5,   # FR (swing - 약간 앞으로)
    0,  0.7, -1.5,   # RL (swing - 약간 앞으로)
    0, -0.5,  1.3    # RR (stance - 약간 뒤로)
])

def cosine_interpolate(q_start, q_end, t, period):
    alpha = 0.5 * (1 - np.cos(2 * np.pi * t / period))
    return q_start + (q_end - q_start) * alpha

def main():
    m = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
    d = mujoco.MjData(m)

    print("=" * 60)
    print("Furo Robot Trot Gait (Conservative)")
    print("=" * 60)
    print("작은 관절 변화로 안정적인 trot 구현")
    print(f"관절 변화: ±0.1 rad (±5.7°)")
    print()

    duration = 20.0
    wait_time = 2.0
    trot_period = 1.0  # 더 느린 주기

    # 초기 자세
    d.qpos[7:19] = q_stand

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start_time = time.time()

        print(f"Phase 1: Standing ({wait_time}s)")
        print(f"Phase 2: Trotting (period={trot_period}s, ±0.1 rad)")
        print()

        while viewer.is_running():
            now = time.time() - start_time

            if now > duration:
                break

            if now < wait_time:
                target_qpos = q_stand
            else:
                t_move = now - wait_time
                t_in_cycle = t_move % trot_period

                if t_in_cycle < trot_period / 2:
                    target_qpos = cosine_interpolate(
                        q_trot_phase1, q_trot_phase2,
                        t_in_cycle, trot_period / 2
                    )
                else:
                    target_qpos = cosine_interpolate(
                        q_trot_phase2, q_trot_phase1,
                        t_in_cycle - trot_period / 2, trot_period / 2
                    )

            d.ctrl[:] = target_qpos

            step_start = time.time()
            mujoco.mj_step(m, d)
            viewer.sync()

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            # 1초마다 출력
            if int(now) != int(now - m.opt.timestep) and now > wait_time:
                pos = d.qpos[:3]
                print(f"t={now-wait_time:4.1f}s | pos=({pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f})")

if __name__ == "__main__":
    main()
