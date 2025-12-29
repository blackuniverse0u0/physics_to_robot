"""
Furo Robot Trot Gait Controller (Simplified)
Based on cosine_moving.py with proper trot gait implementation
"""
import mujoco
import mujoco.viewer
import numpy as np
import time

# XML 파일 경로
MODEL_XML_PATH = "../Furo/scene.xml"

def deg2rad(deg):
    """도를 라디안으로 변환"""
    return deg * (np.pi / 180.0)

# ==========================================
# Trot 걸음걸이 자세 정의 (Keyframe 기반)
# ==========================================
# 관절 순서: FL, FR, RL, RR
# 각 다리: [hip_roll, hip_pitch, knee]
#
# Keyframe 기본 자세:
# FL: [0, 0.6, -1.4]  (앞다리: + hip_pitch, - knee)
# FR: [0, -0.6, 1.4]  (앞다리: - hip_pitch, + knee)
# RL: [0, 0.6, -1.4]  (뒷다리: + hip_pitch, - knee)
# RR: [0, -0.6, 1.4]  (뒷다리: - hip_pitch, + knee)

# Standing pose (keyframe과 동일)
q_stand = np.array([
    0,  0.6, -1.4,   # FL
    0, -0.6,  1.4,   # FR
    0,  0.6, -1.4,   # RL
    0, -0.6,  1.4    # RR
])

# Trot Phase 1: FL + RR swing (앞왼쪽 + 뒤오른쪽 스윙)
# Swing legs: hip_pitch 더 크게 (발 앞으로)
# Stance legs: hip_pitch 작게 (발 뒤로)
q_trot_phase1 = np.array([
    0,  0.8, -1.6,   # FL (swing - 앞으로)
    0, -0.4,  1.2,   # FR (stance - 뒤로)
    0,  0.4, -1.2,   # RL (stance - 뒤로)
    0, -0.8,  1.6    # RR (swing - 앞으로)
])

# Trot Phase 2: FR + RL swing (앞오른쪽 + 뒤왼쪽 스윙)
q_trot_phase2 = np.array([
    0,  0.4, -1.2,   # FL (stance - 뒤로)
    0, -0.8,  1.6,   # FR (swing - 앞으로)
    0,  0.8, -1.6,   # RL (swing - 앞으로)
    0, -0.4,  1.2    # RR (stance - 뒤로)
])


def cosine_interpolate(q_start, q_end, t, period):
    """
    코사인 보간으로 부드러운 궤적 생성

    Args:
        q_start: 시작 자세
        q_end: 끝 자세
        t: 현재 시간
        period: 주기

    Returns:
        보간된 자세
    """
    # alpha: 0 -> 1 (half period), 1 -> 0 (next half period)
    alpha = 0.5 * (1 - np.cos(2 * np.pi * t / period))
    return q_start + (q_end - q_start) * alpha


def main():
    # 모델 로드
    m = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
    d = mujoco.MjData(m)

    print("=" * 60)
    print("Furo Robot Trot Gait Controller (Simplified)")
    print("=" * 60)
    print(f"Model: {m.nq} DOF, {m.nu} actuators")
    print(f"Timestep: {m.opt.timestep}s")
    print(f"Control mode: Position control")
    print()

    # ==========================================
    # 시간 설정
    # ==========================================
    duration = 30.0      # 전체 실행 시간
    wait_time = 2.0      # 초기 대기 시간 (서 있기)
    trot_period = 0.8    # Trot 주기 (0.8초 = 각 단계 0.4초)

    # 초기 자세: 서 있기
    # qpos[7:]에 로봇 관절 설정 (base freejoint 7 DOF 이후)
    d.qpos[7:19] = q_stand

    # Viewer 실행
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start_time = time.time()

        print("Starting simulation...")
        print("Phase 1: Standing (2 seconds)")
        print("Phase 2: Trotting")
        print()

        while viewer.is_running():
            # 현재 경과 시간
            now = time.time() - start_time

            if now > duration:
                print(f"\nSimulation complete! Time: {now:.2f}s")
                break

            # 제어 로직
            if now < wait_time:
                # Phase 1: 서 있기
                target_qpos = q_stand
            else:
                # Phase 2: Trot 걸음걸이
                t_move = now - wait_time

                # 현재 주기 내 위치 (0 ~ trot_period)
                t_in_cycle = t_move % trot_period

                # 주기의 절반마다 phase 전환
                if t_in_cycle < trot_period / 2:
                    # Phase 1 -> Phase 2
                    target_qpos = cosine_interpolate(
                        q_trot_phase1, q_trot_phase2,
                        t_in_cycle, trot_period / 2
                    )
                else:
                    # Phase 2 -> Phase 1
                    target_qpos = cosine_interpolate(
                        q_trot_phase2, q_trot_phase1,
                        t_in_cycle - trot_period / 2, trot_period / 2
                    )

            # 위치 제어 (actuator에 목표 각도 전달)
            d.ctrl[:] = target_qpos

            # 물리 시뮬레이션 스텝
            step_start = time.time()
            mujoco.mj_step(m, d)

            # 화면 업데이트
            viewer.sync()

            # 실시간 동기화
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            # 진행 상황 출력 (1초마다)
            if int(now) != int(now - m.opt.timestep) and now > wait_time:
                pos = d.qpos[:3]
                print(f"t={now-wait_time:.1f}s | pos=({pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f})")


if __name__ == "__main__":
    main()
