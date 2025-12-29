"""
Furo Trot 제어기 테스트 (GUI 없음)
"""
import mujoco
import numpy as np

MODEL_XML_PATH = "../Furo/scene.xml"

def deg2rad(deg):
    return deg * (np.pi / 180.0)

# 자세 정의 (Keyframe 기반)
q_stand = np.array([
    0,  0.6, -1.4,   # FL
    0, -0.6,  1.4,   # FR
    0,  0.6, -1.4,   # RL
    0, -0.6,  1.4    # RR
])

q_walk_phase1 = np.array([
    0,  0.8, -1.6,   # FL (swing - 앞으로)
    0, -0.4,  1.2,   # FR (stance - 뒤로)
    0,  0.4, -1.2,   # RL (stance - 뒤로)
    0, -0.8,  1.6    # RR (swing - 앞으로)
])

q_walk_phase2 = np.array([
    0,  0.4, -1.2,   # FL (stance - 뒤로)
    0, -0.8,  1.6,   # FR (swing - 앞으로)
    0,  0.8, -1.6,   # RL (swing - 앞으로)
    0, -0.4,  1.2    # RR (stance - 뒤로)
])

def cosine_interpolate(q_start, q_end, t, period):
    alpha = 0.5 * (1 - np.cos(2 * np.pi * t / period))
    return q_start + (q_end - q_start) * alpha

# 모델 로드
m = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
d = mujoco.MjData(m)

print("=" * 60)
print("Furo Trot Gait Test (No GUI)")
print("=" * 60)
print(f"Model: {m.nq} DOF, {m.nu} actuators")
print(f"Timestep: {m.opt.timestep}s")
print()

# 초기 자세
d.qpos[7:19] = q_stand

# 시뮬레이션 파라미터
wait_time = 1.0      # 1초 대기
trot_period = 0.8    # Trot 주기
num_steps = 5000     # 시뮬레이션 스텝 수 (10초)

print("Starting simulation...")
print(f"Phase 1: Standing ({wait_time}s)")
print(f"Phase 2: Trotting (period={trot_period}s)")
print()

for step in range(num_steps):
    t = step * m.opt.timestep

    if t < wait_time:
        # Standing
        target_qpos = q_stand
    else:
        # Trotting
        t_move = t - wait_time
        t_in_cycle = t_move % trot_period

        if t_in_cycle < trot_period / 2:
            target_qpos = cosine_interpolate(
                q_walk_phase1, q_walk_phase2,
                t_in_cycle, trot_period / 2
            )
        else:
            target_qpos = cosine_interpolate(
                q_walk_phase2, q_walk_phase1,
                t_in_cycle - trot_period / 2, trot_period / 2
            )

    # 제어
    d.ctrl[:] = target_qpos

    # 시뮬레이션 스텝
    mujoco.mj_step(m, d)

    # 진행 상황 출력 (0.5초마다)
    if step % 250 == 0:
        pos = d.qpos[:3]
        quat = d.qpos[3:7]
        yaw = np.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]),
                        1 - 2*(quat[2]**2 + quat[3]**2))

        phase = "Stand" if t < wait_time else f"Trot{int((t-wait_time)/(trot_period/2))%2+1}"
        print(f"t={t:5.2f}s | {phase:6s} | "
              f"pos=({pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}) | "
              f"yaw={np.degrees(yaw):6.1f}°")

print()
print("=" * 60)
print("Simulation Complete!")
print("=" * 60)

# 최종 결과
pos = d.qpos[:3]
distance = np.sqrt(pos[0]**2 + pos[1]**2)
print(f"Final position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
print(f"Distance traveled: {distance:.3f} m")
print(f"Average speed: {distance/(num_steps*m.opt.timestep):.3f} m/s")
print(f"Final height: {pos[2]:.3f} m")
print()

# 높이 유지 확인
if pos[2] > 0.4:
    print("✅ Robot maintained height successfully!")
else:
    print(f"⚠️  Robot lost height (final z={pos[2]:.3f}m < 0.4m)")

print()
