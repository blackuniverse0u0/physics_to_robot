"""Furo 시뮬레이션 테스트 (GUI 없이, 10스텝)"""
import mujoco as mj
import numpy as np
import os

import globals
from state_machine import state_machine
from forward_kinematics_leg import forward_kinematics_leg
from cartesian_traj import cartesian_traj
from joint_traj import joint_traj
from joint_control import joint_control
from high_level_control import high_level_control

# XML 로드
xml_path = '../Furo/scene.xml'
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)

model = mj.MjModel.from_xml_path(abspath)
data = mj.MjData(model)

# 초기화
globals.init()
mj.mj_resetDataKeyframe(model, data, 0)

print("=" * 60)
print("Furo Trot Gait Simulation Test")
print("=" * 60)
print(f"Model: {model.nq} DOF, {model.nu} actuators")
print(f"Initial position: [{data.qpos[0]:.3f}, {data.qpos[1]:.3f}, {data.qpos[2]:.3f}]")
print(f"Simulation timestep: {model.opt.timestep}s")
print()

# 시뮬레이션 실행
num_steps = 1000
render_every = 100

for step in range(num_steps):
    # 제어 루프
    state_machine()
    cartesian_traj()
    joint_traj()

    globals.time = data.time
    globals.q_act = data.qpos[7:19].copy()
    globals.u_act = data.qvel[6:18].copy()
    globals.pos_quat_trunk = data.qpos[:7].copy()
    globals.vel_angvel_trunk = data.qvel[:6].copy()

    joint_control()
    high_level_control()

    data.ctrl = globals.trq.copy()
    mj.mj_step(model, data)

    # 진행 상황 출력
    if step % render_every == 0:
        pos = data.qpos[:3]
        quat = data.qpos[3:7]
        yaw = np.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]),
                        1 - 2*(quat[2]**2 + quat[3]**2))

        print(f"Step {step:4d} | t={data.time:.3f}s | "
              f"pos=({pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}) | "
              f"yaw={yaw:6.3f} | "
              f"FSM={globals.fsm}")

print()
print("=" * 60)
print("Simulation Complete!")
print("=" * 60)
print(f"Final time: {data.time:.3f}s")
print(f"Final position: x={data.qpos[0]:.3f}m, y={data.qpos[1]:.3f}m, z={data.qpos[2]:.3f}m")

# 최종 yaw 계산
quat = data.qpos[3:7]
final_yaw = np.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]),
                       1 - 2*(quat[2]**2 + quat[3]**2))
print(f"Final yaw: {final_yaw:.3f} rad ({np.degrees(final_yaw):.1f}°)")

# 이동 거리 계산
distance = np.sqrt(data.qpos[0]**2 + data.qpos[1]**2)
print(f"Distance traveled: {distance:.3f}m")
print(f"Average speed: {distance/data.time:.3f} m/s")
print()
print("✅ Simulation successful!")
