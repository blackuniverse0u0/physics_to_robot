"""Furo 초기화 테스트 (GUI 없이)"""
import mujoco as mj
import numpy as np
import os

import utility as ram
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

print("Loading MuJoCo model...")
model = mj.MjModel.from_xml_path(abspath)
data = mj.MjData(model)

print(f"Model loaded successfully!")
print(f"  nq (DOF): {model.nq}")
print(f"  nv (velocity DOF): {model.nv}")
print(f"  nu (actuators): {model.nu}")
print(f"  nkey (keyframes): {model.nkey}")

# 전역 변수 초기화
print("\nInitializing globals...")
globals.init()

# Keyframe 사용
if model.nkey > 0:
    mj.mj_resetDataKeyframe(model, data, 0)
    print(f"Loaded keyframe 'home'")
else:
    print("Warning: No keyframe found")
    data.qpos[:] = 0
    data.qpos[2] = 0.55
    data.qpos[3] = 1.0

print(f"\nInitial qpos (first 20 values): {data.qpos[:20]}")

# 로봇 관절 추출
hip_roll = 0.0
hip_pitch = data.qpos[8]
knee = data.qpos[9]

print(f"\nRobot joint angles:")
print(f"  hip_roll: {hip_roll:.3f}")
print(f"  hip_pitch: {hip_pitch:.3f}")
print(f"  knee: {knee:.3f}")

# 순기구학 테스트
sol = forward_kinematics_leg(np.array([hip_roll, hip_pitch, knee]), 0)
end_eff_pos = sol.end_eff_pos
print(f"\nForward kinematics test:")
print(f"  Foot position: {end_eff_pos}")
print(f"  Foot height (lz0): {end_eff_pos[2]:.4f} m")

# 제어 루프 1스텝 테스트
print("\nTesting control loop (1 step)...")
try:
    state_machine()
    print("  ✓ state_machine()")

    cartesian_traj()
    print("  ✓ cartesian_traj()")

    joint_traj()
    print("  ✓ joint_traj()")

    globals.time = data.time
    globals.q_act = data.qpos[7:19].copy()
    globals.u_act = data.qvel[6:18].copy()
    globals.pos_quat_trunk = data.qpos[:7].copy()
    globals.vel_angvel_trunk = data.qvel[:6].copy()

    joint_control()
    print("  ✓ joint_control()")

    high_level_control()
    print("  ✓ high_level_control()")

    data.ctrl = globals.trq.copy()
    print(f"  Control torques: {data.ctrl}")

    mj.mj_step(model, data)
    print("  ✓ mj_step()")

    print("\n✅ All control modules working correctly!")

except Exception as e:
    print(f"\n❌ Error in control loop: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Test Complete ===")
