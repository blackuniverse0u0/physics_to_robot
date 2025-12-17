import mujoco
import mujoco.viewer
import numpy as np
from scipy.linalg import expm
import time

# ---------------------------------------------------------
# 1. 4족 보행 로봇 다리 모델 정의 (XML string)
# ---------------------------------------------------------
# 구조:
# Base (몸통)
#  -> Joint 1 (Hip Roll, X축): 다리를 옆으로 벌림
#    -> Hip Link
#      -> Joint 2 (Hip Pitch, Y축): 허벅지를 앞으로 듦
#        -> Thigh Link (허벅지)
#          -> Joint 3 (Knee Pitch, Y축): 무릎을 굽힘
#            -> Calf Link (종아리)
#              -> Foot (발끝)
xml = """
<mujoco>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
  </asset>
  
  <worldbody>
    <light pos="0 0 2" mode="trackcom"/>
    <geom name="ground" type="plane" pos="0 0 0" size="2 2 .1" material="grid"/>
    
    <body name="base" pos="0 0 0.6">
      <geom type="box" size="0.1 0.1 0.1" rgba="0.5 0.5 0.5 1"/>
      
      <body name="hip_roll_module" pos="0.1 0 0">
        <joint name="hip_roll" type="hinge" axis="1 0 0" pos="0 0 0"/>
        <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.04" rgba="0.8 0.2 0.2 1"/>
        
        <body name="thigh" pos="0.1 0 0">
          <joint name="hip_pitch" type="hinge" axis="0 1 0" pos="0 0 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.035" rgba="0.2 0.8 0.2 1"/>
          
          <body name="calf" pos="0 0 -0.25">
            <joint name="knee_pitch" type="hinge" axis="0 1 0" pos="0 0 0"/>
            <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.03" rgba="0.2 0.2 0.8 1"/>
            
            <site name="foot" pos="0 0 -0.25" size="0.04" rgba="1 1 0 1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# ---------------------------------------------------------
# 2. Helper 함수들 (PoE용) - 변경 없음
# ---------------------------------------------------------
def vec_to_so3(omega):
    return np.array([
        [0, -omega[2], omega[1]],
        [omega[2], 0, -omega[0]],
        [-omega[1], omega[0], 0]
    ])

def screw_to_se3(S):
    omega = S[:3]
    v = S[3:]
    se3_mat = np.eye(4)
    se3_mat[:3, :3] = vec_to_so3(omega)
    se3_mat[:3, 3] = v
    return se3_mat

# ---------------------------------------------------------
# 3. 초기 정보 (M, S) 자동 추출
# ---------------------------------------------------------
# 초기화
data.qpos = np.zeros(model.nq)
mujoco.mj_forward(model, data) 

# (1) M: 초기 발끝의 위치/자세
foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "foot")
M = np.eye(4)
M[:3, 3] = data.site_xpos[foot_id]
M[:3, :3] = data.site_xmat[foot_id].reshape(3, 3)

print(f"Initial M (Position): {M[:3, 3]}")

# (2) S: 각 관절의 Screw Axis 추출
S_list = []
n_joints = model.njnt
print(f"Number of joints: {n_joints} (Roll -> Pitch -> Pitch)\n")

for i in range(n_joints):
    w = data.xaxis[i]     # Global 회전축
    q = data.xanchor[i]   # Global 관절 위치
    v = -np.cross(w, q)   # v = -w x q
    S = np.concatenate([w, v])
    S_list.append(S)
    
    # 디버깅용 출력
    axis_name = "X-axis" if np.argmax(np.abs(w)) == 0 else "Y-axis"
    print(f"Joint {i+1} ({axis_name}) Screw S: {S}")

# ---------------------------------------------------------
# 4. FK 계산 및 검증
# ---------------------------------------------------------
# 목표 각도 설정 (현실적인 다리 움직임)
# 1. Hip Roll: 0.3 rad (다리를 약간 바깥으로 벌림)
# 2. Hip Pitch: 0.8 rad (허벅지를 앞으로 듦)
# 3. Knee Pitch: -1.2 rad (무릎을 뒤로 접음)
target_thetas = np.array([0.45, 0.8, -1.5])

# A. MuJoCo 정답
data.qpos = target_thetas
mujoco.mj_forward(model, data)
mujoco_pos = data.site_xpos[foot_id].copy()

print("\n--- [MuJoCo Result] ---")
print(f"Foot Position: {mujoco_pos}")

# B. PoE 직접 계산
T_curr = np.eye(4)
for i in range(n_joints):
    S = S_list[i]
    theta = target_thetas[i]
    bracket_S = screw_to_se3(S)
    exp_S_theta = expm(bracket_S * theta)
    T_curr = T_curr @ exp_S_theta

T_final = T_curr @ M
poe_pos = T_final[:3, 3]

print("\n--- [PoE Calculation Result] ---")
print(f"Foot Position: {poe_pos}")

# C. 오차 확인
error = np.linalg.norm(mujoco_pos - poe_pos)
print(f"\nDifference (Error): {error:.6f}")

if error < 1e-5:
    print(">> Perfect Match!")
else:
    print(">> Check Code.")

# ---------------------------------------------------------
# 5. 시각화 (Animation)
# ---------------------------------------------------------
start_qpos = np.zeros(model.nq)
end_qpos = target_thetas
animation_steps = 400 

print(">> Starting Visualization...")
with mujoco.viewer.launch_passive(model, data) as viewer:
    # 0도 자세 보여주기
    data.qpos = start_qpos
    mujoco.mj_forward(model, data)
    viewer.sync()
    time.sleep(1.0)

    # 애니메이션
    for i in range(animation_steps):
        alpha = i / (animation_steps - 1)
        current_qpos = (1 - alpha) * start_qpos + alpha * end_qpos
        
        data.qpos = current_qpos
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(0.005)
        
    # 유지
    while viewer.is_running():
        time.sleep(0.1)