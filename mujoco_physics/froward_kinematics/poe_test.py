# --- PoE (Product of Exponentials) 계산 ---

# skew symmetric matrix : 선형 대수학(Linear Algebra)의 도구들을 회전 운동에 적용하기 위해
# a x b = [a]b 

# M : Home configuration 

import mujoco
import mujoco.viewer
import numpy as np
import time

# ---------------------------------------------------------
# [이론 적용]
# 1. SO(3)는 비선형이므로 직접 더할 수 없습니다.
# 2. 대신 Lie Algebra so(3) (Skew-symmetric matrix) 공간에서
#    선형 연산(행렬 곱)을 수행한 뒤, Exponential Map을 통해 SO(3)로 보냅니다.
# ---------------------------------------------------------

def skew(v):
    """
    [이론] 3차원 벡터 v를 so(3) 접공간의 원소(반대칭 행렬)로 변환
    이 행렬은 벡터 외적(Cross Product)을 선형 변환(행렬 곱)으로 대체합니다.
    a x b = [a] @ b
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def get_screw_exponential(S, theta):
    """
    [이론] Exponential Map: so(3) -> SO(3)
    접공간(Lie Algebra)에서 정의된 Screw S와 각도 theta를 
    비선형 공간(Lie Group)인 변환 행렬 T로 매핑합니다.
    """
    omega = S[:3] # 회전 성분
    v = S[3:]     # 속도 성분
    
    # [omega] : so(3) element (Skew-symmetric matrix)
    omega_mat = skew(omega)
    
    # Rodrigues' Formula (Linear Algebra 연산만 사용)
    # R = I + sin(th)[w] + (1-cos(th))[w]^2
    R = np.eye(3) + np.sin(theta) * omega_mat + (1 - np.cos(theta)) * (omega_mat @ omega_mat)
    
    # G matrix (Translation mapping)
    # p = (I*th + (1-cos(th))[w] + (th-sin(th))[w]^2) * v
    G = (np.eye(3) * theta) + \
        (1 - np.cos(theta)) * omega_mat + \
        (theta - np.sin(theta)) * (omega_mat @ omega_mat)
    
    p = G @ v # 행렬-벡터 곱 (선형 연산)
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T

# ---------------------------------------------------------
# 모델 설정 (이전과 동일)
# ---------------------------------------------------------
xml = """
<mujoco>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
  </asset>
  <worldbody>
    <light pos="0 0 2" mode="trackcom"/>
    <geom name="ground" type="plane" pos="0 0 -0.1" size="2 2 .1" material="grid"/>
    <body name="link1" pos="0 0 0">
      <joint name="joint1" type="hinge" axis="0 0 1" pos="0 0 0"/>
      <geom type="capsule" fromto="0 0 0 0.5 0 0" size="0.05" rgba="0.8 0.2 0.2 1"/>
      <body name="link2" pos="0.5 0 0">
        <joint name="joint2" type="hinge" axis="0 0 1" pos="0 0 0"/>
        <geom type="capsule" fromto="0 0 0 0.5 0 0" size="0.05" rgba="0.2 0.8 0.2 1"/>
        <site name="end_effector" pos="0.5 0 0" size="0.06" rgba="0 0 1 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

target_thetas = np.array([np.deg2rad(30), np.deg2rad(45)])
L1, L2 = 0.5, 0.5

# ---------------------------------------------------------
# PoE Calculation (Pure Linear Algebra Implementation)
# ---------------------------------------------------------

# 1. Home Configuration M
M = np.eye(4)
M[:3, 3] = np.array([L1 + L2, 0, 0])

# 2. Screw Axes Definition (Using Linear Operators)
# Joint 1
w1 = np.array([0, 0, 1])
q1 = np.array([0, 0, 0])
# [수정] np.cross 대신 행렬 곱 사용: v = -[w] @ q
# 이것이 "선형화"된 표현입니다. 외적이라는 기하학적 연산을 행렬 연산으로 바꿈.
v1 = -skew(w1) @ q1 
S1 = np.concatenate((w1, v1))

# Joint 2
w2 = np.array([0, 0, 1])
q2 = np.array([L1, 0, 0]) # Home config 기준 위치
# [수정] np.cross 대신 행렬 곱 사용
v2 = -skew(w2) @ q2 
S2 = np.concatenate((w2, v2))

# 3. Forward Kinematics
T_S1 = get_screw_exponential(S1, target_thetas[0])
T_S2 = get_screw_exponential(S2, target_thetas[1])

# Matrix Chain Multiplication
T_PoE = T_S1 @ T_S2 @ M

# 결과 비교
data.qpos = target_thetas
mujoco.mj_forward(model, data)
mujoco_pos = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")]
poe_pos = T_PoE[:3, 3]

print(f"MuJoCo Pos: {mujoco_pos}")
print(f"PoE    Pos: {poe_pos}")
print(f"Error     : {np.linalg.norm(mujoco_pos - poe_pos):.8f}")

# 시각화
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.sync()
    time.sleep(2)