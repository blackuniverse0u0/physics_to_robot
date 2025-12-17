import mujoco
import mujoco.viewer
import numpy as np
import time

# ---------------------------------------------------------
# 1. 2-Link 로봇 모델 (이전과 동일)
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

"""MuJoCo 설명 
worldbody : 절대 좌표계 안에 존재하게됨. 
light pos 0,0,2에 떠있음. track CoM 로봇 무게중심을 따라 비춤. 
geom : 바닥, 무한히 뻗어있는 평면(plane), pos 0,0,-0.1 위치에 있음. 로봇이 0에 있으니 살짝 아래둔다. material grid 텍스쳐 

body : 강체 rigid body 질량과 관성을 가짐. 

link1 : 물체이름. pos 부모(world) 기준 0,0,0 위치에 이 몸체의 원점을 둠. 
joint : body가 부모에게 고정되지않고 움직일수있음. type hinge 경첩관절. 회전만 가능 1자유도 
axis 0 0 1 회전축이 z방향. 
pos 0 0 0 관절이 회전하는 중심이 link1기준 0 0 0임. 
geom type capsule 실제 link1의 모양을 정의함. 
fromto 시작점에서 끝점까지 캡슐그림. 0 0 0 + 0.5 0 0
즉, 링크의 길이는 x축 방향으로 0.5라는뜻. 


link2 : 두번째 강체 부모기준 위치 
axis : 0 0 1
pos  : 0 0 0 인 이유는 바디에서 pos 0.5 0 0으로 이동했기에. 

"""

# TODO : 각 xml이 어떻게 구도되고 의미가 무엇인지. 
# MuJoCo는 효율적인 시뮬레이션을 위해 정적인 정보와 동적인 정보를 엄격하게 분리

model = mujoco.MjModel.from_xml_string(xml) # 의미 
# 로봇의 링크 길이, 질량, 관성 모멘트, 관절 제한 범위, 마찰 계수 등 시뮬레이션 도중 변하지 않는 상수 값들

data = mujoco.MjData(model) # 의미
# 현재 관절의 각도(qpos), 속도(qvel), 가속도, 센서 측정값, 계산된 월드 좌표(site_xpos) 등 시뮬레이션 도중 매시간 변하는 값



# ---------------------------------------------------------
# 2. SE(3) 행렬 생성 함수 (Local Transformation)
# ---------------------------------------------------------
def get_local_se3(x_translation, theta_z):
    """
    x축으로 이동 후, z축 기준으로 회전하는 4x4 변환 행렬을 반환
    """
    c = np.cos(theta_z)
    s = np.sin(theta_z)
    
    # 4x4 Homogeneous Transformation Matrix
    T = np.array([
        [c, -s, 0, x_translation],  # 회전 및 X축 이동 포함
        [s,  c, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1]
    ])
    
    # 순서 주의: 로컬 좌표계 정의에 따라 
    # "현재 좌표계에서 이동(Trans)하고 -> 거기서 회전(Rot)" 할지
    # "회전(Rot)하고 -> 회전된 축으로 이동(Trans)" 할지에 따라 행렬 곱 순서가 달라짐.
    # MuJoCo XML 구조상: Link 길이만큼 가서(Trans) -> 관절이 회전(Rot)함.
    # 하지만 수식 계산의 편의를 위해 보통 T = Trans * Rot 형태로 만듭니다.
    
    # 여기서는 직관적으로 분리해서 계산해 보겠습니다:
    trans_mat = np.eye(4)
    trans_mat[0, 3] = x_translation
    
    rot_mat = np.eye(4)
    rot_mat[:2, :2] = np.array([[c, -s], [s, c]])
    
    # T = Trans * Rot (이동 후 그 자리에서 회전)
    # return trans_mat @ rot_mat # TODO : 오른쪽 곱셉이 아닌가? 즉, root @ trans @ 원점 이렇게가 아닌가?
    return T 

# ---------------------------------------------------------
# 3. 목표 설정 및 계산
# ---------------------------------------------------------
# 목표 각도
target_thetas = np.array([np.deg2rad(30), np.deg2rad(45)])
L1 = 0.5 # Link 1 길이
L2 = 0.5 # Link 2 길이

# A. MuJoCo 시뮬레이션 결과 (정답)
data.qpos = target_thetas  
mujoco.mj_forward(model, data)
ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
print('end effoector id :', ee_id)


mujoco_pos = data.site_xpos[ee_id]
# MuJoCo의 물리 엔진이 target_thetas 각도를 반영하여 내부적으로 계산해 낸 **End-Effector(발끝)의 전역(World) 좌표 $(x, y, z)$

# TODO : 여기서 의미하는 mujoco_pos가 무엇일까? 

print(f"--- [MuJoCo Result] ---\n{mujoco_pos}\n")


# B. Local SE(3) Chain 계산 (직접 구현)
# 순서: Base -> Joint1 회전 -> Link1 길이만큼 이동 -> Joint2 회전 -> Link2 길이만큼 이동

# 1. Base -> Joint 1 (원점에 있으므로 회전만 있음)
T_01 = get_local_se3(0, target_thetas[0]) 

# 2. Joint 1 -> Joint 2 (Link1 길이만큼 x 이동 후, Joint2 각도만큼 회전)
T_12 = get_local_se3(L1, target_thetas[1])

# 3. Joint 2 -> End Effector (Link2 길이만큼 x 이동, 회전 없음)
T_2E = get_local_se3(L2, 0)

T_pos = T_01 @ T_12 @ T_2E @ np.array([0,0,0,1])

print(T_pos)
# 최종 변환 행렬 (Chain Rule)
T_Global = T_01 @ T_12 @ T_2E
print(T_Global)
print()
# 위치 추출 (마지막 열의 x, y, z)
se3_pos = T_Global[:3, 3]

print(f"--- [Local SE(3) Result] ---\n{se3_pos}\n")


# C. 오차 확인
error = np.linalg.norm(mujoco_pos - se3_pos)
print(f"Difference (Error): {error:.6f}")

if error < 1e-5:
    print(">> Success! Local SE(3) calculation matches.")
else:
    print(">> Calculation Mismatch. Check the transform order.")

# ---------------------------------------------------------
# 4. 시각화
# ---------------------------------------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    # TODO : 위 함수가 의미하는 바를 알려줘.
    # launch_passive: MuJoCo의 3D 뷰어 창을 띄웁니다. "Passive"라는 단어는 **"물리 시뮬레이션이 자동으로 돌아가지 않는다
    viewer.sync()
    time.sleep(2) # 결과 확인을 위해 잠시 대기
