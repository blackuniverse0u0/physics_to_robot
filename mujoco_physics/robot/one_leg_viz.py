import mujoco
import glfw
import numpy as np
from collections import deque
import time

# --- 1. XML 정의 (트리플 펜듈럼) ---
xml = """
<mujoco model="biped_triple_pendulum">
    <compiler angle="radian" coordinate="local" inertiafromgeom="false"/>
    <option gravity="0 0 -9.81" integrator="RK4" timestep="0.01"/>
    
    <visual>
        <rgba haze="0.15 0.25 0.35 1"/>
        <quality shadowsize="2048"/>
        <map stiffness="700" shadowscale="0.5" fogstart="10" fogend="15" zfar="40" haze="0.3"/>
    </visual>

    <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="2 2 0.1" rgba=".9 .9 .9 1"/>
        
        <body name="base_link" pos="0 0 2.0">
            <inertial pos="0 0 0" mass="2.0" diaginertia="0.01 0.01 0.01"/>
            <geom type="box" size="0.2 0.15 0.05" rgba="0.2 0.2 0.2 1"/>
            
            <body name="L_link1" pos="0 0.15 0">
                <inertial pos="0 0.05 0" mass="0.5" diaginertia="0.001 0.001 0.001"/>
                <joint name="L_joint1" type="hinge" axis="1 0 0" pos="0 0 0"/> 
                <geom type="capsule" fromto="0 0 0 0 0.1 0" size="0.03" rgba="1 0 0 1"/>
                
                <body name="L_link2" pos="0 0.1 0">
                    <inertial pos="0 0 -0.15" mass="1.0" diaginertia="0.005 0.005 0.001"/>
                    <joint name="L_joint2" type="hinge" axis="0 1 0" pos="0 0 0"/> 
                    <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.03" rgba="0 1 0 1"/>
                    
                    <body name="L_link3" pos="0 0 -0.3">
                        <inertial pos="0 0 -0.15" mass="0.8" diaginertia="0.004 0.004 0.001"/>
                        <joint name="L_joint3" type="hinge" axis="0 1 0" pos="0 0 0"/> 
                        <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.03" rgba="0 0 1 1"/>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>

    <actuator>
        <motor name="motor_L1" joint="L_joint1" gear="1"/>
        <motor name="motor_L2" joint="L_joint2" gear="1"/>
        <motor name="motor_L3" joint="L_joint3" gear="1"/>
    </actuator>
</mujoco>
"""

# 모델 로드
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# --- 2. 초기화 (GLFW) ---
if not glfw.init():
    raise Exception("GLFW initialization failed")

window = glfw.create_window(1200, 900, "Multi-Joint Torque Plot", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# MuJoCo 구조체
cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()
scn = mujoco.MjvScene(model, maxgeom=1000)
con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

# 카메라 설정
mujoco.mjv_defaultCamera(cam)
cam.distance = 2.5
cam.lookat = [0, 0, 1.5]
cam.azimuth = 130
cam.elevation = -20

# --- 3. 그래프(Figure) 설정 ---
fig = mujoco.MjvFigure()
mujoco.mjv_defaultFigure(fig)

fig.title = "Torques (Joint 1, 2, 3)"
fig.xlabel = "Time"
fig.range[1][0] = -5.0  # Y축 Min
fig.range[1][1] = 5.0   # Y축 Max

# 데이터 버퍼 설정 (3개의 라인을 위한 리스트)
num_points = 300
num_lines = 3
history = [deque([0.0]*num_points, maxlen=num_points) for _ in range(num_lines)]

# 각 라인의 포인트 개수 설정
for i in range(num_lines):
    fig.linepnt[i] = num_points

# [옵션] 라인 이름 및 색상 설정
# MuJoCo 버전에 따라 rgb 속성 이름이 다를 수 있어 try-except로 처리
try:
    # 색상: 빨강, 초록, 파랑
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] 
    names = ["J1", "J2", "J3"]
    
    for i in range(num_lines):
        fig.linename[i] = names[i]
        fig.rgb[i] = colors[i]
except:
    print("Warning: Could not set graph colors/names (API mismatch). Graph will still draw.")

# --- 4. 유틸리티 함수: 화살표 회전 행렬 ---
def get_arrow_mat(direction):
    direction = direction / (np.linalg.norm(direction) + 1e-6)
    quat = np.zeros(4)
    mujoco.mju_quatZ2Vec(quat, direction)
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat)
    return mat

# --- 5. 메인 루프 ---
joint_names = ["L_joint1", "L_joint2", "L_joint3"]
joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]

while not glfw.window_should_close(window):
    # (A) 물리 시뮬레이션 및 제어
    sim_start = data.time
    while data.time - sim_start < 1.0/60.0:
        t = data.time
        
        # 1. 사인파 생성
        sine_wave = np.array([
            2.0 * np.sin(t * 2.0),
            3.0 * np.sin(t * 2.0 + 0.5),
            1.5 * np.sin(t * 2.0 + 1.0)
        ])

        # 2. 제어 입력
        if model.nu == 3:
            data.ctrl[:] = (data.qfrc_bias[:3] + sine_wave) / 5.0

        mujoco.mj_step(model, data)
    
    # (B) 데이터 수집 (3개 관절 모두)
    for i in range(num_lines):
        # 각 관절(i)의 토크값 가져오기
        torque_val = data.qfrc_actuator[joint_ids[i]]
        history[i].append(torque_val)

    # (C) 뷰 업데이트
    width, height = glfw.get_framebuffer_size(window)
    viewport = mujoco.MjrRect(0, 0, width, height)
    
    # 그래프 데이터 업데이트 (3개 라인 모두)
    for line_idx in range(num_lines):
        for pt_idx in range(num_points):
            # X축: 시간 (상대적)
            fig.linedata[line_idx][2*pt_idx] = float(pt_idx - num_points)
            # Y축: 토크 값
            fig.linedata[line_idx][2*pt_idx+1] = history[line_idx][pt_idx]

    # 기본 씬 업데이트
    mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)

    # (D) 화살표 시각화 (3개 관절 모두)
    for i, j_id in enumerate(joint_ids):
        torque = data.qfrc_actuator[j_id]
        if abs(torque) > 0.1:
            pos = data.xanchor[j_id]
            axis = data.xaxis[j_id]
            direction = axis * np.sign(torque)
            scale = abs(torque) * 0.05 
            size = np.array([0.015, 0.015, scale])
            
            # 색상: 그래프와 동일하게 맞춤 (빨, 초, 파)
            arrow_rgba = [0, 0, 0, 1]
            if i == 0: arrow_rgba = [1, 0, 0, 1] # Red
            elif i == 1: arrow_rgba = [0, 1, 0, 1] # Green
            elif i == 2: arrow_rgba = [0, 0, 1, 1] # Blue

            if scn.ngeom < scn.maxgeom:
                mujoco.mjv_initGeom(
                    scn.geoms[scn.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_ARROW,
                    size=size,
                    pos=pos,
                    mat=get_arrow_mat(direction),
                    rgba=np.array(arrow_rgba)
                )
                scn.ngeom += 1

    # (E) 렌더링
    mujoco.mjr_render(viewport, scn, con)

    # 그래프 그리기 (우측 하단)
    fig_viewport = mujoco.MjrRect(width - 400, 0, 400, 300)
    mujoco.mjr_figure(fig_viewport, fig, con)

    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()