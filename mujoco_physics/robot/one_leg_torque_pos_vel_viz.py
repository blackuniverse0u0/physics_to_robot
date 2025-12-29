import mujoco
import glfw
import numpy as np
from collections import deque

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

window = glfw.create_window(1200, 900, "Triple Plot: Pos, Vel, Torque", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# MuJoCo 구조체
cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()
scn = mujoco.MjvScene(model, maxgeom=1000)
con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

mujoco.mjv_defaultCamera(cam)
cam.distance = 2.5
cam.lookat = [0, 0, 1.5]
cam.azimuth = 130
cam.elevation = -20

# --- 3. 그래프 3개 설정 함수 ---
def init_figure(title, y_min, y_max, num_points=300):
    fig = mujoco.MjvFigure()
    mujoco.mjv_defaultFigure(fig)
    fig.title = title
    fig.xlabel = "Time"
    fig.range[1][0] = y_min
    fig.range[1][1] = y_max
    
    # 3개의 라인 (Joint 1, 2, 3)
    for i in range(3):
        fig.linepnt[i] = num_points
        
    # 스타일 (색상 설정 시도)
    try:
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] # Red, Green, Blue
        names = ["J1", "J2", "J3"]
        for i in range(3):
            fig.linename[i] = names[i]
            fig.rgb[i] = colors[i]
    except:
        pass
    return fig

# 3개의 Figure 생성
# 1. Position (각도): 대략 -3 ~ 3 라디안
fig_pos = init_figure("Joint Position (rad)", -3.5, 3.5)
# 2. Velocity (속도): 대략 -10 ~ 10 rad/s
fig_vel = init_figure("Joint Velocity (rad/s)", -10.0, 10.0)
# 3. Torque (토크): 대략 -5 ~ 5 N*m
fig_torque = init_figure("Actuator Torque (N*m)", -6.0, 6.0)

# 데이터 저장소 (3개 그래프 x 3개 관절)
num_points = 300
# 각각 deque 3개씩을 담은 리스트
hist_pos = [deque([0.0]*num_points, maxlen=num_points) for _ in range(3)]
hist_vel = [deque([0.0]*num_points, maxlen=num_points) for _ in range(3)]
hist_tor = [deque([0.0]*num_points, maxlen=num_points) for _ in range(3)]

# --- 4. 유틸리티 ---
def get_arrow_mat(direction):
    direction = direction / (np.linalg.norm(direction) + 1e-6)
    quat = np.zeros(4)
    mujoco.mju_quatZ2Vec(quat, direction)
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat)
    return mat

def update_figure_data(fig, history_list):
    """Figure 구조체에 deque 데이터를 채워넣는 헬퍼 함수"""
    for line_idx in range(3):
        for pt_idx in range(num_points):
            fig.linedata[line_idx][2*pt_idx] = float(pt_idx - num_points)
            fig.linedata[line_idx][2*pt_idx+1] = history_list[line_idx][pt_idx]

# --- 5. 메인 루프 ---
joint_names = ["L_joint1", "L_joint2", "L_joint3"]
joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
# qpos/qvel 인덱스는 joint_id와 다를 수 있으므로(hinge는 같지만), 안전하게 주소 찾기
qpos_adr = [model.jnt_qposadr[jid] for jid in joint_ids]
dof_adr = [model.jnt_dofadr[jid] for jid in joint_ids]

while not glfw.window_should_close(window):
    # (A) 물리 시뮬레이션
    sim_start = data.time
    while data.time - sim_start < 1.0/60.0:
        t = data.time
        # 사인파 제어
        sine_wave = np.array([
            2.0 * np.sin(t * 2.0),
            3.0 * np.sin(t * 2.0 + 0.5),
            1.5 * np.sin(t * 2.0 + 1.0)
        ])
        if model.nu == 3:
            data.ctrl[:] = (data.qfrc_bias[:3] + sine_wave) / 5.0
        mujoco.mj_step(model, data)

    # (B) 데이터 수집 (모든 관절, 모든 물리량)
    for i in range(3):
        # 1. Position
        hist_pos[i].append(data.qpos[qpos_adr[i]])
        # 2. Velocity
        hist_vel[i].append(data.qvel[dof_adr[i]])
        # 3. Torque (Actuator Force)
        hist_tor[i].append(data.qfrc_actuator[joint_ids[i]])

    # (C) 뷰 및 그래프 업데이트
    width, height = glfw.get_framebuffer_size(window)
    viewport = mujoco.MjrRect(0, 0, width, height)
    
    # 그래프 데이터 갱신
    update_figure_data(fig_pos, hist_pos)
    update_figure_data(fig_vel, hist_vel)
    update_figure_data(fig_torque, hist_tor)

    # 3D 씬 업데이트
    mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)

    # 화살표 시각화 (Torque만 표시)
    for i, j_id in enumerate(joint_ids):
        torque = data.qfrc_actuator[j_id]
        if abs(torque) > 0.1:
            pos = data.xanchor[j_id]
            axis = data.xaxis[j_id]
            direction = axis * np.sign(torque)
            scale = abs(torque) * 0.05
            size = np.array([0.015, 0.015, scale])
            
            # 색상 매칭 (Red, Green, Blue)
            rgba = [0,0,0,1]
            if i==0: rgba=[1,0,0,1]
            elif i==1: rgba=[0,1,0,1]
            elif i==2: rgba=[0,0,1,1]

            if scn.ngeom < scn.maxgeom:
                mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_ARROW, 
                                    size, pos, get_arrow_mat(direction), np.array(rgba))
                scn.ngeom += 1

    # (D) 최종 렌더링
    mujoco.mjr_render(viewport, scn, con)

    # --- 그래프 3개 배치 (우측 사이드바 형태) ---
    # 화면 높이를 3등분하여 사용
    plot_width = 400
    plot_height = int(height / 4) # 화면 높이의 1/4 정도 크기
    spacing = 10 # 그래프 간 간격
    
    # 1. Bottom: Torque
    rect_torque = mujoco.MjrRect(width - plot_width, spacing, plot_width, plot_height)
    mujoco.mjr_figure(rect_torque, fig_torque, con)
    
    # 2. Middle: Velocity
    rect_vel = mujoco.MjrRect(width - plot_width, spacing + plot_height + spacing, plot_width, plot_height)
    mujoco.mjr_figure(rect_vel, fig_vel, con)
    
    # 3. Top: Position
    rect_pos = mujoco.MjrRect(width - plot_width, spacing + 2*(plot_height + spacing), plot_width, plot_height)
    mujoco.mjr_figure(rect_pos, fig_pos, con)

    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()