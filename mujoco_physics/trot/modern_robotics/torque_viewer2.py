import mujoco
import glfw
import numpy as np
from collections import deque

# --- 1. 모델 정의 (이중 진자) ---
xml_string = """
<mujoco>
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 0 3"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba=".8 .9 .8 1"/>
    <body name="link1" pos="0 0 2">
      <joint name="joint1" type="hinge" axis="0 1 0" damping="0.0"/>
      <geom type="capsule" size="0.05" fromto="0 0 0 0 0 -1" rgba="1 0.5 0.5 1"/>
      <body name="link2" pos="0 0 -1">
        <joint name="joint2" type="hinge" axis="0 1 0" damping="0.0"/>
        <geom type="capsule" size="0.05" fromto="0 0 0 0 0 -1" rgba="0.5 0.5 1 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# 모델 및 데이터 로드
model = mujoco.MjModel.from_xml_string(xml_string)
data = mujoco.MjData(model)

# --- 2. GLFW 및 MuJoCo 컨텍스트 초기화 ---
if not glfw.init():
    raise Exception("GLFW initialization failed")

# 맥북 등 HiDPI 화면 대응을 위해 윈도우 크기 넉넉히 잡기
window = glfw.create_window(1200, 900, "MuJoCo Native Plotting", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# MuJoCo 시각화 데이터 구조
cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()
scn = mujoco.MjvScene(model, maxgeom=1000)
con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

# 카메라 초기 설정
mujoco.mjv_defaultCamera(cam)
cam.distance = 5
cam.lookat = [0, 0, 1]

# --- 3. 그래프(Figure) 초기화 ---
fig = mujoco.MjvFigure()
mujoco.mjv_defaultFigure(fig)

# [중요] 에러를 피하기 위해 필수 설정만 남깁니다.
fig.title = "Joint 1 Torque"
fig.xlabel = "Time"

# Y축 범위 수동 설정 (자동 확장이 안될 수 있으므로)
# 이중 진자의 토크는 보통 -10 ~ 10 사이를 오갑니다.
fig.range[1][0] = -20.0  # Min
fig.range[1][1] = 20.0   # Max

# 데이터 버퍼
num_points = 300
torque_history = deque([0.0]*num_points, maxlen=num_points)

# 라인 포인트 개수 설정 (이것은 필수입니다)
fig.linepnt[0] = num_points

# [삭제됨] fig.rgb, fig.flg_bar 등 에러 유발 속성 제거
# 기본 색상(흰색 또는 회색)으로 그려집니다.

# --- 4. 메인 루프 ---
while not glfw.window_should_close(window):
    # (1) 물리 시뮬레이션
    sim_start = data.time
    while data.time - sim_start < 1.0/60.0:
        mujoco.mj_step(model, data)
        
    # (2) 데이터 수집
    # 관절 1의 수동 토크(중력 등)
    val = data.qfrc_passive[0]
    torque_history.append(val)

    # (3) mjvFigure 데이터 업데이트
    # linedata는 1차원 배열: [x0, y0, x1, y1, ...]
    for i in range(num_points):
        p_idx = 2 * i
        # X축: -299 ~ 0 (과거 ~ 현재)
        fig.linedata[0][p_idx] = float(i - num_points)
        # Y축: 토크 값
        fig.linedata[0][p_idx + 1] = torque_history[i]

    # (4) 렌더링
    # 현재 창 크기 가져오기
    width, height = glfw.get_framebuffer_size(window)
    viewport = mujoco.MjrRect(0, 0, width, height)
    
    # 3D 장면 업데이트 및 렌더링
    mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
    mujoco.mjr_render(viewport, scn, con)

    # 그래프 그리기 (우측 하단)
    # 창 크기에 비례하여 위치 잡기 (우측 하단 1/4 크기)
    fig_w = int(width / 3)
    fig_h = int(height / 4)
    fig_x = width - fig_w - 10
    fig_y = 10 # OpenGL은 아래가 0
    
    fig_viewport = mujoco.MjrRect(fig_x, fig_y, fig_w, fig_h)
    mujoco.mjr_figure(fig_viewport, fig, con)

    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()