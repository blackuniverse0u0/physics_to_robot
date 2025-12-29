import mujoco as mj
from mujoco.glfw import glfw
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

flag_trajectory_generation = 0

# Furo 로봇 XML 파일 경로
xml_path = '../Furo/scene.xml'
simend = 10  # 시뮬레이션 시간 (초)
print_camera_config = 0
print_model = 0

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0


def init_controller(model, data):
    """제어기 초기화"""
    pass


def controller(model, data):
    """제어 루프"""
    pass


def keyboard(window, key, scancode, act, mods):
    """키보드 콜백"""
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)


def mouse_button(window, button, act, mods):
    """마우스 버튼 콜백"""
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    glfw.get_cursor_pos(window)


def mouse_move(window, xpos, ypos):
    """마우스 이동 콜백"""
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    if (not button_left) and (not button_middle) and (not button_right):
        return

    width, height = glfw.get_window_size(window)

    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)


def scroll(window, xoffset, yoffset):
    """스크롤 콜백"""
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)


# 경로 설정
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo 데이터 구조
model = mj.MjModel.from_xml_path(xml_path)
print(f"Model nq: {model.nq}")
data = mj.MjData(model)
cam = mj.MjvCamera()
opt = mj.MjvOption()

# GLFW 초기화
glfw.init()
window = glfw.create_window(1200, 900, "Furo Trot Gait", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# 시각화 데이터 구조 초기화
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# GLFW 콜백 설치
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# 제어기 초기화
init_controller(model, data)
mj.set_mjcb_control(controller)

# 전역 변수 초기화
globals.init()

# Keyframe을 사용하여 초기 자세 설정
if model.nkey > 0:
    # Keyframe 0 ("home") 사용
    mj.mj_resetDataKeyframe(model, data, 0)
    print(f"Loaded keyframe 'home' with {model.nq} DOFs")
else:
    # Keyframe이 없으면 수동 설정
    print("Warning: No keyframe found, using manual initialization")
    data.qpos[:] = 0
    data.qpos[2] = 0.55  # Base height
    data.qpos[3] = 1.0   # Quaternion w

# 로봇 관절 각도 추출 (base freejoint 7 + leg joints 12)
hip_roll = 0.0
hip_pitch = data.qpos[8]   # FL hip pitch
knee = data.qpos[9]        # FL knee

# 순기구학으로 초기 발 위치 계산
sol = forward_kinematics_leg(np.array([hip_roll, hip_pitch, knee]), 0)
end_eff_pos = sol.end_eff_pos
lz0 = end_eff_pos[2]
print(f"Initial foot height in leg frame: {lz0:.4f} m")
print(f"Initial joint angles: hip_roll={hip_roll:.3f}, hip_pitch={hip_pitch:.3f}, knee={knee:.3f}")

# 메인 루프
while not glfw.window_should_close(window):
    time_prev = data.time

    while (data.time - time_prev < 1.0/15.0):

        state_machine()
        cartesian_traj()
        joint_traj()
        globals.time = data.time

        if (flag_trajectory_generation == 1):  # 운동학 모드
            data.time += 0.001
            # 로봇 부분만 업데이트 (base 7 + legs 12 = indices 0:19)
            data.qpos[7:19] = globals.q_ref
            mj.mj_forward(model, data)
        else:  # 동역학 모드
            # 로봇 관절만 읽기 (base 7 DOF 이후 12개 관절)
            globals.q_act = data.qpos[7:19].copy()
            globals.u_act = data.qvel[6:18].copy()
            globals.pos_quat_trunk = data.qpos[:7].copy()
            globals.vel_angvel_trunk = data.qvel[:6].copy()
            joint_control()
            high_level_control()
            data.ctrl = globals.trq.copy()
            mj.mj_step(model, data)

    if (data.time >= simend):
        break

    # Get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    # 카메라 설정 (로봇 추적)
    if (print_camera_config == 1):
        print('cam.azimuth =', cam.azimuth, ';', 'cam.elevation =',
              cam.elevation, ';', 'cam.distance = ', cam.distance)
        print('cam.lookat =np.array([', cam.lookat[0], ',',
              cam.lookat[1], ',', cam.lookat[2], '])')

    # 카메라가 로봇 따라가기
    cam.lookat[0] = data.qpos[0]
    cam.lookat[1] = data.qpos[1]
    cam.azimuth = 90
    cam.elevation = -30
    cam.distance = 3.5

    # 씬 업데이트 및 렌더링
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # OpenGL 버퍼 교환
    glfw.swap_buffers(window)

    # GUI 이벤트 처리
    glfw.poll_events()

glfw.terminate()

print("\n=== Furo Trot Simulation Complete ===")
print(f"Final position: x={data.qpos[0]:.3f} m, y={data.qpos[1]:.3f} m")
print(f"Final orientation (yaw): {np.arctan2(2*(data.qpos[3]*data.qpos[6] + data.qpos[4]*data.qpos[5]), 1-2*(data.qpos[5]**2+data.qpos[6]**2)):.3f} rad")
