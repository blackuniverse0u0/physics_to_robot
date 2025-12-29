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

xml_path = '../unitree_robotics_a1/scene.xml' #xml file (assumes this is in the same folder as this file)
simend = 10 #simulation time
print_camera_config = 0 #set to 1 to print camera config
                        #this is useful for initializing view of the model)
print_model = 0 #set to 1 to print the model info in the file model.txt in the current location

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

def init_controller(model,data):
    #initialize the controller here. This function is called once, in the beginning
    pass

def controller(model, data):
    #put the controller here. This function is called inside the simulation.
    pass

def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
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
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)

#get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

#print the model
if (print_model==1):
    model_name = 'model.txt'
    model_path = os.path.join(dirname + "/" + model_name)
    mujoco.mj_printModel(model,model_path)


# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                # MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# Example on how to set camera configuration
# cam.azimuth = 90
# cam.elevation = -45
# cam.distance = 2
# cam.lookat = np.array([0.0, 0.0, 0])

#initialize the controller
init_controller(model,data)

#set the controller
mj.set_mjcb_control(controller)

globals.init()

hip = 0
pitch = 0.9
knee = -1.8

sol = forward_kinematics_leg(np.array([hip,pitch,knee]),0)
end_eff_pos = sol.end_eff_pos
lz0 = end_eff_pos[2]
#print(lz0)

pos = np.array([0, 0, 0.3])
quat = np.array([1,0,0,0])
#euler = np.array([0,0,np.pi/2])
#quat = ram.bryant2quat(euler)

qleg = np.array([hip,pitch,knee])
data.qpos = np.concatenate((pos,quat,qleg,qleg,qleg,qleg))

while not glfw.window_should_close(window):
    time_prev = data.time



    while (data.time - time_prev < 1.0/15.0):

        state_machine()
        cartesian_traj()
        joint_traj()
        globals.time = data.time

        if (flag_trajectory_generation==1): #kinematic mode
            data.time += 0.001 #model.opt.timestep
            data.qpos = np.concatenate((pos,quat,globals.q_ref))
            mj.mj_forward(model,data)
        else: #dynamic mode
            globals.q_act = data.qpos[7:].copy()
            globals.u_act = data.qvel[6:].copy()
            globals.pos_quat_trunk = data.qpos[:7].copy()
            globals.vel_angvel_trunk = data.qvel[:6].copy()
            joint_control()
            high_level_control()
            data.ctrl = globals.trq.copy()
            mj.mj_step(model, data)

    if (data.time>=simend):
        break;

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    #print camera configuration (help to initialize the view)
    if (print_camera_config==1):
        print('cam.azimuth =',cam.azimuth,';','cam.elevation =',cam.elevation,';','cam.distance = ',cam.distance)
        print('cam.lookat =np.array([',cam.lookat[0],',',cam.lookat[1],',',cam.lookat[2],'])')

    cam.lookat =np.array([ 1.0462306782204185 , 1.743116392639941 , 0.0 ])
    cam.azimuth = 97.06857368262065 ; cam.elevation = -50.88745261308294 ; cam.distance =  4.28353653567233


    cam.lookat[0] = data.qpos[0]
    cam.lookat[1] = data.qpos[1]
    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

glfw.terminate()
