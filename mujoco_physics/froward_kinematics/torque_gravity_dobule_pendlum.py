import mujoco
import mujoco.viewer
import numpy as np
import time

# 1. Load Model
model = mujoco.MjModel.from_xml_path("double_pendulum.xml")
data = mujoco.MjData(model)

# Constants for Gravity Compensation
L1, L2 = 0.5, 0.5
m1, m2 = 1.0, 1.0
g = 9.81

def compute_gravity_torque(q):
    t1, t2 = q[0], q[1]
    x1 = L1 * np.cos(t1)
    x2 = L1 * np.cos(t1) + L2 * np.cos(t1 + t2)
    tau1 = m1 * g * x1 + m2 * g * x2
    tau2 = m2 * g * (L2 * np.cos(t1 + t2))
    return tau1, tau2

# 2. Setup MuJoCo Figure (Internal Plot)
fig = mujoco.MjvFigure()
mujoco.mjv_defaultFigure(fig)
fig.title = "Joint Torques (Live)"
fig.xlabel = "Time Steps"
fig.range[0][0], fig.range[0][1] = -100, 0   # X-axis: last 100 steps
fig.range[1][0], fig.range[1][1] = -30, 30    # Y-axis: Torque range
fig.linepnt[0] = 0
fig.linepnt[1] = 0

def update_plot(fig, t1, t2):
    # Shift data points to the left and add new point
    pnt = min(200, fig.linepnt[0] + 1)
    for i in range(2): # Two lines: Tau1 and Tau2
        if fig.linepnt[i] < 200:
            fig.linepnt[i] += 1
        else:
            fig.linedata[i][0:199*2] = fig.linedata[i][2:200*2]
        
        # Set X and Y
        val = t1 if i == 0 else t2
        fig.linedata[i][(fig.linepnt[i]-1)*2] = -fig.linepnt[i]
        fig.linedata[i][(fig.linepnt[i]-1)*2 + 1] = val

# 3. Main Loop
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Set initial position
    data.qpos[0] = np.pi/4
    data.qpos[1] = np.pi/6
    
    # Customizing UI: Add the figure to the overlay
    viewer.user_scn.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = 1

    step_counter = 0
    while viewer.is_running():
        step_start = time.time()

        # Apply Gravity Compensation
        tau1, tau2 = compute_gravity_torque(data.qpos)
        data.ctrl[0] = tau1
        data.ctrl[1] = tau2

        # Step simulation
        mujoco.mj_step(model, data)

        # Update Live Plot every 10 steps to save performance
        if step_counter % 10 == 0:
            update_plot(fig, tau1, tau2)
            # Add figure to the viewer's overlay
            viewer.overlay[mujoco.mjtGridPos.mjGRID_TOPRIGHT] = fig
        
        viewer.sync()
        step_counter += 1

        # Real-time sync
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)