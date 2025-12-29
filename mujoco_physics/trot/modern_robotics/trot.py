
import mujoco
import mujoco.viewer
import numpy as np
import time

model = mujoco.MjModel.from_xml_path('quadruped.xml')
data = mujoco.MjData(model)

def get_leg_target(t, phase_offset=0):
    # Smooth gait trajectory
    swing = 0.25 * np.sin(2 * np.pi * 1.0 * t + phase_offset)
    thigh = 0.4 * swing + 0.1
    knee = -0.5 * np.abs(swing) - 0.5
    return np.array([0, thigh, knee])

with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    
    while viewer.is_running():
        step_start = time.time()
        t = time.time() - start_time

        # Target positions for all 12 joints
        targets = np.concatenate([
            get_leg_target(t, 0),      # Front Left
            get_leg_target(t, np.pi),  # Front Right
            get_leg_target(t, np.pi),  # Rear Left
            get_leg_target(t, 0)       # Rear Right
        ])

        # Simply set the control targets (angles in radians)
        data.ctrl[:] = targets

        mujoco.mj_step(model, data)
        viewer.sync()

        # Real-time sync
        elapsed = time.time() - step_start
        if elapsed < model.opt.timestep:
            time.sleep(model.opt.timestep - elapsed)