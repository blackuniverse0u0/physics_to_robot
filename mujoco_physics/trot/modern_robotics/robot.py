# import mujoco
# import mujoco.viewer
# import numpy as np
# import time

# # Load the model
# model = mujoco.MjModel.from_xml_path('robot.xml')
# data = mujoco.MjData(model)

# with mujoco.viewer.launch_passive(model, data) as viewer:
#     start_time = time.time()
    
#     while viewer.is_running():
#         step_start = time.time()
#         curr_t = time.time() - start_time

#         # Target positions (in radians)
#         # Joint 1 (Hip), Joint 2 (Thigh), Joint 3 (Knee)
#         target_q = np.array([
#             0.5 * np.sin(curr_t),      # Hip swing
#             0.7 * np.sin(curr_t),      # Thigh flex
#             -1.2 * np.abs(np.sin(curr_t)) # Knee "tuck"
#         ])
        
#         # PD Control Parameters
#         kp = 10.0
#         kd = 1.0
        
#         # Control Law: tau = Kp(q_des - q) - Kd(v)
#         torques = kp * (target_q - data.qpos) - kd * data.qvel
        
#         # Apply to actuators
#         data.ctrl[:] = torques

#         mujoco.mj_step(model, data)
#         viewer.sync()

#         # Real-time synchronization
#         elapsed = time.time() - step_start
#         if elapsed < model.opt.timestep:
#             time.sleep(model.opt.timestep - elapsed)



import mujoco
import mujoco.viewer
import numpy as np
import time

model = mujoco.MjModel.from_xml_path('robot.xml')
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    
    while viewer.is_running():
        step_start = time.time()
        t = time.time() - start_time

        # Target angles for Left Leg
        L_target = np.array([
            0.1 * np.sin(t),       # Roll
            0.5 * np.sin(t),       # Hip Pitch
            -0.8 * np.abs(np.sin(t)) # Knee Pitch
        ])

        # Target angles for Right Leg (Opposite phase)
        R_target = np.array([
            0.1 * np.sin(t + np.pi), 
            0.5 * np.sin(t + np.pi), 
            -0.8 * np.abs(np.sin(t + np.pi))
        ])

        # Combined targets
        all_targets = np.concatenate([L_target, R_target])
        
        # PD Control (Joint indices start after the 7 freejoint coordinates)
        # qpos index: 0-6 (freejoint), 7-9 (Left), 10-12 (Right)
        kp = 15.0
        kd = 1.0
        
        # current_q = data.qpos[7:] 
        # current_v = data.qvel[6:] # Velocity index for freejoint is 6
        
        current_q = data.qpos[:]  # Takes all 6 joint positions
        current_v = data.qvel[:]  # Takes all 6 joint velocities
        
        torques = kp * (all_targets - current_q) - kd * current_v
        # torques = 0 
        data.ctrl[:] = torques

        mujoco.mj_step(model, data)
        viewer.sync()

        elapsed = time.time() - step_start
        if elapsed < model.opt.timestep:
            time.sleep(model.opt.timestep - elapsed)