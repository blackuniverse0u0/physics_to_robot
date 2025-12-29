"""
Test if cosine_moving logic works
"""
import mujoco
import numpy as np

MODEL_XML_PATH = "../Furo/scene.xml"

def cosine_interpolate(q_start, q_end, t, period):
    alpha = 0.5 * (1 - np.cos(2 * np.pi * t / period))
    return q_start + (q_end - q_start) * alpha

# Load model
m = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
d = mujoco.MjData(m)

# Poses from cosine_moving.py
q_down = np.array([
    0,  0.6, -1.4,   # FL
    0, -0.6,  1.4,   # FR
    0,  0.6, -1.4,   # RL
    0, -0.6,  1.4    # RR
])

q_up = np.array([
    0,  0.8, -1.6,   # FL
    0, -0.8,  1.6,   # FR
    0,  0.8, -1.6,   # RL
    0, -0.8,  1.6    # RR
])

# Initialize (like cosine_moving.py)
d.qpos[7:19] = q_down

print("=== Testing cosine_moving logic ===")
print(f"Initial config:")
print(f"  Base height: {d.qpos[2]:.3f} m")
print(f"  Joints: {d.qpos[7:19]}")
print()

freq = 0.5  # Hz
wait_time = 1.0
num_steps = 2000  # 4 seconds

for step in range(num_steps):
    t = step * m.opt.timestep

    if t < wait_time:
        # Standing
        target_qpos = q_down
    else:
        # Moving
        t_move = t - wait_time
        target_qpos = cosine_interpolate(q_down, q_up, freq * t_move, 1.0)

    d.ctrl[:] = target_qpos
    mujoco.mj_step(m, d)

    # Print every 0.5s
    if step % 250 == 0:
        max_force = max(abs(d.actuator_force))
        at_limit = any(abs(d.actuator_force) >= 49.9)
        status = "AT_LIMIT!" if at_limit else "ok"
        print(f"t={t:.2f}s: h={d.qpos[2]:.3f}m, max_f={max_force:5.1f}N [{status}]")

print()
if d.qpos[2] > 0.4:
    print(f"✅ Robot maintained reasonable height: {d.qpos[2]:.3f}m")
else:
    print(f"⚠️  Robot collapsed: {d.qpos[2]:.3f}m")
