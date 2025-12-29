"""
Furo Robot Trot Gait Controller (Working Version)
Fixed force limits and proper initialization
"""
import mujoco
import mujoco.viewer
import numpy as np
import time

MODEL_XML_PATH = "scene_corrected.xml"

def cosine_interpolate(q_start, q_end, t, period):
    """Smooth cosine interpolation"""
    alpha = 0.5 * (1 - np.cos(2 * np.pi * t / period))
    return q_start + (q_end - q_start) * alpha

# Standing pose (from keyframe)
q_stand = np.array([
    0,  0.6, -1.4,   # FL
    0, -0.6,  1.4,   # FR
    0,  0.6, -1.4,   # RL
    0, -0.6,  1.4    # RR
])

# Trot Phase 1: FL + RR swing
q_trot_phase1 = np.array([
    0,  0.7, -1.5,   # FL (swing)
    0, -0.5,  1.3,   # FR (stance)
    0,  0.5, -1.3,   # RL (stance)
    0, -0.7,  1.5    # RR (swing)
])

# Trot Phase 2: FR + RL swing
q_trot_phase2 = np.array([
    0,  0.5, -1.3,   # FL (stance)
    0, -0.7,  1.5,   # FR (swing)
    0,  0.7, -1.5,   # RL (swing)
    0, -0.5,  1.3    # RR (stance)
])

def main():
    # Load model
    m = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
    d = mujoco.MjData(m)

    print("=" * 70)
    print("Furo Robot Trot Gait Controller (Corrected)")
    print("=" * 70)
    print(f"Model: {m.nq} DOF, {m.nu} actuators")
    print(f"Force limits: ±{m.actuator_forcerange[0,1]:.0f} N")
    print(f"kp gains: {m.actuator_gainprm[0,0]:.0f}")
    print()

    # Timing
    duration = 20.0
    wait_time = 2.0
    trot_period = 1.0

    # CRITICAL: Initialize using keyframe for proper base height!
    mujoco.mj_resetDataKeyframe(m, d, 0)

    print(f"Initialized from keyframe:")
    print(f"  Base height: {d.qpos[2]:.3f} m")
    print(f"  Joint config: {d.qpos[7:19]}")
    print()

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start_time = time.time()

        print(f"Phase 1: Standing ({wait_time}s)")
        print(f"Phase 2: Trotting (period={trot_period}s)")
        print()

        while viewer.is_running():
            now = time.time() - start_time

            if now > duration:
                print(f"\nSimulation complete at t={now:.2f}s")
                break

            # Control logic
            if now < wait_time:
                # Standing
                target_qpos = q_stand
            else:
                # Trotting
                t_move = now - wait_time
                t_in_cycle = t_move % trot_period

                if t_in_cycle < trot_period / 2:
                    # Phase 1 -> Phase 2
                    target_qpos = cosine_interpolate(
                        q_trot_phase1, q_trot_phase2,
                        t_in_cycle, trot_period / 2
                    )
                else:
                    # Phase 2 -> Phase 1
                    target_qpos = cosine_interpolate(
                        q_trot_phase2, q_trot_phase1,
                        t_in_cycle - trot_period / 2, trot_period / 2
                    )

            # Apply control
            d.ctrl[:] = target_qpos

            # Step physics
            step_start = time.time()
            mujoco.mj_step(m, d)
            viewer.sync()

            # Real-time synchronization
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            # Progress output (every 1 second)
            if int(now) != int(now - m.opt.timestep):
                pos = d.qpos[:3]
                max_force = max(abs(d.actuator_force))
                force_pct = max_force / m.actuator_forcerange[0,1] * 100

                if now < wait_time:
                    phase_str = "Stand"
                else:
                    cycle = (now - wait_time) / trot_period
                    phase_num = 1 if (cycle % 1.0) < 0.5 else 2
                    phase_str = f"Trot{phase_num}"

                print(f"t={now:5.1f}s | {phase_str:6s} | "
                      f"pos=({pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}) | "
                      f"force={max_force:6.1f}N ({force_pct:4.1f}%)")

        # Final summary
        print()
        print("=" * 70)
        print("Final Results:")
        print(f"  Position: ({d.qpos[0]:.3f}, {d.qpos[1]:.3f}, {d.qpos[2]:.3f})")
        print(f"  Height maintained: {d.qpos[2] > 0.8}")
        print(f"  Distance traveled: {np.sqrt(d.qpos[0]**2 + d.qpos[1]**2):.3f} m")
        print("=" * 70)

if __name__ == "__main__":
    main()
