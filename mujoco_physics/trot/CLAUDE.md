# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **trot gait controller implementation** for the Unitree A1 quadruped robot using MuJoCo physics simulation. The project implements a hierarchical control architecture that generates stable trot gaits through:
- Finite state machine (FSM) for gait coordination
- Cartesian trajectory generation for foot placement
- Analytical inverse kinematics for joint angles
- PD control with gravity compensation and virtual force control for stance legs

This is part of the larger `physics_to_robot` educational project focusing on understanding classical control theory and robot dynamics from first principles.

## Repository Structure

```
trot/
├── unitree_robotics_a1/    # MuJoCo model files (MJCF/XML)
│   ├── a1.xml              # Unitree A1 robot description
│   └── scene.xml           # Scene with robot, ground, lighting
└── a1_trot/                # Trot controller implementation
    ├── mj_a1_trot.py       # Main simulation entry point
    ├── globals.py          # Global state variables
    ├── parameters.py       # Robot and controller parameters
    ├── state_machine.py    # FSM for trot gait coordination
    ├── cartesian_traj.py   # Cartesian foot trajectory generation
    ├── joint_traj.py       # Joint space trajectory from Cartesian
    ├── joint_control.py    # Low-level PD control with force control
    ├── high_level_control.py  # Velocity command ramping
    ├── forward_kinematics_*.py  # Forward kinematics (leg/robot)
    ├── inverse_kinematics_analytic.py  # Analytical IK solver
    ├── jac_end_effector_leg.py  # Jacobian computation
    ├── stance_force.py     # Virtual force control for stance legs
    ├── quintic_poly.py     # Quintic polynomial interpolation
    ├── set_command_step.py  # Command rate limiting
    ├── utility.py          # Rotation utilities (quaternion, matrix, etc.)
    └── robot_data.py       # Robot kinematic/dynamic data
```

## Running the Simulation

**Main entry point:**
```bash
cd /Users/joonhyunshin/Physics/physics_to_robot/mujoco_physics/trot/a1_trot
python mj_a1_trot.py
```

**Requirements:**
- Python 3.x
- MuJoCo (`import mujoco`)
- NumPy
- GLFW for visualization

**Simulation controls:**
- **Backspace**: Reset simulation
- **Mouse left drag**: Rotate camera
- **Mouse right drag**: Move camera
- **Mouse scroll**: Zoom camera
- **Shift + mouse**: Alternative camera controls

## Control Architecture

The controller implements a hierarchical structure executed at 1000 Hz (MuJoCo default timestep = 0.001s):

### 1. High-Level Control (`high_level_control.py`)
- Updates desired velocities (`xdot_ref`, `ydot_ref`, `psidot_ref`) every step
- Uses rate limiting to smoothly ramp to target velocities
- Currently hardcoded: forward velocity = 1.0 m/s, yaw rate = 1.0 rad/s

### 2. State Machine (`state_machine.py`)
- Manages gait coordination using per-leg FSM states:
  - **fsm_stand (1)**: Initial standing state
  - **fsm_stance (2)**: Foot on ground, supporting body
  - **fsm_swing (3)**: Foot in air, repositioning
- **Trot gait pattern**: Diagonal leg pairs move together
  - Pair 1: FL (leg 0) + RR (leg 3)
  - Pair 2: FR (leg 1) + RL (leg 2)
- Timing parameters:
  - `t_stand = 0.1s`: Initial standing duration
  - `t_step = 0.15s`: Duration of swing/stance phase

### 3. Cartesian Trajectory Generation (`cartesian_traj.py`)
Generates foot trajectories in leg frame using quintic polynomials:
- **Stance phase**: Foot moves backward relative to body (lx, ly sweep)
- **Swing phase**: Parabolic lift-and-place trajectory
  - Horizontal (lx, ly): Linear interpolation based on velocity command
  - Vertical (lz): Two-phase quintic (lift up, then down)
  - Clearance height: `hcl = 0.075m`

### 4. Joint Trajectory Generation (`joint_traj.py`)
Converts Cartesian trajectories to joint space:
- Uses analytical inverse kinematics for joint angles
- Computes joint velocities via Jacobian inverse: `u_leg = J_inv @ Xdot_ref`

### 5. Joint Control (`joint_control.py`)
State-dependent torque commands:
- **Stand/Swing**: Simple PD control
  - `trq = gain * (-kp*(q_act - q_ref) - kd*(u_act - u_ref))`
  - `gain=10, kp=10, kd=1`
- **Stance**: PD + gravity compensation via virtual force control
  - Computes virtual ground reaction forces using `stance_force()`
  - Maps forces to joint torques: `trq_grav = -J.T @ F`
  - Total torque: `trq = trq_grav + PD_torque`

### 6. Virtual Force Control (`stance_force.py`)
For stance legs, solves for ground reaction forces that stabilize the body:
- Formulates `A @ F = b` where:
  - `A`: Maps leg forces to body wrench (force + moment)
  - `b`: Desired body wrench (position/orientation control + gravity compensation)
- Control objectives:
  - Height stabilization: `z_ref = -lz0 = 0.249m`
  - Velocity tracking: `xdot_ref, ydot_ref, psidot_ref`
  - Attitude stabilization: Roll/pitch to zero
- Uses pseudoinverse to distribute forces across stance legs

## Key Parameters (`parameters.py`)

**Robot geometry:**
- `lz0 = -0.249m`: Nominal foot position (height in leg frame)
- Thigh/shank length: `L = 0.2m` (defined in kinematics files)
- Hip offset: `w = ±0.08505m` (lateral offset)
- Body half-length: `c = 0.183m` (for yaw moment calculation)

**Gait timing:**
- `t_stand = 0.1s`: Initial standing time
- `t_step = 0.15s`: Swing/stance duration
- `hcl = 0.075m`: Foot clearance height during swing

**Velocity limits:**
- Forward: `[-2.0, 2.0] m/s`, rate: `0.1 m/s²`
- Lateral: `[-1.0, 1.0] m/s`, rate: `0.05 m/s²`
- Yaw: `[-2.0, 2.0] rad/s`, rate: `0.1 rad/s²`

**Robot properties:**
- `mass = 12.453 kg`
- `gravity = 9.81 m/s²`

## Global State (`globals.py`)

All modules share state via `globals.py` (initialized by `globals.init()`):

**FSM state:**
- `fsm[4]`: Current state for each leg (1=stand, 2=stance, 3=swing)
- `t_fsm[4]`: Timestamp when current state started

**Trajectory references:**
- `lx_ref, ly_ref, lz_ref[4]`: Cartesian foot positions (leg frame)
- `lxdot_ref, lydot_ref, lzdot_ref[4]`: Cartesian velocities
- `q_ref[12]`: Joint angle references (3 per leg)
- `u_ref[12]`: Joint velocity references

**Sensor data:**
- `q_act[12]`: Actual joint angles from MuJoCo
- `u_act[12]`: Actual joint velocities
- `pos_quat_trunk[7]`: Base position + quaternion [x,y,z,q0,qx,qy,qz]
- `vel_angvel_trunk[6]`: Base linear/angular velocity [vx,vy,vz,ωx,ωy,ωz]

**Commands:**
- `xdot_ref, ydot_ref, psidot_ref`: Desired body velocities
- `trq[12]`: Computed joint torques (sent to MuJoCo)

## Leg Numbering Convention

```
    0 (FR) --- 1 (FL)
      |         |
      |  BODY   |
      |         |
    2 (RR) --- 3 (RL)
```
- Leg 0: Front Right (FR)
- Leg 1: Front Left (FL)
- Leg 2: Rear Right (RR)
- Leg 3: Rear Left (RL)

Diagonal pairs for trot:
- **Pair A**: Legs 0 + 3 (FR + RL)
- **Pair B**: Legs 1 + 2 (FL + RR)

## Kinematics

**Forward Kinematics** (`forward_kinematics_leg.py`):
- 3-DOF leg: hip abduction, hip pitch, knee pitch
- Uses homogeneous transforms (DH-like convention)
- Returns `end_eff_pos` (foot position) and transformation matrices `H01, H02, H03`

**Inverse Kinematics** (`inverse_kinematics_analytic.py`):
- Closed-form solution for 3R leg
- Input: Cartesian position `[lx, ly, lz]` in leg frame
- Output: Joint angles `[q_abduction, q_hip, q_knee]`

**Jacobian** (`jac_end_effector_leg.py`):
- 3x3 Jacobian mapping joint velocities to foot velocities
- Used for velocity control and force mapping

## Modifying Controller Behavior

**Change desired velocity** (`high_level_control.py:11-20`):
```python
vx_ = 1.0  # Forward velocity (m/s)
vy_ = 0.0  # Lateral velocity (m/s)
omega_ = 1.0  # Yaw rate (rad/s)
```

**Change gait parameters** (`parameters.py`):
- Adjust `t_step` for faster/slower gait
- Adjust `hcl` for higher/lower foot clearance

**Switch between kinematic and dynamic mode** (`mj_a1_trot.py:15`):
- `flag_trajectory_generation = 0`: Dynamic simulation (with physics)
- `flag_trajectory_generation = 1`: Kinematic mode (just visualize trajectories)

**Control gains** (`joint_control.py`):
- PD gains: `kp=10, kd=1, gain=10`
- Stance force control gains: See `stance_force.py:64-69`

## Utility Functions (`utility.py`)

Provides rotation and spatial math utilities:
- `quat2mat()`, `mat2quat()`: Quaternion ↔ rotation matrix
- `quat2bryant()`, `bryant2quat()`: Quaternion ↔ Euler angles (ZYX)
- `rotation(angle, axis)`: Elementary rotation matrix
- `vec2skew()`: Vector to skew-symmetric matrix (for cross products)

## Important Notes

- **Global state pattern**: All modules communicate via `globals.py`. When modifying state, ensure consistency across modules.
- **Coordinate frames**:
  - Leg frame origin at hip, z-axis points down
  - Body frame: x-forward, y-left, z-up
- **Simulation timestep**: MuJoCo default `dt=0.001s`, control loop runs at 1000 Hz
- **Camera tracking**: Camera automatically follows robot position (`cam.lookat[0:2] = data.qpos[0:2]`)
- **Visualization rate**: Rendering limited to 15 FPS (line 180), simulation runs at full speed
