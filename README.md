

# Physics_to_Robot 🦾

**From First Principles to Agile Locomotion.**

> Model-based Optimization for Control 

> Model-free OnPolicy RL 


![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![MuJoCo](https://img.shields.io/badge/Sim-MuJoCo-orange?logo=mujoco)
![JAX](https://img.shields.io/badge/Library-JAX%2FMJX-green?logo=google-jax)
![Control](https://img.shields.io/badge/Method-MPC_%26_RL-purple)

**From First Principles to Agile Locomotion.**

Building a quadruped robot controller from scratch: Bridging Rigid Body Dynamics, Model-Based Control (MPC), and JAX-accelerated Reinforcement Learning

Instead of treating RL as a "black box," this project emphasizes understanding the underlying physics, kinematics, and classical control theories to build robust and explainable locomotion policies.

**Our Goal: Scalable Perceptive Locomotion with Reliability & Safety for Real World .**

##  Roadmap & Milestones



### Phase 1: Modern Robotics & Physics & Mathematics (Basics)

**Rigid-body motions**
- SO(3), SE(3), so(3), se(3), D.H params, PoE(product of Exponentials), Lie Group, Lie Algebra 

**Forward kinematics**

**Velocity kinematics and statics**

**Inverse kinematics**

**Robot Control**

<!-- *Goal: Understand how rigid bodies move and how to control limbs using geometry.*
* **Simulation Setup**: MuJoCo with Python bindings.
* **Kinematics**: Implementing Forward Kinematics (FK) and Inverse Kinematics (IK) using analytical and numerical (Jacobian) methods.
* **Dynamics**: Understanding Inertia, Coriolis, and Gravity matrices using *Modern Robotics* theory.
* **Trajectory Planning**  -->

Robotics 3D : https://pab47.github.io/robotics/robotics25.html


### Phase 2: Model-Based Optimal Control (MPC)
<!-- *Goal: Implement the Quadruped for dynamic locomotion.*
* **Reference**: Based on the seminal paper *"Dynamic Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive Control"*.
* **Convex MPC**: Formulating the locomotion problem as a QP (Quadratic Programming) to solve for Ground Reaction Forces (GRF).
* **Whole Body Control (WBC)**: Mapping optimal forces to joint torques using Jacobian transpose/pseudo-inverse.
* **Validation**: Getting the robot to Trot/Gallop in MuJoCo using pure math-based control. -->

<!-- https://underactuated.mit.edu/ -->

### Phase 3: State Estimation
<!-- *Goal: Closing the Sim-to-Real gap by dealing with noisy sensors.*
* **Sensor Simulation**: Injecting realistic Gaussian noise into IMU and encoder data in MuJoCo.
* **Filter Implementation**: Implementing Linear Kalman Filters (LKF) to fuse kinematics and IMU data for estimating body velocity and position.
* **Application**: Feeding estimated states into the MPC controller instead of ground truth. -->

### Phase 4: Reinforcement Learning with MJX
<!-- *Goal: Achieving robust and agile behaviors using massive parallel simulation.*
* **Tech Stack**: JAX + MuJoCo MJX + rsl_rl.
* **Environment**: Porting the MuJoCo environment to JAX for GPU-accelerated training.
* **Policy Learning**: Training PPO agents with a curriculum.
* **Guided Learning**: Using the Phase 2 MPC controller as a reference (Teacher) to warm-start or guide the RL agent. -->


PPO details : https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/



## Tech Stack

* **Simulation**: [MuJoCo](https://mujoco.org/)
* **Language**: Python 
* **Math & Physics**: NumPy, SciPy, ... etc
* **Optimization**: OSQP, CVXPY (for MPC)
* **Reinforcement Learning**: Pytorch, JAX, Brax, MuJoCo MJX, [rsl_rl](https://github.com/leggedrobotics/rsl_rl)

## References

1.  **Theory**: *Introduction to Mathematical Statistics Edition* (Robert V. Hogg , Joeseph McKean , Allen T. Craig), *Modern Robotics: Mechanics, Planning, and Control* (Lynch & Park), *Reinforcement Learning: An Introduction* (Sutton, Richard S. , Barto, Andrew G.),

2.  **MPC** : Dynamic Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive Control
3.  **State Estimation** : 
4.  **Reference Code**: [google-deepmind/mujoco_mpc](https://github.com/google-deepmind/mujoco_mpc) & [leggedrobotics/rsl_rl](https://github.com/leggedrobotics/rsl_rl)

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/physics_to_robot.git
cd physics_to_robot

# Create a virtual environment
uv venv --python 3.11 
source venv/bin/activate  # Windows: venv\Scripts\activate

cd mujoco_playground

# Install CUDA 12 jax
uv pip install -U "jax[cuda12]"

#Verify GPU backend: python -c "import jax; print(jax.default_backend())" should print gpu

uv pip install -e ".[all]"
# Verify installation (and download Menagerie): python -c "import mujoco_playground"
```

<!-- # Future Work 
https://xbpeng.github.io/projects/DeepMimic/index.html

https://github.com/hongsukchoi/VideoMimic -->



## ⚖️ License & Acknowledgments

This project is released under the **MIT License**. However, it leverages and modifies code from several open-source projects. We gratefully acknowledge their contributions.

### 1. MuJoCo MPC & Playground
Parts of the model-based control and simulation environment are adapted from **Google DeepMind**.
* **Source**: [google-deepmind/mujoco_mpc](https://github.com/google-deepmind/mujoco_mpc), [google-deepmind/mujoco_playground](https://github.com/google-deepmind/mujoco_playground)
* **License**: Apache License 2.0
* **Copyright**: 2022 DeepMind Technologies Limited.
<!-- * **Note**: Texture assets from *Polyhaven* are licensed under CC0. -->

### 2. rsl_rl
The reinforcement learning implementation (PPO) is based on **rsl_rl** by ETH Zurich RSL.
* **Source**: [leggedrobotics/rsl_rl](https://github.com/leggedrobotics/rsl_rl)
* **Citation**:
```bibtex
@article{schwarke2025rslrl,
  title={RSL-RL: A Learning Library for Robotics Research},
  author={Schwarke, Clemens and Mittal, Mayank and Rudin, Nikita and Hoeller, David and Hutter, Marco},
  journal={arXiv preprint arXiv:2509.10771},
  year={2025}
}

