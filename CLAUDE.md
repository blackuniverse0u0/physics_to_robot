# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Physics to Robot** is a research project building quadruped robot controllers from first principles, bridging rigid body dynamics, model-based control (MPC), and JAX-accelerated reinforcement learning. The goal is to achieve scalable perceptive locomotion with reliability and safety for real-world deployment.

The project emphasizes understanding underlying physics, kinematics, and classical control theories rather than treating RL as a "black box."

## Repository Structure

The repository consists of four main components:

### 1. **mujoco_playground/**
GPU-accelerated RL environments built with MuJoCo MJX (forked from google-deepmind/mujoco_playground)
- `mujoco_playground/_src/locomotion/` - Quadruped/bipedal locomotion environments (go1, berkeley_humanoid, etc.)
- `mujoco_playground/_src/manipulation/` - Manipulation environments (panda, leap_hand, etc.)
- `mujoco_playground/_src/dm_control_suite/` - Classic control environments
- `mujoco_playground/config/` - Environment-specific hyperparameters
- `learning/train_jax_ppo.py` - JAX-based PPO training script (Brax implementation)
- `learning/train_rsl_rl.py` - PyTorch-based PPO training with rsl_rl
- Package name: `playground` (not `mujoco_playground`)

### 2. **rsl_rl/**
Fast PyTorch implementation of PPO and student-teacher distillation (from ETH Zurich RSL)
- `rsl_rl/algorithms/` - PPO and other RL algorithms
- `rsl_rl/modules/` - Actor-critic networks, normalizers
- `rsl_rl/runners/` - OnPolicyRunner for training
- Package name: `rsl-rl-lib` on PyPI

### 3. **mujoco_mpc/**
Model Predictive Control framework from google-deepmind (C++ and Python)
- Contains convex MPC implementation for dynamic locomotion
- Python API available at `mujoco_mpc/python/`

### 4. **mujoco_physics/**
Custom physics implementations and lecture materials
- `lecture/` - Educational examples (forward/inverse kinematics, gaits)
- `froward_kinematics/`, `inverse_kinematics/` - Custom implementations

### 5. **rl_baseline/**
Custom RL implementations and experiments
- `ppo_mujoco.py` - PPO implementation for MuJoCo environments
- `ppo_mujoco.sh` - Training script
- `test.py` - Testing scripts

## Development Setup

### Initial Installation

```bash
# Create virtual environment
uv venv --python 3.11
source .venv/bin/activate

# Install JAX with CUDA 12 support
cd mujoco_playground
uv pip install -U "jax[cuda12]"

# Verify GPU backend (should print "gpu")
python -c "import jax; print(jax.default_backend())"

# Install playground with all dependencies
uv pip install -e ".[all]"
```

### Important Environment Variables

The training scripts set these automatically, but for custom scripts:
```bash
export JAX_DEFAULT_MATMUL_PRECISION=highest  # Critical for reproducibility on Ampere GPUs (RTX 30/40 series)
export XLA_FLAGS="--xla_gpu_triton_gemm_any=True"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
```

## Common Commands

### Training

**JAX-based PPO (MJX backend, GPU-accelerated):**
```bash
cd mujoco_playground
python learning/train_jax_ppo.py --env_name CartpoleBalance
python learning/train_jax_ppo.py --env_name PandaPickCube --use_wandb --num_timesteps 10_000_000
python learning/train_jax_ppo.py --env_name Go1FlatTerrain --impl warp  # Use Warp backend instead of MJX
```

**PyTorch-based PPO (rsl_rl, for locomotion/manipulation):**
```bash
cd mujoco_playground
python learning/train_rsl_rl.py --env_name BerkeleyHumanoidJoystickFlatTerrain --num_envs 4096
python learning/train_rsl_rl.py --env_name Go1FlatTerrain --use_wandb --multi_gpu
```

**Custom RL baseline:**
```bash
cd rl_baseline
bash ppo_mujoco.sh  # or run ppo_mujoco.py directly
```

### Training with Visualization (rscope)

```bash
pip install rscope
python learning/train_jax_ppo.py --env_name PandaPickCube --rscope_envs 16 --run_evals=False --deterministic_rscope=True

# In separate terminal:
python -m rscope
```

### Testing and Linting

**mujoco_playground:**
```bash
cd mujoco_playground

# Run tests
pytest mujoco_playground/_src/

# Linting and formatting
pre-commit run --all-files  # Runs ruff, pyink, mypy, etc.
ruff check .
pyink --check .
mypy .
```

**rsl_rl:**
```bash
cd rsl_rl

# Linting and formatting
pre-commit run --all-files
ruff check .
```

## Architecture and Design Patterns

### Environment Registration System

Environments are registered in `mujoco_playground/_src/registry.py` and organized by category:
- `registry.dm_control_suite._envs` - Classic control tasks
- `registry.locomotion._envs` - Locomotion tasks (quadrupeds, humanoids)
- `registry.manipulation._envs` - Manipulation tasks (arms, hands)

To get all available environments: `registry.ALL_ENVS`

### MJX Environment Architecture

All environments inherit from `MjxEnv` base class defined in `mujoco_playground/_src/mjx_env.py`:
- `reset()` - Initialize environment state
- `step()` - Step dynamics forward and compute rewards
- State is a PyTree containing: `pipeline_state`, `obs`, `reward`, `done`, `info`
- Environments are fully jittable and vmappable for massive parallelization

### Training Wrappers

**JAX training (train_jax_ppo.py):**
- Uses `wrapper.Wrapper` from `mujoco_playground._src.wrapper`
- Wraps MjxEnv to conform to Brax's training interface
- Supports both MJX and Warp backends via `impl` parameter

**PyTorch training (train_rsl_rl.py):**
- Uses `wrapper_torch.TorchWrapper` from `mujoco_playground._src.wrapper_torch`
- Converts JAX/MJX environments to PyTorch-compatible vectorized environments
- Integrates with rsl_rl's OnPolicyRunner

### Configuration System

Environment-specific hyperparameters are defined in:
- `mujoco_playground/config/dm_control_suite_params.py`
- `mujoco_playground/config/locomotion_params.py`
- `mujoco_playground/config/manipulation_params.py`

Each provides:
- `jax_config(env_name)` - Returns PPO hyperparameters for JAX training
- `rsl_rl_config(env_name)` - Returns config_dict for rsl_rl training

### RSL-RL Training Flow

1. Create environment using `wrapper_torch.TorchWrapper`
2. Get RL config using `locomotion_params.rsl_rl_config()` or `manipulation_params.rsl_rl_config()`
3. Create `OnPolicyRunner` with environment, config, and device
4. Call `runner.learn()` to train
5. Checkpoints saved to `./logs/{run_name}/`

## Key Technical Considerations

### GPU Precision and Reproducibility

On NVIDIA Ampere GPUs (RTX 30/40 series), JAX defaults to TF32 precision which can cause instability. Always set:
```bash
export JAX_DEFAULT_MATMUL_PRECISION=highest
```
Or add to `~/.bashrc` for persistence.

### Backend Selection (MJX vs Warp)

- **MJX (default)**: JAX-native implementation, better for smaller-scale experiments
- **Warp**: NVIDIA's parallel simulation framework, optimized for very large-scale training
- Switch using `--impl warp` flag

### Multi-GPU Training

For rsl_rl training with multiple GPUs:
```bash
python learning/train_rsl_rl.py --env_name <ENV> --multi_gpu --num_envs 8192
```

### Domain Randomization

Available for JAX training:
```bash
python learning/train_jax_ppo.py --env_name <ENV> --domain_randomization
```

## Project Phases and Implementation Status

Based on the README roadmap:

1. **Phase 1 (Physics & Kinematics)**: Implemented in `mujoco_physics/`
2. **Phase 2 (MPC)**: Framework available in `mujoco_mpc/`
3. **Phase 3 (State Estimation)**: Planned
4. **Phase 4 (RL with MJX)**: Core infrastructure complete in `mujoco_playground/`, active development area

## Important Notes

- **Package naming**: Install as `uv pip install -e ".[all]"` but import as `import mujoco_playground`
- **Pre-release dependencies**: mujoco_playground depends on pre-release versions of `mujoco` and `mujoco-mjx` from custom indices
- **Custom indices**: Configured in `pyproject.toml` under `[[tool.uv.index]]` (mujoco index, nvidia index)
- **Menagerie download**: First import of `mujoco_playground` automatically downloads MuJoCo Menagerie models

## External References

- MuJoCo MPC paper: "Predictive Sampling: Real-time Behaviour Synthesis with MuJoCo"
- RSL-RL paper: "RSL-RL: A Learning Library for Robotics Research" (arXiv:2509.10771)
- Playground paper: GPU-accelerated robot learning framework (2025)
- Theory references: Lynch & Park "Modern Robotics", Sutton & Barto "RL: An Introduction"
