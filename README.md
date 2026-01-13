# AC-DMPC: Actor-Critic Differentiable Model Predictive Control

A framework combining deep reinforcement learning with differentiable Model Predictive Control (MPC) for robot control tasks.

## Overview

This codebase implements **Actor-Critic MPC (AC-DMPC)**, which uses:
- **Transformer-XL or MLP** networks for feature extraction (configurable)
- **Differentiable MPC** controllers for optimal control
- **PPO (Proximal Policy Optimization)** for policy learning

The key innovation is using neural networks to learn MPC cost function parameters (Economic MPC) rather than directly outputting actions.

**Backbone Options:**
- **Transformer-XL**: For sequential modeling with long-term dependencies
- **MLP**: Simple 2-layer 512-ReLU networks (as described in the paper) for faster training

## Installation

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install torch numpy scipy gymnasium psutil matplotlib pyyaml
```

## Quick Start

### 1. Train on Double Integrator Waypoint Task (Transformer-XL)

```bash
python examples/double_integrator_AC_DRL.py \
  --config configs/double_integrator_waypoint.yaml \
  --total-iters 50 \
  --num-envs 8 \
  --device cuda
```

### 1b. Train on Double Integrator Waypoint Task (MLP Backbone)

For faster training with MLP networks (2-layer 512-ReLU as in the paper):

```bash
python examples/double_integrator_AC_DRL.py \
  --config configs/double_integrator_waypoint_mlp.yaml \
  --total-iters 50 \
  --num-envs 8 \
  --device cuda
```

### 1c. Train with Real-time Visualization

Enable real-time visualization during training to see the agent's behavior:

```bash
python examples/double_integrator_AC_DRL.py \
  --config configs/double_integrator_waypoint_mlp.yaml \
  --total-iters 50 \
  --num-envs 8 \
  --device cuda \
  --visualize \
  --visualize-env-id 0
```

The `--visualize` flag opens a matplotlib window showing the training environment in real-time. Use `--visualize-env-id` to specify which environment to visualize (default: 0).

### 1d. Evaluate Trained Model

Evaluate a trained checkpoint without training:

```bash
python examples/double_integrator_AC_DRL.py \
  --config configs/double_integrator_waypoint_mlp.yaml \
  --eval-only \
  --eval-runs 20 \
  --eval-output-dir eval_outputs/double_integrator \
  --device cuda
```

**Evaluation Parameters:**
- `--eval-only`: Skip training and only run evaluation on available checkpoints
- `--eval-runs`: Number of evaluation episodes to run (default: 20)
- `--eval-output-dir`: Directory for evaluation plots and summaries
- `--eval-max-steps`: Maximum steps per evaluation episode (defaults to episode length)
- `--eval-seed`: Random seed for evaluation (default: training_seed + 1337)
- `--resume`: Checkpoint selection strategy - `auto`, `best`, `latest`, or `none` (default: auto)

The evaluation will automatically find the best or latest checkpoint from the checkpoint directory and generate plots showing the agent's performance.

### 2. Train on SE(2) Kinematic Waypoint Task

```bash
python examples/se2_kinematic_AC_DRL.py \
  --config configs/se2_kinematic_fixed.yaml \
  --total-iters 100 \
  --num-envs 8 \
  --device cuda
```

### 3. Train with Obstacles and Lidar

```bash
python examples/se2_kinematic_obstacles_AC_DRL.py \
  --config configs/se2_kinematic_obstacles_expanded.yaml \
  --total-iters 100 \
  --num-envs 8 \
  --device cuda
```

## Project Structure

```
acdmpc_clean/
├── ACMPC/              # Core Actor-Critic MPC module
│   ├── models/         # Neural networks (Actor, Critic)
│   ├── mpc/            # MPC controllers (Economic MPC)
│   ├── training/       # Training loop (PPO, GAE)
│   ├── sampling/       # Data collection (RolloutCollector)
│   ├── envs/           # Environment definitions
│   └── agent.py        # High-level Agent interface
├── DifferentialMPC/    # Differentiable MPC implementation
├── examples/           # Example training scripts
├── configs/            # Configuration files (YAML)
└── docs/               # Documentation
```

## Key Components

### Actor Networks
- **TransformerActor**: Uses Transformer-XL to process state history with long-term memory
- **MLPActor**: Uses simple 2-layer 512-ReLU MLP for faster training (as in paper)
- Both predict MPC cost parameters from latent space
- Economic MPC head generates optimal control actions

### Critic Networks
- **TransformerCritic**: Uses Transformer-XL to estimate state values
- **MLPCritic**: Uses 2-layer 512-ReLU MLP to estimate state values
- Both provide value function for PPO training

### MPC Controller
- Differentiable MPC solver (iLQR-based)
- Supports state/action constraints
- Gradient computation for end-to-end training

## Configuration

Configuration files are in YAML format. See `docs/config_quickstart.md` for details.

### Transformer-XL Configuration Example:
```yaml
seed: 7
device: cuda

model:
  actor:
    backbone_type: transformer  # or "mlp"
    transformer:
      d_model: 128
      n_layers: 3
      n_heads: 4
    mpc:
      horizon: 5
      state_dim: 4
      action_dim: 2
      latent_dim: 128  # Must match transformer.d_model

training:
  ppo_epochs: 4
  clip_param: 0.2
  value_loss_coeff: 0.5
  entropy_coeff: 0.01
```

### MLP Configuration Example:
```yaml
seed: 7
device: cuda

model:
  actor:
    backbone_type: mlp  # Use MLP instead of Transformer
    mlp:
      hidden_dim: 512
      output_dim: 512
      num_layers: 2
      activation: relu
      dropout: 0.0
    mpc:
      horizon: 5
      state_dim: 4
      action_dim: 2
      latent_dim: 512  # Must match mlp.output_dim
    cost_map:
      hidden_dim: 512
      num_layers: 2  # 2-layer 512-ReLU + sigmoid output (as in paper)
  
  critic:
    backbone_type: mlp
    mlp:
      hidden_dim: 512
      output_dim: 512
      num_layers: 2  # 2-layer 512-ReLU (as in paper)
      activation: relu

training:
  ppo_epochs: 4
  clip_param: 0.2
  value_loss_coeff: 0.5
  entropy_coeff: 0.01
```

## Environments

- **Double Integrator Waypoint**: 2D point-mass robot tracking waypoints
- **SE(2) Kinematic Waypoint**: Planar robot with orientation tracking waypoints
- **SE(2) with Obstacles**: Same as above with obstacles and lidar sensing

## Citation

If you use this code, please cite the original paper:
- Transformer-based Actor-Critic with Differentiable MPC

## License

See LICENSE file for details.

