"""3-D quadrotor waypoint environment with RK4 integration.

State  (nx=15): [p(3), v(3), R_flat(9)]  –  position, velocity, flattened 3x3 rotation matrix
Action (nu=4):  [thrust, ωx, ωy, ωz]     –  collective thrust + body angular rates

Dynamics (continuous-time):
    ṗ = v
    v̇ = g + R @ [0, 0, thrust/mass]
    Ṙ = R @ skew(ω)

Integrated with a 4th-order Runge-Kutta (RK4) scheme.

Reference: rpg_flightning/flightning/objects/quadrotor_simple_obj.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class QuadrotorWaypointEnvConfig:
    """Hyper-parameters for the 3-D quadrotor waypoint environment."""
    ## No effect
    dt: float = 0.02
    episode_len: int = 500
    waypoint_range: float = 2.0
    goal_radius: float = 0.3
    min_start_radius: float = 0.3
    mass: float = 0.752
    gravity: float = 9.81
    max_thrust: float = 20.0        # N  (roughly 2.7 * m * g)
    max_body_rate: float = 6.0      # rad/s
    progress_gain: float = 10.0
    action_penalty: float = 0.01
    living_penalty: float = 0.01
    goal_bonus: float = 10.0
    control_gain: float = 0.0
    max_distance: float = 5.0       # terminate if dist to waypoint exceeds this
    boundary_penalty: float = 10.0  # penalty applied on termination


# ---------------------------------------------------------------------------
# NumPy helpers (used inside the Gym environment step)
# ---------------------------------------------------------------------------

def _skew_np(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix from a 3-vector."""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=v.dtype)


def _reorthogonalize_np(R: np.ndarray) -> np.ndarray:
    """Project a 3x3 matrix back to SO(3) via SVD."""
    U, _, Vt = np.linalg.svd(R)
    det = np.linalg.det(U @ Vt)
    Vt[-1] *= np.sign(det)
    return U @ Vt


def _quadrotor_continuous_np(
    state: np.ndarray,
    action: np.ndarray,
    mass: float,
    gravity: float,
) -> np.ndarray:
    """Continuous-time derivative of the quadrotor state (NumPy)."""
    p = state[:3]
    v = state[3:6]
    R = state[6:15].reshape(3, 3)

    thrust = action[0]
    omega = action[1:4]

    p_dot = v
    thrust_body = np.array([0.0, 0.0, thrust / mass], dtype=state.dtype)
    v_dot = np.array([0.0, 0.0, -gravity], dtype=state.dtype) + R @ thrust_body
    R_dot = R @ _skew_np(omega)

    return np.concatenate([p_dot, v_dot, R_dot.ravel()])


def _rk4_step_np(
    state: np.ndarray,
    action: np.ndarray,
    dt: float,
    mass: float,
    gravity: float,
) -> np.ndarray:
    """Single RK4 integration step (NumPy)."""
    f = lambda s: _quadrotor_continuous_np(s, action, mass, gravity)
    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)
    next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    # Re-orthogonalize rotation
    R = next_state[6:15].reshape(3, 3)
    next_state[6:15] = _reorthogonalize_np(R).ravel()
    return next_state


# ---------------------------------------------------------------------------
# Gymnasium environment
# ---------------------------------------------------------------------------

class QuadrotorWaypointEnv(gym.Env):
    """3-D quadrotor chasing randomly sampled waypoints."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        *,
        dt: float = 0.02,
        episode_len: int = 500,
        waypoint_range: float = 2.0,
        goal_radius: float = 0.3,
        min_start_radius: float = 0.3,
        mass: float = 0.752,
        gravity: float = 9.81,
        max_thrust: float = 20.0,
        max_body_rate: float = 6.0,
        progress_gain: float = 10.0,
        action_penalty: float = 0.01,
        living_penalty: float = 0.01,
        goal_bonus: float = 10.0,
        control_gain: float = 0.0,
        max_distance: float = 5.0,
        boundary_penalty: float = 10.0,
        env_id: int = 0,
    ):
        super().__init__()
        self.dt = dt
        self.episode_len = episode_len
        self.waypoint_range = waypoint_range
        self.goal_radius = goal_radius
        self.min_start_radius = min_start_radius
        self.mass = mass
        self.gravity = gravity
        self.max_thrust = max_thrust
        self.max_body_rate = max_body_rate
        self.progress_gain = progress_gain
        self.action_penalty = action_penalty
        self.living_penalty = living_penalty
        self.goal_bonus = goal_bonus
        self.control_gain = control_gain
        self.max_distance = max_distance
        self.boundary_penalty = boundary_penalty
        self.env_id = env_id

        self.nx = 15  # p(3) + v(3) + R_flat(9)
        self.nu = 4   # thrust + ω(3)

        obs_dim = self.nx + 3  # state + 3-D waypoint
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )
        act_high = np.array(
            [max_thrust, max_body_rate, max_body_rate, max_body_rate],
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(low=-act_high, high=act_high, dtype=np.float32)

        # Internal state
        self.state: np.ndarray = np.zeros(self.nx, dtype=np.float32)
        self.target_waypoint: np.ndarray = np.zeros(3, dtype=np.float32)
        self.prev_distance: float = 0.0
        self.steps: int = 0
        self.waypoints_reached: int = 0
        self._rng = np.random.default_rng()
        self._last_thrust: float = 0.0

    # ---- reset / step ----

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Start near hover: identity rotation, randomised position, zero velocity
        start_range = min(self.waypoint_range * 0.5, 1.0)
        p0 = self._rng.uniform(-start_range, start_range, size=3).astype(np.float32)
        v0 = np.zeros(3, dtype=np.float32)
        R0 = np.eye(3, dtype=np.float32).ravel()
        self.state = np.concatenate([p0, v0, R0])

        self.steps = 0
        self.waypoints_reached = 0
        self.target_waypoint = self._sample_waypoint()
        self.prev_distance = float(np.linalg.norm(self.state[:3] - self.target_waypoint))

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray):
        self.steps += 1
        action = np.asarray(action, dtype=np.float32)
        # Clamp thrust to [0, max_thrust] and body rates to [-max, max]
        clipped = action.copy()
        clipped[0] = np.clip(clipped[0], 0.0, self.max_thrust)
        clipped[1:] = np.clip(clipped[1:], -self.max_body_rate, self.max_body_rate)
        self._last_thrust = float(clipped[0])

        # RK4 integration
        next_state = _rk4_step_np(self.state, clipped, self.dt, self.mass, self.gravity)
        next_state = next_state.astype(np.float32)

        # Reward
        pos = next_state[:3]
        distance = float(np.linalg.norm(pos - self.target_waypoint))
        progress = self.prev_distance - distance

        reward = self.progress_gain * progress
        reward -= self.living_penalty
        control_mag = float(np.linalg.norm(clipped))
        if self.action_penalty > 0.0:
            reward -= self.action_penalty * control_mag
        elif self.control_gain > 0.0:
            reward += self.control_gain * control_mag

        waypoint_reached = distance <= self.goal_radius
        if waypoint_reached:
            reward += self.goal_bonus
            self.waypoints_reached += 1
            self.target_waypoint = self._sample_waypoint()
            distance = float(np.linalg.norm(pos - self.target_waypoint))

        self.prev_distance = distance
        self.state = next_state

        truncated = self.steps >= self.episode_len
        terminated = distance > self.max_distance
        if terminated:
            reward -= self.boundary_penalty

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    # ---- helpers ----

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self.state, self.target_waypoint]).astype(np.float32)

    def _get_info(self) -> Dict:
        return {
            "target_waypoint": self.target_waypoint.copy(),
            "waypoints_reached": self.waypoints_reached,
            "distance": float(np.linalg.norm(self.state[:3] - self.target_waypoint)),
        }

    def _sample_waypoint(self) -> np.ndarray:
        # Sample uniformly from the full waypoint range for all waypoints.
        # Reject if the waypoint falls within goal_radius of the drone's current
        # position (would be immediately "reached" before the agent acts).
        for _ in range(20):
            wp = self._rng.uniform(
                -self.waypoint_range, self.waypoint_range, size=3,
            ).astype(np.float32)
            if np.linalg.norm(wp - self.state[:3]) > self.goal_radius:
                return wp
        return wp

    # ---- rendering ----

    def render(self, mode: str = "human"):
        """Render the 3-D quadrotor environment with matplotlib."""
        if mode not in self.metadata["render_modes"]:
            return None

        try:
            import matplotlib
            if mode == "human":
                try:
                    matplotlib.use("TkAgg")
                except Exception:
                    try:
                        matplotlib.use("Qt5Agg")
                    except Exception:
                        pass
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: F401
        except ImportError:
            return None

        # --- first call: create figure ---
        if not hasattr(self, "_fig") or self._fig is None:
            self._fig = plt.figure(figsize=(9, 9))
            self._ax = self._fig.add_subplot(111, projection="3d")
            lim = 2
            self._ax.set_xlim(-lim, lim)
            self._ax.set_ylim(-lim, lim)
            self._ax.set_zlim(-lim, lim)
            self._ax.set_xlabel("X")
            self._ax.set_ylabel("Y")
            self._ax.set_zlabel("Z")
            self._ax.set_box_aspect([1, 1, 1])
            self._trajectory: List[np.ndarray] = []
            plt.ion()
            plt.show(block=False)

        # record trajectory
        pos = self.state[:3].copy()
        if len(self._trajectory) == 0 or not np.allclose(self._trajectory[-1], pos):
            self._trajectory.append(pos)

        # --- redraw ---
        self._ax.cla()
        lim = 2
        self._ax.set_xlim(-lim, lim)
        self._ax.set_ylim(-lim, lim)
        self._ax.set_zlim(-lim, lim)
        self._ax.set_xlabel("X")
        self._ax.set_ylabel("Y")
        self._ax.set_zlabel("Z")
        self._ax.set_title(
            f"Quadrotor (Env {self.env_id}) | "
            f"Step {self.steps} | WP {self.waypoints_reached}"
        )

        # trajectory
        if len(self._trajectory) > 1:
            traj = np.array(self._trajectory)
            self._ax.plot(
                traj[:, 0], traj[:, 1], traj[:, 2],
                "b-", alpha=0.4, linewidth=1, label="Path",
            )

        # --- X-configuration quadrotor ---
        R = self.state[6:15].reshape(3, 3)
        arm_len = 0.18   # visual arm length (m)
        rotor_r = 0.05   # rotor disk radius (m)
        sq2 = 1.0 / np.sqrt(2.0)
        # 4 arm directions in body frame (X config: ±45° in body XY plane)
        arm_dirs_body = np.array([
            [ sq2,  sq2, 0.0],   # front-right (CW)
            [-sq2,  sq2, 0.0],   # front-left  (CCW)
            [-sq2, -sq2, 0.0],   # rear-left   (CW)
            [ sq2, -sq2, 0.0],   # rear-right  (CCW)
        ])
        motor_colors = ["#FF3333", "#33CC33", "#FF3333", "#33CC33"]
        bx, by = R[:, 0], R[:, 1]   # body X and Y axes in world frame

        for arm_b, mc in zip(arm_dirs_body, motor_colors):
            tip_w = pos + arm_len * (R @ arm_b)
            # Arm
            self._ax.plot(
                [pos[0], tip_w[0]], [pos[1], tip_w[1]], [pos[2], tip_w[2]],
                color="k", linewidth=2.5,
            )
            # Rotor disk (circle in body XY plane centred at tip)
            theta = np.linspace(0, 2 * np.pi, 20)
            c_pts = tip_w[:, None] + rotor_r * (
                np.outer(bx, np.cos(theta)) + np.outer(by, np.sin(theta))
            )
            self._ax.plot(c_pts[0], c_pts[1], c_pts[2], color=mc, linewidth=1.5)

        # Central hub
        self._ax.scatter(*pos, c="black", s=12, zorder=7, label="Quad")

        # thrust arrow: along body z-axis (R[:,2]), scaled by thrust / max_thrust
        thrust_scale = (self._last_thrust / self.max_thrust) * 0.6  # max visual length 0.6 m
        thrust_dir = R[:, 2] * thrust_scale
        self._ax.quiver(
            pos[0], pos[1], pos[2],
            thrust_dir[0], thrust_dir[1], thrust_dir[2],
            color="yellow", linewidth=2.0, arrow_length_ratio=0.2, label=f"T={self._last_thrust:.1f}N",
        )

        # velocity arrow
        vel = self.state[3:6]
        vel_norm = np.linalg.norm(vel)
        if vel_norm > 0.05:
            scale = min(0.3, vel_norm * 0.1)
            v_dir = vel / vel_norm * scale
            self._ax.quiver(
                pos[0], pos[1], pos[2],
                v_dir[0], v_dir[1], v_dir[2],
                color="cyan", linewidth=1.5, arrow_length_ratio=0.25, label="Vel",
            )

        # waypoint + goal sphere
        wp = self.target_waypoint
        self._ax.scatter(*wp, c="red", s=80, marker="*", zorder=5, label="Target")
        # wireframe sphere for goal radius
        u_sp = np.linspace(0, 2 * np.pi, 16)
        v_sp = np.linspace(0, np.pi, 12)
        xs = wp[0] + self.goal_radius * np.outer(np.cos(u_sp), np.sin(v_sp))
        ys = wp[1] + self.goal_radius * np.outer(np.sin(u_sp), np.sin(v_sp))
        zs = wp[2] + self.goal_radius * np.outer(np.ones_like(u_sp), np.cos(v_sp))
        self._ax.plot_wireframe(xs, ys, zs, color="orange", alpha=0.15, linewidth=0.4)

        # dashed line from quad to waypoint
        self._ax.plot(
            [pos[0], wp[0]], [pos[1], wp[1]], [pos[2], wp[2]],
            "r--", alpha=0.3, linewidth=1,
        )

        self._ax.legend(loc="upper right", fontsize=8)

        if mode == "human":
            plt.draw()
            plt.pause(0.01)
            return None
        elif mode == "rgb_array":
            self._fig.canvas.draw()
            buf = np.asarray(self._fig.canvas.buffer_rgba(), dtype=np.uint8)
            # RGBA -> RGB
            return buf[:, :, :3].copy()
        return None

    def close(self):
        """Close rendering resources."""
        if hasattr(self, "_fig") and self._fig is not None:
            try:
                import matplotlib.pyplot as plt
                plt.close(self._fig.number if hasattr(self._fig, "number") else self._fig)
            except Exception:
                pass
            self._fig = None
            self._ax = None


# ---------------------------------------------------------------------------
# PyTorch dynamics for MPC  (RK4)
# ---------------------------------------------------------------------------

def _skew_torch(v: Tensor) -> Tensor:
    """Batched skew-symmetric matrix: v [*, 3] -> [*, 3, 3]."""
    z = torch.zeros_like(v[..., 0])
    return torch.stack([
        torch.stack([z, -v[..., 2], v[..., 1]], dim=-1),
        torch.stack([v[..., 2], z, -v[..., 0]], dim=-1),
        torch.stack([-v[..., 1], v[..., 0], z], dim=-1),
    ], dim=-2)


# def _reorthogonalize_torch(R_flat: Tensor) -> Tensor:
#     """Project batched flattened rotation matrices back to SO(3).

#     Uses Gram-Schmidt orthonormalization which is fully differentiable
#     and vmap-compatible (no SVD, avoids NaN gradients from degenerate
#     singular values).

#     Input:  R_flat  [*, 9]
#     Output: R_flat  [*, 9]
#     """
#     shape = R_flat.shape[:-1]
#     R = R_flat.reshape(*shape, 3, 3)

#     # Gram-Schmidt on columns
#     c0 = R[..., :, 0]  # [*, 3]
#     c1 = R[..., :, 1]
#     # Normalize first column
#     e0 = c0 / (torch.linalg.norm(c0, dim=-1, keepdim=True) + 1e-8)
#     # Orthogonalize second column
#     c1_orth = c1 - (c1 * e0).sum(dim=-1, keepdim=True) * e0
#     e1 = c1_orth / (torch.linalg.norm(c1_orth, dim=-1, keepdim=True) + 1e-8)
#     # Third column via cross product (guarantees right-handedness)
#     e2 = torch.linalg.cross(e0, e1, dim=-1)

#     R_ortho = torch.stack([e0, e1, e2], dim=-1)  # [*, 3, 3]
#     return R_ortho.reshape(*shape, 9)


def _prepare_inputs_quad(x: Tensor, u: Tensor) -> Tuple[Tensor, Tensor]:
    """Ensure inputs are batched: [nx] -> [1, nx]."""
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if u.ndim == 1:
        u = u.unsqueeze(0)
    assert x.shape[-1] == 15, f"Expected state dim 15, got {x.shape[-1]}"
    assert u.shape[-1] == 4, f"Expected action dim 4, got {u.shape[-1]}"
    return x, u


def _rodrigues_torch(v: Tensor) -> Tensor:
    """Compute rotation matrix from axis-angle vector via Rodrigues' formula.

    Matches ``rotation_matrix_from_vector`` from rpg_flightning.

    Input:  v  [*, 3]  axis-angle vector (angle = ||v||, axis = v/||v||)
    Output: R  [*, 3, 3]
    """
    K = _skew_torch(v)                                     # [*, 3, 3]
    theta = torch.linalg.norm(v, dim=-1, keepdim=True).unsqueeze(-1)  # [*, 1, 1]
    theta = theta + 1e-5  # avoid division by zero
    I = torch.eye(3, dtype=v.dtype, device=v.device)       # noqa: E741
    K2 = torch.matmul(K, K)
    sin_term = torch.sin(theta) / theta
    cos_term = (1.0 - torch.cos(theta)) / (theta * theta)
    return I + sin_term * K + cos_term * K2


def build_quadrotor_dynamics(
    *,
    mass: float = 0.752,
    gravity: float = 9.81,
    max_thrust: float = 20.0,
    max_body_rate: float = 6.0,
) -> Tuple[
    Callable[[Tensor, Tensor, float], Tensor],
    None,
]:
    """Factory returning ``(f_dyn, None)`` for the quadrotor.

    Matches the simplified dynamics from rpg_flightning:
    - Euler step for position and velocity
    - Exact rotation step via Rodrigues' formula

    State  (nx=15): [p(3), v(3), R_flat(9)]
    Action (nu=4):  [thrust, ωx, ωy, ωz]

    Returns
    -------
    f_dyn : callable(x, u, dt) -> x_next
        Differentiable dynamics (PyTorch).
    None
        Placeholder for the Jacobian function (autodiff will be used).
    """
    g_vec_vals = [0.0, 0.0, -gravity]

    def f_dyn_torch(x: Tensor, u: Tensor, dt: float) -> Tensor:
        x, u = _prepare_inputs_quad(x, u)
        batch = x.shape[0]
        dtype, device = x.dtype, x.device
        dt_t = torch.as_tensor(dt, dtype=dtype, device=device)
        g_vec = torch.tensor(g_vec_vals, dtype=dtype, device=device)

        p = x[..., 0:3]
        v = x[..., 3:6]
        R_flat = x[..., 6:15]
        R = R_flat.reshape(*R_flat.shape[:-1], 3, 3)

        # Clamp controls
        thrust = torch.clamp(u[..., 0:1], 0.0, max_thrust)   # [B, 1]
        omega = torch.clamp(u[..., 1:4], -max_body_rate, max_body_rate)  # [B, 3]

        # Acceleration: a = thrust / mass  (scalar, applied along body z)
        accel_body = torch.zeros_like(v)
        accel_body[..., 2] = thrust.squeeze(-1) / mass
        accel_world = g_vec + torch.einsum("...ij,...j->...i", R, accel_body)

        # Euler step for position and velocity
        p_new = p + dt_t * v
        v_new = v + dt_t * accel_world

        # Exact rotation step: R_new = R @ rodrigues(dt * omega)
        R_delta = _rodrigues_torch(dt_t * omega)              # [B, 3, 3]
        R_new = torch.matmul(R, R_delta)                      # [B, 3, 3]

        x_next = torch.cat([p_new, v_new, R_new.reshape(*R_new.shape[:-2], 9)], dim=-1)

        if batch == 1:
            return x_next.squeeze(0)
        return x_next

    return f_dyn_torch, None

# Quaternion-based dynamics (10-D state)

def _quat_rotate_z_torch(q: Tensor, s: Tensor) -> Tensor:
    """Compute R(q) @ [0, 0, s] without building the full rotation matrix.

    Only the third column of R(q) is needed, scaled by *s*.

    Input:  q [*, 4]  quaternion (scalar-first)
            s [*, 1]  scalar (thrust / mass)
    Output: v [*, 3]
    """
    q0, qx, qy, qz = q[..., 0:1], q[..., 1:2], q[..., 2:3], q[..., 3:4]
    # Third column of R(q) = [2(qx*qz + q0*qy), 2(qy*qz - q0*qx), 1 - 2(qx^2 + qy^2)]
    return torch.cat([
        2.0 * (qx * qz + q0 * qy) * s,
        2.0 * (qy * qz - q0 * qx) * s,
        (1.0 - 2.0 * (qx * qx + qy * qy)) * s,
    ], dim=-1)


def _quat_omega_product_torch(omega: Tensor, q: Tensor) -> Tensor:
    """Compute q_dot = 0.5 * Omega(omega) @ q as a quaternion product.

    Equivalent to 0.5 * [0, omega] ⊗ q  (Hamilton product), avoiding
    construction of the 4x4 Omega matrix.

    Input:  omega [*, 3]  body angular velocity
            q     [*, 4]  quaternion (scalar-first)
    Output: q_dot [*, 4]
    """
    q0 = q[..., 0:1]
    q_vec = q[..., 1:4]   # [qx, qy, qz]
    # [0, omega] ⊗ q = [-omega·q_vec, q0*omega + omega×q_vec]
    scalar = -(omega * q_vec).sum(dim=-1, keepdim=True)
    vector = q0 * omega + torch.linalg.cross(omega, q_vec, dim=-1)
    return 0.5 * torch.cat([scalar, vector], dim=-1)


def _prepare_inputs_quat(x: Tensor, u: Tensor) -> Tuple[Tensor, Tensor]:
    """Ensure inputs are batched: [nx] -> [1, nx]."""
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if u.ndim == 1:
        u = u.unsqueeze(0)
    assert x.shape[-1] == 10, f"Expected state dim 10, got {x.shape[-1]}"
    assert u.shape[-1] == 4, f"Expected action dim 4, got {u.shape[-1]}"
    return x, u


def build_quadrotor_dynamics_quat(
    *,
    mass: float = 0.752,
    gravity: float = 9.81,
    max_thrust: float = 20.0,
    max_body_rate: float = 6.0,
) -> Tuple[
    Callable[[Tensor, Tensor, float], Tensor],
    None,
]:
    """Factory returning ``(f_dyn, None)`` for quaternion-based quadrotor.

    State  (nx=10): [p(3), v(3), q(4)]  with q = [q0, qx, qy, qz] (scalar-first)
    Action (nu=4):  [f_r, ωx, ωy, ωz]   collective thrust + body rates

    Dynamics:
        ṗ = v
        v̇ = (1/m) R(q) f_r e_z − g e_z
        q̇ = (1/2) Ω(ω) q

    Returns
    -------
    f_dyn : callable(x, u, dt) -> x_next
        Differentiable RK4 dynamics (PyTorch).
    None
        Placeholder for the Jacobian function (autodiff will be used).
    """
    g_vec_vals = [0.0, 0.0, -gravity]

    def _continuous(x: Tensor, u: Tensor, g_vec: Tensor) -> Tensor:
        """Continuous-time derivative.  x: [B, 10], u: [B, 4] -> dx: [B, 10]."""
        v = x[..., 3:6]
        q = x[..., 6:10]

        thrust = u[..., 0:1]  # [B, 1]
        omega = u[..., 1:4]   # [B, 3]

        # ṗ = v
        p_dot = v

        # v̇ = (1/m) R(q) f_r e_z − g e_z
        v_dot = g_vec + _quat_rotate_z_torch(q, thrust / mass)

        # q̇ = (1/2) Ω(ω) q  (computed as quaternion product)
        q_dot = _quat_omega_product_torch(omega, q)

        return torch.cat([p_dot, v_dot, q_dot], dim=-1)

    def f_dyn_torch(x: Tensor, u: Tensor, dt: float) -> Tensor:
        x, u = _prepare_inputs_quat(x, u)
        batch = x.shape[0]
        dtype, device = x.dtype, x.device
        dt_t = torch.as_tensor(dt, dtype=dtype, device=device)

        g_vec = torch.tensor(g_vec_vals, dtype=dtype, device=device)

        # Clamp controls
        u_clamped = u.clone()
        u_clamped[..., 0] = torch.clamp(u[..., 0], 0.0, max_thrust)
        u_clamped[..., 1:] = torch.clamp(u[..., 1:], -max_body_rate, max_body_rate)

        # RK4
        k1 = _continuous(x, u_clamped, g_vec)
        k2 = _continuous(x + 0.5 * dt_t * k1, u_clamped, g_vec)
        k3 = _continuous(x + 0.5 * dt_t * k2, u_clamped, g_vec)
        k4 = _continuous(x + dt_t * k3, u_clamped, g_vec)
        x_next = x + (dt_t / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Normalize quaternion
        q_next = x_next[..., 6:10]
        q_next = q_next / (torch.linalg.norm(q_next, dim=-1, keepdim=True) + 1e-8)
        x_next = torch.cat([x_next[..., :6], q_next], dim=-1)

        if batch == 1:
            return x_next.squeeze(0)
        return x_next

    return f_dyn_torch, None
