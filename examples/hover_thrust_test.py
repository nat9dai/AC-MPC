#!/usr/bin/env python3
"""
Hover thrust visual test with roll/pitch manoeuvres and thrust variation.

Sequence:
  1. Hover (level, 1.0× hover thrust)
  2. Thrust increase (1.5×) → hover (1.0×) → thrust decrease (0.5×) → hover
  3. Roll right  (+ωx)  → stop → roll left  (-ωx)  → stop
  4. Yaw
  5. Final hover (level)

Run from the repo root:
    python examples/hover_thrust_test.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ACMPC.envs.quadrotor_waypoint import QuadrotorWaypointEnv


# ---------------------------------------------------------------------------
# Manoeuvre sequence:  (steps, thrust_factor, wx, wy, wz, label)
# thrust_factor is a multiplier of hover_thrust (1.0 = steady hover)
# body rates in rad/s
# ---------------------------------------------------------------------------
RATE = 0.1   # rad/s used for all roll / pitch manoeuvres

PHASES = [
    # steps  T_factor  wx     wy     wz    label
    (40,     1.0,      0.0,   0.0,   0.0,  "Hover"),
    (30,     1.2,      0.0,   0.0,   0.0,  "Thrust +50%"),
    (40,     1.0,      0.0,   0.0,   0.0,  "Hover"),
    (70,     0.7,      0.0,   0.0,   0.0,  "Thrust -50%"),
    (40,     1.0,      0.0,   0.0,   0.0,  "Hover"),
    (20,     1.0,     +RATE,  0.0,   0.0,  "Roll +"),
    (20,     1.0,     -RATE,  0.0,   0.0,  "Roll -"),
    (40,     1.0,      0.0,   0.0,   0.0,  "Hover"),
    (40,     1.0,      0.0,   0.0,   RATE, "Yaw"),
    (40,     1.0,      0.0,   0.0,   0.0,  "Hover"),
]


def main() -> None:
    env = QuadrotorWaypointEnv(dt=0.02, episode_len=10_000)

    obs, info = env.reset(seed=0)
    # Start exactly at origin, level
    env.state[:3] = 0.0
    env.state[3:6] = 0.0
    env.state[6:15] = np.eye(3, dtype=np.float32).ravel()

    hover_thrust = env.mass * env.gravity   # [N]

    print(f"mass    = {env.mass:.3f} kg")
    print(f"gravity = {env.gravity:.3f} m/s²")
    print(f"hover T = {hover_thrust:.3f} N  (= mass * g)")
    print("Close the window or press Ctrl-C to stop.\n")

    try:
        for n_steps, t_factor, wx, wy, wz, label in PHASES:
            thrust = hover_thrust * t_factor
            action = np.array([thrust, wx, wy, wz], dtype=np.float32)
            print(f"--- {label:14s} | T={thrust:.2f}N ({t_factor:.1f}×)  wx={wx:+.1f}  wy={wy:+.1f}  wz={wz:+.1f} ---")

            for _ in range(n_steps):
                _, _, terminated, truncated, _ = env.step(action)
                env.render(mode="human")
                if terminated or truncated:
                    print("Episode ended early.")
                    return

            pos = env.state[:3]
            print(f"  end pos: [{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}]")

        # keep hovering at the end of the sequence
        while True:
            _, _, terminated, truncated, _ = env.step(action)
            env.render(mode="human")
            # print velocity
            vel = env.state[3:6]
            print(f"  velocity: [{vel[0]:+.3f}, {vel[1]:+.3f}, {vel[2]:+.3f}]")
            if terminated or truncated:
                print("Episode ended early.")
                return

        print("\nSequence complete.")

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
