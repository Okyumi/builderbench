"""Keyboard teleop for CreativeCubeEnv (no GUI).

Controls the robot end-effector via keyboard — no Tkinter window required.

Key bindings:
  Left / Right  — move left / right   (Y axis)
  Up / Down     — move forward / back (X axis)
  w / s         — move up / down      (Z axis)
  a / d         — yaw CCW / CW
  z / x         — open / close gripper
  End           — reset environment
  Esc / Ctrl-C  — quit

Usage:
    python teleop_keyboard.py --level-id cube-1-task-1
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from dataclasses import dataclass, field

import numpy as np
from pynput import keyboard

from builderbench.creative_cube_env import CreativeCubeEnv


# ── tuneable constants ────────────────────────────────────────────────────────
POS_STEP = 0.001      # metres per control step while key is held
YAW_STEP = 0.03       # radians per control step while key is held
GRIPPER_STEP = 0.05   # gripper fraction per control step while key is held
PRINT_HZ = 5.0        # status line refresh rate (Hz)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TeleopState:
    target_pos: np.ndarray
    target_yaw: float
    target_gripper: float
    current_pos: np.ndarray
    current_yaw: float
    current_gripper: float
    reset_requested: bool = False
    quit_requested: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


# Keys currently held down (shared between listener thread and sim thread).
_held: set = set()
_held_lock = threading.Lock()


def _norm_key(key) -> str | None:
    """Return a canonical string for keys we care about, else None."""
    try:
        c = key.char
        if c is not None:
            return c.lower()
    except AttributeError:
        pass
    mapping = {
        keyboard.Key.up: "up",
        keyboard.Key.down: "down",
        keyboard.Key.left: "left",
        keyboard.Key.right: "right",
        keyboard.Key.end: "end",
        keyboard.Key.esc: "esc",
    }
    return mapping.get(key)


def _on_press(key):
    k = _norm_key(key)
    if k:
        with _held_lock:
            _held.add(k)


def _on_release(key):
    k = _norm_key(key)
    if k:
        with _held_lock:
            _held.discard(k)


def _held_keys() -> set:
    with _held_lock:
        return set(_held)


def apply_keyboard_delta(state: TeleopState, bounds: np.ndarray) -> None:
    """Mutate *state* target values based on currently held keys (call with lock held)."""
    keys = _held_keys()

    # Position deltas
    dx = dy = dz = 0.0
    if "up" in keys:
        dx += POS_STEP
    if "down" in keys:
        dx -= POS_STEP
    if "left" in keys:
        dy += POS_STEP
    if "right" in keys:
        dy -= POS_STEP
    if "w" in keys:
        dz += POS_STEP
    if "s" in keys:
        dz -= POS_STEP

    dyaw = 0.0
    if "a" in keys:
        dyaw += YAW_STEP
    if "d" in keys:
        dyaw -= YAW_STEP

    dgripper = 0.0
    if "z" in keys:
        dgripper += GRIPPER_STEP
    if "x" in keys:
        dgripper -= GRIPPER_STEP

    # Apply and clamp
    new_pos = state.target_pos + np.array([dx, dy, dz])
    new_pos = np.clip(new_pos, bounds[0], bounds[1])
    state.target_pos[:] = new_pos

    state.target_yaw = float(np.clip(state.target_yaw + dyaw, -np.pi, np.pi))
    state.target_gripper = float(np.clip(state.target_gripper + dgripper, 0.0, 1.0))

    # One-shot keys
    if "end" in keys:
        state.reset_requested = True
        with _held_lock:
            _held.discard("end")

    if "esc" in keys:
        state.quit_requested = True


def get_eff_target_action(
    env: CreativeCubeEnv,
    target_pos: np.ndarray,
    target_yaw: float,
    target_gripper: float,
    min_norm: float = 0.0,
    gain_pos: float = 5.0,
    gain_yaw: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    ob_info = env.compute_ob_info()
    eef_pos = ob_info["proprio/effector_pos"].copy()
    eef_yaw = float(ob_info["proprio/effector_yaw"][0])
    gripper_opening = float(ob_info["proprio/gripper_opening"][0])

    diff = target_pos - eef_pos
    diff_norm = float(np.linalg.norm(diff))
    if diff_norm < min_norm:
        diff = diff / (diff_norm + 1e-6) * min_norm

    action = np.zeros(5, dtype=np.float32)
    action[:3] = diff[:3] * gain_pos
    action[3] = ((target_yaw - eef_yaw + np.pi) % (2 * np.pi) - np.pi) * gain_yaw
    action[4] = float(np.clip(target_gripper - gripper_opening, -1, 1))

    return action, eef_pos, eef_yaw, gripper_opening


def print_status(state: TeleopState) -> None:
    with state.lock:
        tp = state.target_pos.copy()
        ty = state.target_yaw
        tg = state.target_gripper
        cp = state.current_pos.copy()
        cy = state.current_yaw
        cg = state.current_gripper

    line = (
        f"\r  target  pos=({tp[0]:+.3f}, {tp[1]:+.3f}, {tp[2]:+.3f})  "
        f"yaw={ty:+.3f}  grip={tg:.2f}  |  "
        f"actual  pos=({cp[0]:+.3f}, {cp[1]:+.3f}, {cp[2]:+.3f})  "
        f"yaw={cy:+.3f}  grip={cg:.2f}  "
    )
    sys.stdout.write(line)
    sys.stdout.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keyboard teleop for CreativeCubeEnv")
    parser.add_argument("--level-id", type=str, default="cube-1-task-1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step-sleep", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = CreativeCubeEnv(args.level_id)
    env._total_timesteps = np.inf
    obs, info = env.reset(seed=args.seed)
    _ = (obs, info)

    bounds = env._workspace_bounds.copy()
    ob_info = env.compute_ob_info()
    init_pos = ob_info["proprio/effector_pos"].copy()
    init_yaw = float(ob_info["proprio/effector_yaw"][0])
    init_gripper = float(ob_info["proprio/gripper_opening"][0])

    state = TeleopState(
        target_pos=init_pos.copy(),
        target_yaw=init_yaw,
        target_gripper=init_gripper,
        current_pos=init_pos.copy(),
        current_yaw=init_yaw,
        current_gripper=init_gripper,
    )

    print("\n=== CreativeCube Keyboard Teleop ===\n")
    print("  Up/Down    — forward / back  (X)\n")
    print("  Left/Right — left / right    (Y)\n")
    print("  w/s        — up / down       (Z)\n")
    print("  a/d        — yaw CCW / CW\n")
    print("  z/x        — open / close gripper\n")
    print("  End        — reset environment\n")
    print("  Esc        — quit\n")
    print("=" * 36)

    # Start keyboard listener (runs in its own daemon thread).
    listener = keyboard.Listener(on_press=_on_press, on_release=_on_release, suppress=True)
    listener.start()

    last_print = time.monotonic()
    print_interval = 1.0 / PRINT_HZ

    def sim_loop() -> None:
        nonlocal last_print
        with env.unwrapped.passive_viewer() as viewer:
            # Suppress MuJoCo viewer's built-in key bindings so that teleop
            # keys (w/s/a/d, e/f, etc.) don't also trigger viewer shortcuts
            # (colour changes, geom toggles, etc.).
            if hasattr(viewer, "key_callback"):
                viewer.key_callback = lambda *args: None
            while True:
                if hasattr(viewer, "is_running") and not viewer.is_running():
                    with state.lock:
                        state.quit_requested = True
                    break

                with state.lock:
                    if state.quit_requested:
                        if hasattr(viewer, "close"):
                            viewer.close()
                        break

                    # Apply held-key deltas inside the lock.
                    apply_keyboard_delta(state, bounds)

                    # Handle reset.
                    do_reset = state.reset_requested
                    if do_reset:
                        state.reset_requested = False

                    target_pos = state.target_pos.copy()
                    target_yaw = float(state.target_yaw)
                    target_gripper = float(state.target_gripper)

                if do_reset:
                    env.reset(seed=args.seed)

                action, eef_pos, eef_yaw, gripper_opening = get_eff_target_action(
                    env,
                    target_pos=target_pos,
                    target_yaw=target_yaw,
                    target_gripper=target_gripper,
                )

                with state.lock:
                    state.current_pos[:] = eef_pos
                    state.current_yaw = eef_yaw
                    state.current_gripper = gripper_opening

                _, _, _, _, _ = env.step(action)
                env.sync_passive_viewer()

                now = time.monotonic()
                if now - last_print >= print_interval:
                    print_status(state)
                    last_print = now

                if args.step_sleep > 0:
                    time.sleep(args.step_sleep)

    sim_thread = threading.Thread(target=sim_loop, daemon=True)
    sim_thread.start()

    # Block main thread until quit is requested.
    try:
        while True:
            with state.lock:
                if state.quit_requested:
                    break
            time.sleep(0.05)
    except KeyboardInterrupt:
        with state.lock:
            state.quit_requested = True

    listener.stop()
    sim_thread.join(timeout=1.0)
    print("\nTeleop ended.")
    os._exit(0)


if __name__ == "__main__":
    main()
