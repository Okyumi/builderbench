"""Simple teleop UI for CreativeCubeEnv (no language wrapper).

Takes a task as input (--level-id), simulates the scene with the corresponding
number of blocks and robot, and lets the user control the robot via 5 sliders
without episode termination.

Sliders set the target end-effector state:
- x, y, z: target end-effector position (meters)
- yaw: target end-effector yaw (radians)
- gripper: target gripper opening (0 to 1)

The targets are converted to low-level actions via a PD-style controller in
`get_eff_target_action`. The episode never terminates automatically; use the
"Reset Env" button to reset manually.

Usage:
    python teleop_gui.py --level-id cube-3-task-1
"""

from __future__ import annotations

import argparse
import os
import threading
import time
import tkinter as tk
from dataclasses import dataclass

import numpy as np

from builderbench.creative_cube_env import CreativeCubeEnv


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
    lock: threading.Lock | None = None


class SliderUI:
    def __init__(self, state: TeleopState, level: str, bounds: np.ndarray):
        self.state = state
        self.root = tk.Tk()
        self.root.title(f"CreativeCube Teleop — {level}")
        self.root.protocol("WM_DELETE_WINDOW", self._on_quit)

        self._scales: list[tk.Scale] = []
        labels = ["x", "y", "z", "yaw", "gripper"]
        ranges = [
            (float(bounds[0, 0]), float(bounds[1, 0])),
            (float(bounds[0, 1]), float(bounds[1, 1])),
            (float(bounds[0, 2]), float(bounds[1, 2])),
            (-np.pi, np.pi),
            (0.0, 1.0),
        ]

        frame = tk.Frame(self.root, padx=12, pady=12)
        frame.pack(fill=tk.BOTH, expand=True)

        help_text = (
            "Each slider sets a target end-effector pose (x, y, z, yaw) and gripper.\n"
            "Targets are converted to low-level actions with a PD-style controller."
        )
        tk.Label(frame, text=help_text, justify=tk.LEFT).pack(anchor="w", pady=(0, 8))

        for i, label in enumerate(labels):
            row = tk.Frame(frame)
            row.pack(fill=tk.X, pady=2)

            tk.Label(row, text=label, width=9, anchor="w").pack(side=tk.LEFT)
            low, high = ranges[i]
            scale = tk.Scale(
                row,
                from_=low,
                to=high,
                resolution=0.001 if i < 3 else 0.01,
                orient=tk.HORIZONTAL,
                length=360,
                command=lambda val, idx=i: self._update_action(idx, val),
            )
            scale.set((low + high) / 2.0)
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self._scales.append(scale)

        button_row = tk.Frame(frame)
        button_row.pack(fill=tk.X, pady=(10, 0))

        tk.Button(button_row, text="Snap to Current", command=self._snap_to_current).pack(side=tk.LEFT)
        tk.Button(button_row, text="Reset Env", command=self._request_reset).pack(side=tk.LEFT, padx=8)
        tk.Button(button_row, text="Quit", command=self._on_quit).pack(side=tk.RIGHT)

    def _update_action(self, idx: int, value: str):
        if self.state.lock is not None:
            with self.state.lock:
                self._set_target_value(idx, float(value))
        else:
            self._set_target_value(idx, float(value))

    def _set_target_value(self, idx: int, value: float):
        if idx < 3:
            self.state.target_pos[idx] = value
        elif idx == 3:
            self.state.target_yaw = value
        else:
            self.state.target_gripper = value

    def _snap_to_current(self):
        if self.state.lock is not None:
            with self.state.lock:
                pos = self.state.current_pos.copy()
                yaw = float(self.state.current_yaw)
                grip = float(self.state.current_gripper)
        else:
            pos = self.state.current_pos.copy()
            yaw = float(self.state.current_yaw)
            grip = float(self.state.current_gripper)

        values = [pos[0], pos[1], pos[2], yaw, grip]
        for i, scale in enumerate(self._scales):
            scale.set(values[i])
        if self.state.lock is not None:
            with self.state.lock:
                self.state.target_pos[:] = pos
                self.state.target_yaw = yaw
                self.state.target_gripper = grip
        else:
            self.state.target_pos[:] = pos
            self.state.target_yaw = yaw
            self.state.target_gripper = grip

    def _check_quit(self):
        """Periodically checks if the background thread requested a quit."""
        quit_now = False
        if self.state.lock is not None:
            with self.state.lock:
                quit_now = self.state.quit_requested
        else:
            quit_now = self.state.quit_requested

        if quit_now:
            # If the viewer closed, shut down the UI
            self.root.destroy()
            self.root.quit()
        else:
            # Otherwise, schedule this check again in 100 milliseconds
            self.root.after(100, self._check_quit)

    def _request_reset(self):
        self.state.reset_requested = True

    def _on_quit(self):
        self.state.quit_requested = True
        self.root.destroy()
        self.root.quit()

    def run(self):
        self.root.after(100, self._check_quit)
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teleop CreativeCubeEnv with 5D action sliders")
    parser.add_argument("--level-id", type=str, default="cube-1-task-1")
    parser.add_argument("--seed", type=int, default=0, help="Reset seed")
    parser.add_argument(
        "--step-sleep",
        type=float,
        default=0.0,
        help="Optional wall-clock sleep per control step for slower playback (seconds)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = CreativeCubeEnv(args.level_id)
    env._total_timesteps = np.inf
    obs, info = env.reset(seed=args.seed)
    _ = (obs, info)  # kept for parity/debug convenience

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
        lock=threading.Lock(),
    )
    ui = SliderUI(state=state, level=args.level_id, bounds=bounds)

    ui._snap_to_current()

    print("Teleop started.")
    print("- Use slider window to set target EEF pose + gripper in real time")
    print("- Viewer shortcuts: mouse drag/scroll in MuJoCo viewer")

    def sim_loop() -> None:
        with env.unwrapped.passive_viewer() as viewer:
            while True:
                # Detect if the user closed the MuJoCo viewer window directly.
                if hasattr(viewer, "is_running") and not viewer.is_running():
                    with state.lock:
                        state.quit_requested = True
                    break

                with state.lock:
                    quit_now = state.quit_requested

                if quit_now:
                    if hasattr(viewer, "close"):
                        viewer.close()
                    break

                with state.lock:
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

                if args.step_sleep > 0:
                    time.sleep(args.step_sleep)

    sim_thread = threading.Thread(target=sim_loop, daemon=True)
    sim_thread.start()

    # Tkinter must run on the main thread.
    ui.run()

    with state.lock:
        state.quit_requested = True
    sim_thread.join(timeout=1.0)
    print("Teleop ended.")
    os._exit(0)


def get_eff_target_action(
    env: CreativeCubeEnv,
    target_pos: np.ndarray,
    target_yaw: float,
    target_gripper: float,
    min_norm: float = 0.0,
    gain_pos: float = 5.0,
    gain_yaw: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Compute a low-level action from target EEF pose/gripper.
    """
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


if __name__ == "__main__":
    main()
