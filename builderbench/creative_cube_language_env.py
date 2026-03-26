import json
import logging
import numpy as np

from gymnasium import Wrapper
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from typing import Any
from dataclasses import dataclass

import builderbench.lie as lie
from builderbench.creative_cube_env import CreativeCubeEnv

logger = logging.getLogger(__name__)

ENV_SYSTEM_PROMPT = """
You are an agent who must control a UR5e robot arm with a Robotiq 2F-85 parallel jaw gripper in a simulated environment with cube-shaped blocks.

## Environment Overview

1. Simulation is implemented using MuJoCo and approximates Newtonian physics.
2. All positions are in meters; all rotations are in radians. All coordinates are in the global frame, with z=0 as the ground surface.
3. Each cube has an edge length of 0.04 meters.
4. The gripper's maximum opening is 0.085 meters.
5. The environment provides these observations:
    - Current timestep and the total number of timesteps in the episode.
    - End-effector position and yaw.
    - Potential target position for the end-effector.
    - Positions and yaws of all cubes.
    - Target locations for some cubes.
    - Success condition for cubes with targets.
6. You have to output an action (conforming to the defined Action Schema below) at each step with the goal of achieving the success condition for ALL cubes that have an assigned target. Success condition: all cubes are at their respective targets and remain there stably. Move the end effector at its target position (if specified) upon task completion.

## Action Schema

### Action Types
1. "pick_and_place": Executes a Pick -> Lift -> Place -> Retreat plan using low-level controls (no collision avoidance).
    - "cube_id": int — ID of the object to grasp
    - "grasp_yaw": int — 0 or 1 (perpendicular axes for grasping a cube)
    - "pos": [x, y, z] — Target position to place the cube
    - "yaw": float — Cube placement rotation (radians)
    - Note: After placing, the arm retreats to [0.3, 0.0, 0.25].

2. "pick_and_hold": Executes a Pick -> Lift -> Hold plan (no collision avoidance).
    - "cube_id": int — ID of the object to grasp
    - "grasp_yaw": int — 0 or 1
    - "pos": [x, y, z] — Target hold position
    - "yaw": float — Target hold rotation (radians)
    - Note: The arm holds the cube in the specified pose.

3. "eef_target": Uses a PD controller to move the end-effector to a specified target position and yaw (no collision avoidance).
    - "pos": [x, y, z] — Target end-effector position (meters)
    - "yaw": float — Target end-effector yaw (radians)
    - "gripper": float — 0.0 (open) to 1.0 (closed)

4. "low_level": Applies delta end-effector control for a few timesteps (fine-grained control; plan sequences to achieve high-level tasks).
    - "action": [delta_x, delta_y, delta_z, delta_yaw, delta_gripper_strength]

### Output Format
Always output actions as a single, valid JSON object with the specified key/value structure.

### Examples
- Pick and Place:
  {"type": "pick_and_place", "cube_id": 0, "grasp_yaw": 0, "pos": [0.5, -0.2, 0.02], "yaw": 1.57}

- Pick and Hold:
  {"type": "pick_and_hold", "cube_id": 0, "grasp_yaw": 0, "pos": [0.5, -0.2, 0.2], "yaw": 0.0}

- End-Effector Target:
  {"type": "eef_target", "pos": [0.45, 0.1, 0.3], "yaw": 1.57, "gripper": 1.0}

- Low Level:
  {"type": "low_level", "action": [0.3, 0.0, 0.1, 0.0, 1.0]}
""".strip()


@dataclass
class BlockState:
    position: list[float]
    yaw: float

@dataclass
class SceneState:
    """Complete snapshot of the environment state at a single timestep.

    Attributes:
        effector_pos: End-effector XYZ position in metres.
        effector_yaw: End-effector yaw in radians.
        gripper_opening: Gripper opening fraction (0 = closed, 1 = open).
        blocks: Per-cube block states keyed by cube index.
        targets: Per-cube target XYZ positions keyed by cube index.
        masks: Per-cube goal masks (1 = active target, 0 = no target).
        successes: Per-cube success flags keyed by cube index.
        return_default: Whether the current task requires returning the end-effector to a default position.
        default_eef_position: Optional default end-effector position.
        time: Current simulation time in seconds.
        total_time: Total episode duration in seconds.
    """

    effector_pos: list[float]
    effector_yaw: float
    gripper_opening: float
    blocks: dict[int, BlockState]
    targets: dict[int, list[float]]
    masks: dict[int, int]
    successes: dict[int, bool]
    return_default: bool
    default_eef_position: list[float]
    time: float
    total_time: float

def get_language_description_from_scene(scene):
    """Format a SceneState into a human-readable text description of the environment."""
    lines = []
    for i, block in scene.blocks.items():
        target = scene.targets.get(i)
        success = scene.successes.get(i)
        mask = scene.masks.get(i)

        target_str = f"[{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}]" if mask else "N/A"
        success_str = f"{success[0]}" if mask else "N/A"

        lines.append(
            f"- Cube {i}: pos=[{block.position[0]:.3f}, {block.position[1]:.3f}, {block.position[2]:.3f}], "
            f"yaw={block.yaw:.3f}, "
            f"target={target_str}, "
            f"success={success_str}"
        )
    
    blocks_text = "\n".join(lines)
    
    target_line = ""
    if scene.return_default:
        target_line = f", target=[{scene.default_eef_position[0]:.3f}, {scene.default_eef_position[1]:.3f}, {scene.default_eef_position[2]:.3f}]"

    return (
        f"Time: {scene.time:.2f} / {scene.total_time:.2f}\n\n"
        f"End Effector State\n"
        f"- End Effector: pos=[{scene.effector_pos[0]:.3f}, {scene.effector_pos[1]:.3f}, {scene.effector_pos[2]:.3f}], yaw={scene.effector_yaw:.3f}{target_line}\n"
        f"- Gripper: {scene.gripper_opening:.3f}\n\n"
        f"Cube State\n"
        f"{blocks_text}"
    )

class CreativeCubeLanguageWrapper(Wrapper):
    """Gymnasium wrapper that exposes a language-level action interface.

    Instead of raw 5-D delta actions, callers submit JSON-encoded actions of
    four types:

    * ``pick_and_place`` -- grasp a cube, move it to a target, release, and retreat.
    * ``pick_and_hold`` -- grasp a cube and hold it at a target pose.
    * ``eef_target`` -- move the end-effector to an absolute target pose.
    * ``low_level`` -- pass a single delta action directly to the underlying env.

    Each high-level action is executed over multiple low-level timesteps until
    completion or the per-action step budget is exhausted.
    """

    def __init__(
        self,
        env: CreativeCubeEnv,
        max_steps_per_action: int = 250,
        min_norm: float = 0.0,
        dt: float = 0.2,
        noise: float = 0.0,
        noise_smoothing: float = 0.0,
    ):
        """Initialize the language wrapper around a CreativeCubeEnv.

        Args:
            env: The underlying CreativeCubeEnv instance.
            max_steps_per_action: Maximum number of low-level simulation steps
                allowed per high-level action before forcing termination.
            min_norm: Minimum norm enforced on delta actions by shape_diff,
                preventing near-zero movements from being zeroed out entirely.
            dt: Duration in seconds between consecutive keyframes in a motion plan.
            noise: Scale of Gaussian noise injected into planned trajectories.
                Set to 0.0 to disable noise entirely.
            noise_smoothing: Standard deviation for temporal Gaussian smoothing applied to
                the trajectory noise, controlling its temporal correlation.
        """
        super().__init__(env)
        self._max_steps_per_action = max_steps_per_action
        self._min_norm = min_norm
        self._env_dt = env._control_timestep  # Simulation control timestep, inherited from the environment.
        self._dt = dt
        self._noise = noise
        self._noise_smoothing = noise_smoothing

    def _is_vec(self, v, length):
        """Helper: Checks if v is a list of numbers with specific length."""

        return isinstance(v, list) and len(v) == length and all(isinstance(x, (int, float)) for x in v)

    def check_action_validity(self, candidate_action):
        """Validate a JSON-encoded action string.

        Returns:
            A tuple ``(action_dict, is_valid)``.  When the action is invalid
            the returned dict is a no-op ``eef_target`` fallback.
        """
        
        fallback_action = {
                'type': 'eef_target'
            }
        
        try:
            candidate_action = json.loads(candidate_action)
        except json.JSONDecodeError as e:
            logger.warning("Action Validation Failed: Invalid JSON format.")
            return fallback_action, False
        
        if not isinstance(candidate_action, dict) or 'type' not in candidate_action:
            logger.warning("Action Validation Failed: Missing 'type' field.")
            return fallback_action, False

        candidate_action_type = candidate_action['type']
        if candidate_action_type == "eef_target":
            valid = (self._is_vec(candidate_action.get("pos"), 3) and
                    isinstance(candidate_action.get("yaw"), (int, float)) and
                    isinstance(candidate_action.get("gripper"), (int, float)))

        elif candidate_action_type == "low_level":
            valid = self._is_vec(candidate_action.get("action"), 5)

        elif candidate_action_type in ["pick_and_place", "pick_and_hold"]:
            valid = (isinstance(candidate_action.get("cube_id"), int) and
                    candidate_action.get("cube_id") >= 0 and
                    candidate_action.get("cube_id") < self.env._num_cubes and
                    isinstance(candidate_action.get("grasp_yaw"), int) and
                    self._is_vec(candidate_action.get("pos"), 3) and
                    isinstance(candidate_action.get("yaw"), (int, float)))
        else:
            logger.warning("Action Validation Failed: Unknown type '%s'.", candidate_action_type)
            return fallback_action, False

        if valid:
            return candidate_action, True
        else:
            logger.warning("Action Validation Failed: Invalid parameters for '%s'.", candidate_action_type)
            return fallback_action, False

    def shape_diff(self, diff):
        """Shape the difference vector to have a minimum norm."""

        diff_norm = np.linalg.norm(diff)
        if diff_norm >= self._min_norm:
            return diff
        else:
            return diff / (diff_norm + 1e-6) * self._min_norm
    
    def to_pose(self, pos, yaw):
        """Build an SE3 pose from a position and a yaw angle."""

        return lie.SE3.from_rotation_and_translation(
            rotation=lie.SO3.from_z_radians(yaw),
            translation=pos,
        )
    
    def get_yaw(self, pose):
        """Extract the yaw angle from an SE3 pose, mapped to [0, 2*pi)."""

        yaw = pose.rotation().compute_yaw_radians()
        if yaw < 0.0:
            return yaw + 2 * np.pi
        return yaw
    
    def shortest_yaw(self, eff_yaw, obj_yaw, translation, n=4, offset=0.0):
        """Find the symmetry-aware shortest yaw angle to the object."""

        symmetries = np.array([i * 2 * np.pi / n + obj_yaw + offset for i in range(-n, n + 1)])
        d = np.argmin(np.abs(eff_yaw - symmetries))
        return lie.SE3.from_rotation_and_translation(
            rotation=lie.SO3.from_z_radians(symmetries[d]),
            translation=translation,
        )
    
    def above(self, pose, z):
        """Return a copy of *pose* translated upward by *z* metres."""

        return (
            lie.SE3.from_rotation_and_translation(
                rotation=lie.SO3.identity(),
                translation=np.array([0.0, 0.0, z]),
            )
            @ pose
        )
    
    def get_eef_state(self):
        """Return the current end-effector position, yaw, and gripper opening."""

        ob_info = self.env.compute_ob_info()
        eef_pos = ob_info['proprio/effector_pos'].copy()
        eef_yaw = float(ob_info['proprio/effector_yaw'][0])
        gripper_opening = float(ob_info['proprio/gripper_opening'][0])
        return eef_pos, eef_yaw, gripper_opening
    
    def reset(self, options=None, *args, **kwargs):
        """Reset the environment and return a scene description string."""

        obs, info = super().reset(options=options, *args, **kwargs)
        scene = self.extract_scene_state(info)
        scene_str = get_language_description_from_scene(scene)
        info['scene_description'] = scene_str
        return info['scene_description'], info
    
    def step(self, action, render=False, video_stride=1):
        """Execute a high-level action, returning a scene description and cumulative reward."""

        obs_list = []
        action_info = self.retrieve_action_info(action)
        step_reward = 0.0
        video_stride = max(1, int(video_stride))
        for t in range(self._max_steps_per_action):
            low_level_action, low_level_terminated = self.retrieve_low_level_action(action, action_info)
            obs, reward, terminated, truncated, info = self.env.step(low_level_action)
            step_reward += reward

            step_idx = int(round(info["time"][0] / self.env._control_timestep))
            should_render = render and (step_idx % video_stride == 0)
            obs_list.append({
                'time': info['time'][0],
                'scene_description': info.get('scene_description', ''),
                'rendered_frame': self.env.render() if should_render else None
            })

            if low_level_terminated or terminated or truncated:
                break

        scene = self.extract_scene_state(info)
        scene_str = get_language_description_from_scene(scene)
        info['scene_description'] = scene_str
        info['action_steps'] = (t + 1)
        info['observation_history'] = obs_list
        return info['scene_description'], step_reward, terminated, truncated, info

    def extract_scene_state(self, info):
        """Return a SceneState instance from an info dict."""

        effector_pos = info["proprio/effector_pos"].tolist()
        effector_yaw = float(info["proprio/effector_yaw"][0])
        gripper_opening = float(info["proprio/gripper_opening"][0])
        time_value = float(info["time"][0])

        target_ids = getattr(self.env, "_cube_target_mocap_ids", [])
        blocks: Dict[int, BlockState] = {}
        targets: Dict[int, list[float]] = {}
        masks: Dict[int, int] = {}
        successes: Dict[int, bool] = {}
        for idx in range(len(target_ids)):
            blocks[idx] = BlockState(
                position=info[f"privileged/block_{idx}_pos"].tolist(),
                yaw=float(info[f"privileged/block_{idx}_yaw"][0]),
            )
            targets[idx] = self.env._data.mocap_pos[target_ids[idx]].copy().tolist()
            masks[idx] = self.env._goal_masks[idx]
            successes[idx] = info[f'privileged/block_{idx}_success']

        return SceneState(
            effector_pos=effector_pos,
            effector_yaw=effector_yaw,
            gripper_opening=gripper_opening,
            blocks=blocks,
            targets=targets,
            masks=masks,
            successes=successes,
            return_default=self.env.cur_task_info['return_defaults'],
            default_eef_position=self.env._neutral_arm_position.copy().tolist(),
            time=time_value,
            total_time=self.env._total_timesteps * self.env._control_timestep
        )

    def retrieve_low_level_action(self, action, action_info):
        """Dispatch to the appropriate low-level action generator."""

        action_type = action.get('type')
        if action_type == 'low_level':
            return action['action'], True
        elif action_type == 'eef_target':
            return self.get_eff_target_action(
                action_info['plan'], action_info['time']
            )
        elif action_type == 'pick_and_hold':
            return self.get_pick_and_hold_action(
                action_info['plan'], action_info['time']
            )
        elif action_type == 'pick_and_place':
            return self.get_pick_and_place_action(
                action_info['plan'], action_info['time']
            )
        else:
            raise ValueError(f"Unknown action type: {action_type}")

    def retrieve_action_info(self, action):
        """Pre-compute plan data needed by `retrieve_low_level_action` method.

        Supported action types: ``low_level``, ``eef_target``,
        ``pick_and_hold``, and ``pick_and_place``.
        """

        action_type = action.get('type')
        if action_type == 'low_level':

            return {}
        
        elif action_type == 'eef_target':

            ob_info = self.env.compute_ob_info()
            target_eef_pos = np.array(action.get('pos')) if action.get('pos') is not None else ob_info['proprio/effector_pos'].copy()
            target_eef_yaw = action.get('yaw') if action.get('yaw') is not None else float(ob_info['proprio/effector_yaw'][0])
            target_gripper_opening = action.get('gripper') if action.get('gripper') is not None else float(ob_info['proprio/gripper_opening'][0])
            plan = self.compute_eef_target_plan(
                effector_initial=self.to_pose(
                    pos=ob_info['proprio/effector_pos'],
                    yaw=ob_info['proprio/effector_yaw'][0],
                ),
                gripper_initial=float(ob_info['proprio/gripper_opening'][0]),
                target_eef_pos=target_eef_pos,
                target_eef_yaw=target_eef_yaw,
                target_gripper_opening=target_gripper_opening,
            )
            return {
                'plan': plan,
                'time': ob_info['time'][0]
            }
        
        elif action_type == 'pick_and_hold':

            cube_id = action['cube_id']
            grasp_yaw_id = action.get('grasp_yaw', 0)
            target_cube_pos = np.array(action.get('pos'), dtype=float)
            target_cube_yaw = float(action.get('yaw'))
            ob_info = self.env.compute_ob_info()
            plan_input = {
                'effector_initial': self.to_pose(
                    pos=ob_info['proprio/effector_pos'],
                    yaw=ob_info['proprio/effector_yaw'][0],
                ),
                'effector_goal': self.to_pose(
                    pos=np.array([0.3, 0.0, 0.25]),
                    yaw=0,
                ),
                'block_initial': self.to_pose(
                    pos=ob_info[f'privileged/block_{cube_id}_pos'],
                    yaw=ob_info[f'privileged/block_{cube_id}_yaw'][0],
                ),
                'block_goal': self.to_pose(
                    pos=target_cube_pos,
                    yaw=target_cube_yaw,
                ),
                'grasp_yaw_id': grasp_yaw_id,
            }
            times, poses, grasps = self.compute_pick_and_hold_keyframes(plan_input)
            poses = [poses[name] for name in times.keys()]
            grasps = [grasps[name] for name in times.keys()]
            times = list(times.values())
            plan = self.compute_pick_and_hold_plan(times, poses, grasps)
            return {
                'plan': plan,
                'time': ob_info['time'][0]
            }

        elif action_type == 'pick_and_place':

            cube_id = action['cube_id']
            grasp_yaw_id = action.get('grasp_yaw', 0)
            target_cube_pos = np.array(action.get('pos'), dtype=float)
            target_cube_yaw = float(action.get('yaw'))
            ob_info = self.env.compute_ob_info()
            plan_input = {
                'effector_initial': self.to_pose(
                    pos=ob_info['proprio/effector_pos'],
                    yaw=ob_info['proprio/effector_yaw'][0],
                ),
                'effector_goal': self.to_pose(
                    pos=np.array([0.3, 0.0, 0.25]),
                    yaw=0,
                ),
                'block_initial': self.to_pose(
                    pos=ob_info[f'privileged/block_{cube_id}_pos'],
                    yaw=ob_info[f'privileged/block_{cube_id}_yaw'][0],
                ),
                'block_goal': self.to_pose(
                    pos=target_cube_pos,
                    yaw=target_cube_yaw,
                ),
                'grasp_yaw_id': grasp_yaw_id,
            }
            times, poses, grasps = self.compute_pick_and_place_keyframes(plan_input)
            poses = [poses[name] for name in times.keys()]
            grasps = [grasps[name] for name in times.keys()]
            times = list(times.values())
            plan = self.compute_pick_and_place_plan(times, poses, grasps)
            return {
                'plan': plan,
                'time': ob_info['time'][0]
            }
        else:
            raise ValueError(f"Unknown action type: {action_type}")

    def get_pick_and_hold_action(self, plan, t_init):
        """Return the next low-level action for an in-progress pick-and-hold plan."""

        info = self.env.compute_ob_info()
        cur_plan_idx = int((info['time'][0] - t_init + 1e-7) // self._env_dt)
        if cur_plan_idx >= len(plan) - 1:
            cur_plan_idx = len(plan) - 1
            done = True
        else:
            done = False
        # Compute the difference between the current state and the current plan.
        ab_action = plan[cur_plan_idx]
        action = np.zeros(5)
        action[:3] = ab_action[:3] - info['proprio/effector_pos']
        action[3] = ab_action[3] - info['proprio/effector_yaw'][0]
        action[4] = ab_action[4] - info['proprio/gripper_opening'][0]
        action = self.env.normalize_action(action)
        return action, done

    def compute_pick_and_hold_keyframes(self, plan_input):
        """Compute time, pose, and grasp keyframes for a pick-and-hold action."""

        # Poses.
        poses = {}

        # Pick.
        block_initial = self.shortest_yaw(
            eff_yaw=self.get_yaw(plan_input['effector_initial']),
            obj_yaw=self.get_yaw(plan_input['block_initial']),
            translation=plan_input['block_initial'].translation(),
            n=2,
            offset=plan_input['grasp_yaw_id'] * np.pi / 2
        )
        poses['initial'] = plan_input['effector_initial']
        poses['pick'] = self.above(block_initial, 0.1 + np.random.uniform(0, 0.1))
        poses['pick_start'] = block_initial
        poses['pick_end'] = block_initial
        poses['postpick'] = poses['pick']

        # Place.
        block_goal = self.shortest_yaw(
            eff_yaw=self.get_yaw(poses['postpick']),
            obj_yaw=self.get_yaw(plan_input['block_goal']),
            translation=plan_input['block_goal'].translation(),
        )
        poses['hold'] = block_goal
        poses['hold_start'] = block_goal
        poses['hold_end'] = block_goal
        poses['final'] = poses['hold']

        # Clearance.
        midway = lie.interpolate(poses['postpick'], poses['hold'])
        poses['clearance'] = lie.SE3.from_rotation_and_translation(
            rotation=midway.rotation(),
            translation=np.array([*midway.translation()[:2], poses['initial'].translation()[-1]])
            + np.random.uniform([-0.1, -0.1, 0], [0.1, 0.1, 0.2]),
        )

        # Times.
        times = {}
        times['initial'] = 0.0
        times['pick'] = times['initial'] + self._dt * 2
        times['pick_start'] = times['pick'] + self._dt * 2
        times['pick_end'] = times['pick_start'] + self._dt
        times['postpick'] = times['pick_end'] + self._dt
        times['clearance'] = times['postpick'] + self._dt
        times['hold'] = times['clearance'] + self._dt
        times['hold_start'] = times['hold'] + self._dt
        times['hold_end'] = times['hold_start'] + self._dt
        times['final'] = times['hold_end'] + self._dt
        for time in times.keys():
            if time != 'initial':
                times[time] += np.random.uniform(-1, 1) * self._dt * 0.01

        # Grasps.
        g = 0.0
        grasps = {}
        for name in times.keys():
            if name in {'pick_end'}:
                g = 1.0 - g
            grasps[name] = g

        return times, poses, grasps

    def compute_pick_and_hold_plan(self, times, poses, grasps):
        """Interpolate keyframes into a dense plan array for pick-and-hold."""
        
        # Interpolate grasps.
        grasp_interp = interp1d(times, grasps, kind='linear', axis=0, assume_sorted=True)

        # Interpolate poses.
        xyzs = [p.translation() for p in poses]
        xyz_interp = interp1d(times, xyzs, kind='linear', axis=0, assume_sorted=True)

        # Interpolate orientations.
        quats = [p.rotation() for p in poses]

        def quat_interp(t):
            s = np.searchsorted(times, t, side='right') - 1
            interp_time = (t - times[s]) / (times[s + 1] - times[s])
            interp_time = np.clip(interp_time, 0.0, 1.0)
            return lie.interpolate(quats[s], quats[s + 1], interp_time)

        # Generate the plan.
        plan = []
        t = 0.0
        while t < times[-1]:
            action = np.zeros(5)
            action[:3] = xyz_interp(t)
            action[3] = quat_interp(t).compute_yaw_radians()
            action[4] = grasp_interp(t)
            plan.append(action)
            t += self._env_dt

        plan = np.array(plan)

        # Add temporally correlated noise to the plan.
        if self._noise > 0:
            noise = np.random.normal(0, 1, size=(len(plan), 5)) * np.array([0.05, 0.05, 0.05, 0.3, 1.0]) * self._noise
            noise = gaussian_filter1d(noise, axis=0, sigma=self._noise_smoothing)
            plan += noise

        return plan

    def get_pick_and_place_action(self, plan, t_init):
        """Return the next low-level action for an in-progress pick-and-place plan."""

        info = self.env.compute_ob_info()
        cur_plan_idx = int((info['time'][0] - t_init + 1e-7) // self._env_dt)
        if cur_plan_idx >= len(plan) - 1:
            cur_plan_idx = len(plan) - 1
            done = True
        else:
            done = False
        # Compute the difference between the current state and the current plan.
        ab_action = plan[cur_plan_idx]
        action = np.zeros(5)
        action[:3] = ab_action[:3] - info['proprio/effector_pos']
        action[3] = ab_action[3] - info['proprio/effector_yaw'][0]
        action[4] = ab_action[4] - info['proprio/gripper_opening'][0]
        action = self.env.normalize_action(action)
        return action, done

    def compute_pick_and_place_keyframes(self, plan_input):
        """Compute time, pose, and grasp keyframes for a pick-and-place action."""

        # Poses.
        poses = {}

        # Pick.
        block_initial = self.shortest_yaw(
            eff_yaw=self.get_yaw(plan_input['effector_initial']),
            obj_yaw=self.get_yaw(plan_input['block_initial']),
            translation=plan_input['block_initial'].translation(),
            n=2,
            offset=plan_input['grasp_yaw_id'] * np.pi / 2
        )
        poses['initial'] = plan_input['effector_initial']
        poses['pick'] = self.above(block_initial, 0.1 + np.random.uniform(0, 0.1))
        poses['pick_start'] = block_initial
        poses['pick_end'] = block_initial
        poses['postpick'] = poses['pick']

        # Place.
        block_goal = self.shortest_yaw(
            eff_yaw=self.get_yaw(poses['postpick']),
            obj_yaw=self.get_yaw(plan_input['block_goal']),
            translation=plan_input['block_goal'].translation(),
        )
        poses['place'] = self.above(block_goal, 0.1 + np.random.uniform(0, 0.1))
        poses['place_start'] = block_goal
        poses['place_end'] = block_goal
        poses['postplace'] = poses['place']
        poses['final'] = plan_input['effector_goal']

        # Clearance.
        midway = lie.interpolate(poses['postpick'], poses['place'])
        poses['clearance'] = lie.SE3.from_rotation_and_translation(
            rotation=midway.rotation(),
            translation=np.array([*midway.translation()[:2], poses['initial'].translation()[-1]])
            + np.random.uniform([-0.1, -0.1, 0], [0.1, 0.1, 0.2]),
        )

        # Times.
        times = {}
        times['initial'] = 0.0
        times['pick'] = times['initial'] + self._dt * 2
        times['pick_start'] = times['pick'] + self._dt * 2
        times['pick_end'] = times['pick_start'] + self._dt
        times['postpick'] = times['pick_end'] + self._dt
        times['clearance'] = times['postpick'] + self._dt
        times['place'] = times['clearance'] + self._dt
        times['place_start'] = times['place'] + self._dt * 2
        times['place_end'] = times['place_start'] + self._dt
        times['postplace'] = times['place_end'] + self._dt
        times['final'] = times['postplace'] + self._dt * 2
        for time in times.keys():
            if time != 'initial':
                times[time] += np.random.uniform(-1, 1) * self._dt * 0.01

        # Grasps.
        g = 0.0
        grasps = {}
        for name in times.keys():
            if name in {'pick_end', 'place_end'}:
                g = 1.0 - g
            grasps[name] = g

        return times, poses, grasps
    
    def compute_pick_and_place_plan(self, times, poses, grasps):
        """Interpolate keyframes into a dense plan array for pick-and-place."""
        
        # Interpolate grasps.
        grasp_interp = interp1d(times, grasps, kind='linear', axis=0, assume_sorted=True)

        # Interpolate poses.
        xyzs = [p.translation() for p in poses]
        xyz_interp = interp1d(times, xyzs, kind='linear', axis=0, assume_sorted=True)

        # Interpolate orientations.
        quats = [p.rotation() for p in poses]

        def quat_interp(t):
            s = np.searchsorted(times, t, side='right') - 1
            interp_time = (t - times[s]) / (times[s + 1] - times[s])
            interp_time = np.clip(interp_time, 0.0, 1.0)
            return lie.interpolate(quats[s], quats[s + 1], interp_time)

        # Generate the plan.
        plan = []
        t = 0.0
        while t < times[-1]:
            action = np.zeros(5)
            action[:3] = xyz_interp(t)
            action[3] = quat_interp(t).compute_yaw_radians()
            action[4] = grasp_interp(t)
            plan.append(action)
            t += self._env_dt

        plan = np.array(plan)

        # Add temporally correlated noise to the plan.
        if self._noise > 0:
            noise = np.random.normal(0, 1, size=(len(plan), 5)) * np.array([0.05, 0.05, 0.05, 0.3, 1.0]) * self._noise
            noise = gaussian_filter1d(noise, axis=0, sigma=self._noise_smoothing)
            plan += noise

        return plan

    def get_eff_target_action(self, plan, t_init):
        """Return the next low-level action for an in-progress eef_target plan."""
        info = self.env.compute_ob_info()
        cur_plan_idx = int((info['time'][0] - t_init + 1e-7) // self._env_dt)
        if cur_plan_idx >= len(plan) - 1:
            cur_plan_idx = len(plan) - 1
            done = True
        else:
            done = False

        ab_action = plan[cur_plan_idx]
        action = np.zeros(5)
        action[:3] = ab_action[:3] - info['proprio/effector_pos']
        action[3] = ab_action[3] - info['proprio/effector_yaw'][0]
        action[4] = ab_action[4] - info['proprio/gripper_opening'][0]
        action = self.env.normalize_action(action)
        return action, done

    def compute_eef_target_plan(
        self,
        effector_initial,
        gripper_initial,
        target_eef_pos,
        target_eef_yaw,
        target_gripper_opening,
    ):
        """Compute a dense waypoint plan interpolating the eef from its current pose to the target pose."""
        diff = target_eef_pos - effector_initial.translation()
        yaw_diff = abs((target_eef_yaw - self.get_yaw(effector_initial) + np.pi) % (2 * np.pi) - np.pi)
        duration = max(self._dt * 2, np.linalg.norm(diff) / 0.5, yaw_diff / 1.0)
        settle_duration = self._dt * 2

        target_pose = self.to_pose(pos=target_eef_pos, yaw=target_eef_yaw)
        times = [0.0, duration, duration + settle_duration]
        poses = [
            effector_initial,
            target_pose,
            target_pose,
        ]
        grasps = [gripper_initial, target_gripper_opening, target_gripper_opening]

        grasp_interp = interp1d(times, grasps, kind='linear', axis=0, assume_sorted=True)
        xyzs = [p.translation() for p in poses]
        xyz_interp = interp1d(times, xyzs, kind='linear', axis=0, assume_sorted=True)
        quats = [p.rotation() for p in poses]

        def quat_interp(t):
            s = np.searchsorted(times, t, side='right') - 1
            interp_time = (t - times[s]) / (times[s + 1] - times[s])
            interp_time = np.clip(interp_time, 0.0, 1.0)
            return lie.interpolate(quats[s], quats[s + 1], interp_time)

        plan = []
        t = 0.0
        while t < times[-1]:
            action = np.zeros(5)
            action[:3] = xyz_interp(t)
            action[3] = quat_interp(t).compute_yaw_radians()
            action[4] = grasp_interp(t)
            plan.append(action)
            t += self._env_dt

        plan = np.array(plan)

        if self._noise > 0:
            noise = np.random.normal(0, 1, size=(len(plan), 5)) * np.array([0.05, 0.05, 0.05, 0.3, 1.0]) * self._noise
            noise = gaussian_filter1d(noise, axis=0, sigma=self._noise_smoothing)
            plan += noise

        return plan