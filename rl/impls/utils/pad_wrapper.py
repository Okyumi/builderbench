"""Pads BuilderBench observations and goals to a fixed dimensionality.

When tasks have fewer cubes than MAX_CUBES, the extra cube slots in the
observation and goal are filled with zeros.  This allows networks trained
on one task to be transferred to another task with a different cube count
without shape mismatches.

Observation layout (N cubes):
    gripper_pos(3) + gripper_quat(4) + gripper_linvel(3)               = 10
    obj_pos(3*N) + obj_quat(4*N) + obj_linvel(3*N) + obj_angvel(3*N)  = 13*N
    finger_pos(8)                                                       = 8
    Total: 18 + 13*N

Goal layout: achieved_goal = positions of target cubes = 3 * num_target dims.
"""
import jax.numpy as jnp

from builderbench.env_utils import State

MAX_CUBES = 3

FIXED_OBS_PREFIX = 10   # gripper_pos(3) + gripper_quat(4) + gripper_linvel(3)
FIXED_OBS_SUFFIX = 8    # finger_pos(8)
PER_CUBE_OBS = 13       # pos(3) + quat(4) + linvel(3) + angvel(3)
UNIFIED_OBS_DIM = FIXED_OBS_PREFIX + PER_CUBE_OBS * MAX_CUBES + FIXED_OBS_SUFFIX  # 57
UNIFIED_GOAL_DIM = 3 * MAX_CUBES  # 9


class PaddedEnvWrapper:
    """Wraps a BuilderBench env to produce fixed-dim observations and goals.

    Must wrap *outside* of ``wrap_env`` (VmapWrapper → EpisodeWrapper →
    AutoResetWrapper) so that auto-reset bookkeeping sees the padded shapes.

    Args:
        env: The underlying (already-wrapped) BuilderBench environment.
        actual_cubes: Number of cubes in this task's environment.
    """

    def __init__(self, env, actual_cubes: int):
        self._env = env
        self._actual_cubes = actual_cubes
        self._pad_cubes = MAX_CUBES - actual_cubes
        assert actual_cubes <= MAX_CUBES, (
            f'Task has {actual_cubes} cubes but MAX_CUBES={MAX_CUBES}')

    # ---- size properties (unified across all tasks) -----------------------

    @property
    def observation_size(self):
        return UNIFIED_OBS_DIM

    @property
    def goal_size(self):
        return UNIFIED_GOAL_DIM

    @property
    def action_size(self):
        return self._env.action_size

    # ---- padding helpers --------------------------------------------------

    def _pad_obs(self, obs):
        """Pad observation from actual_cubes to MAX_CUBES."""
        if self._pad_cubes == 0:
            return obs
        actual_obj_dim = PER_CUBE_OBS * self._actual_cubes
        prefix = obs[..., :FIXED_OBS_PREFIX]
        obj_data = obs[..., FIXED_OBS_PREFIX:FIXED_OBS_PREFIX + actual_obj_dim]
        suffix = obs[..., FIXED_OBS_PREFIX + actual_obj_dim:]
        pad_shape = obs.shape[:-1] + (PER_CUBE_OBS * self._pad_cubes,)
        obj_pad = jnp.zeros(pad_shape, dtype=obs.dtype)
        return jnp.concatenate([prefix, obj_data, obj_pad, suffix], axis=-1)

    def _pad_goal(self, goal):
        """Pad goal from actual target cubes to MAX_CUBES."""
        actual_goal_dim = goal.shape[-1]
        if actual_goal_dim >= UNIFIED_GOAL_DIM:
            return goal[..., :UNIFIED_GOAL_DIM]
        pad_shape = goal.shape[:-1] + (UNIFIED_GOAL_DIM - actual_goal_dim,)
        return jnp.concatenate([goal, jnp.zeros(pad_shape, dtype=goal.dtype)],
                               axis=-1)

    def _pad_state(self, state):
        """Pad obs and goal fields in a State.

        NOTE: We deliberately do NOT pad ``first_obs`` / ``first_achieved_goal``
        stored by ``AutoResetWrapper``.  Those are internal bookkeeping arrays
        that get swapped with raw (unpadded) obs inside ``AutoResetWrapper.
        post_step`` via ``jnp.where``, so they must keep the raw shape.
        We only pad the fields that the agent (policy / critic) sees.
        """
        new_obs = self._pad_obs(state.obs)
        new_info = dict(state.info)
        if 'achieved_goal' in new_info:
            new_info['achieved_goal'] = self._pad_goal(new_info['achieved_goal'])
        if 'target_goal' in new_info:
            new_info['target_goal'] = self._pad_goal(new_info['target_goal'])
        return State(
            physics_state=state.physics_state,
            sensordata=state.sensordata,
            ctrl=state.ctrl,
            obs=new_obs,
            reward=state.reward,
            done=state.done,
            metrics=state.metrics,
            info=new_info,
        )

    # ---- env interface (matches Wrapper protocol) -------------------------

    def reset(self, rng):
        state = self._env.reset(rng)
        return self._pad_state(state)

    def pre_step(self, state, action):
        return self._env.pre_step(state, action)

    def step(self, state, action):
        return self._env.step(state, action)

    def post_step(self, state, physics_state, sensor_data):
        state = self._env.post_step(state, physics_state, sensor_data)
        return self._pad_state(state)

    # ---- forward everything else to the underlying env --------------------

    def __getattr__(self, name):
        return getattr(self._env, name)
