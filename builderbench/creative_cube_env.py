import re
import mujoco
import numpy as np
from dm_control import mjcf
from pathlib import Path

import builderbench.lie as lie
from builderbench.manispace_env import ManipSpaceEnv

class CreativeCubeEnv(ManipSpaceEnv):
    """Creative Cube environment.

    This environment consists of a configurable number of cubes. The goal is to move the cubes to target positions,
    optionally requiring the arm to return to a neutral position after placement. Levels are specified in the format
    'cube-{num_cubes}-task-{task_id}' (e.g. 'cube-3-task-2').
    """

    def __init__(self, level, permute_blocks=False, physics_timestep=None, *args, **kwargs):
        """Initialize the Creative Cube environment.

        Args:
            level: Level string in the format 'cube-{num_cubes}-task-{task_id}', e.g. 'cube-3-task-2'.
                   Determines the number of cubes and which task configuration to load.
            permute_blocks: Whether to randomly permute the order of the blocks at task initialization.
            *args: Additional arguments to pass to the parent class.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """
        self._level = level
        self._permute_blocks = permute_blocks
        self._success_threshold = 0.005
        self._easy_success_threshold = 0.01
        self._neutral_arm_position = np.array([0.3, 0.0, 0.25])

        if re.fullmatch(r"cube-\d+-task-\d+", level):
            num_cubes = int( re.search(r"cube-(\d+)", level).group(1))
            task_id = int( re.search(r"task-(\d+)", level).group(1)) - 1
            self._num_cubes = num_cubes
            self._task_id = task_id
        else:
            raise ValueError(f'Invalid level: {level}')

        if self._num_cubes <= 10:
            physics_timestep = 0.002 if physics_timestep is None else physics_timestep
        else:
            physics_timestep = 0.001 if physics_timestep is None else physics_timestep

        super().__init__(physics_timestep=physics_timestep, *args, **kwargs)

        # Define constants.
        self._cube_colors, self._cube_success_colors = self.get_color_arrays()

    def get_color_arrays(self):
        # Slice the pairs to the number of cubes you currently have
        selected_pairs = self._color_pairs[:self._num_cubes]
        
        # Extract the base and light arrays
        base_arr = np.array([self._colors[p[0]] for p in selected_pairs])
        success_arr = np.array([self._colors[p[1]] for p in selected_pairs])

        return base_arr, success_arr
    
    def set_tasks(self):
        task_data = np.load( Path(__file__).resolve().parent / f'tasks/cube-{self._num_cubes}.npz')        
        self.task_infos = [
            dict(
                task_name=task_data['task_names'][i],
                init_xyzs=task_data['starts'][i],
                goal_xyzs=task_data['goals'][i],
                goal_masks=task_data['goal_masks'][i],
                return_defaults=task_data['return_defaults'][i],
                episode_lengths=task_data['episode_lengths'][i],
            )
            for i in range(len(task_data['task_names']))
        ]

        self._total_timesteps = task_data['episode_lengths'][self._task_id]


        if self._reward_task_id == 0:
            self._reward_task_id = 2  # Default task.

    def add_objects(self, arena_mjcf):
        # Add cube scene.
        cube_outer_mjcf = mjcf.from_path((self._desc_dir / 'cube_outer.xml').as_posix())
        arena_mjcf.include_copy(cube_outer_mjcf)

        # Add `num_cubes` cubes to the scene.
        distance = 0.05
        for i in range(self._num_cubes):
            cube_mjcf = mjcf.from_path((self._desc_dir / 'cube_inner.xml').as_posix())
            pos = -distance * (self._num_cubes - 1) + 2 * distance * i
            cube_mjcf.find('body', 'object_0').pos[1] = pos
            cube_mjcf.find('body', 'object_target_0').pos[1] = pos
            for tag in ['body', 'joint', 'geom', 'site']:
                for item in cube_mjcf.find_all(tag):
                    if hasattr(item, 'name') and item.name is not None and item.name.endswith('_0'):
                        item.name = item.name[:-2] + f'_{i}'
            arena_mjcf.include_copy(cube_mjcf)

        # Save cube geoms.
        self._cube_geoms_list = []
        for i in range(self._num_cubes):
            self._cube_geoms_list.append(arena_mjcf.find('body', f'object_{i}').find_all('geom'))
        self._cube_target_geoms_list = []
        for i in range(self._num_cubes):
            self._cube_target_geoms_list.append(arena_mjcf.find('body', f'object_target_{i}').find_all('geom'))

        # Add cameras.
        cameras = {
            'front': {
                'pos': (1.287, 0.000, 0.509),
                'xyaxes': (0.000, 1.000, 0.000, -0.342, 0.000, 0.940),
            },
            'front_pixels': {
                'pos': (1.053, -0.014, 0.639),
                'xyaxes': (0.000, 1.000, 0.000, -0.628, 0.001, 0.778),
            },
        }
        for camera_name, camera_kwargs in cameras.items():
            arena_mjcf.worldbody.add('camera', name=camera_name, **camera_kwargs)

    def post_compilation_objects(self):
        # Cube geom IDs.
        self._cube_geom_ids_list = [
            [self._model.geom(geom.full_identifier).id for geom in cube_geoms] for cube_geoms in self._cube_geoms_list
        ]
        self._cube_target_mocap_ids = [
            self._model.body(f'object_target_{i}').mocapid[0] for i in range(self._num_cubes)
        ]
        self._cube_target_geom_ids_list = [
            [self._model.geom(geom.full_identifier).id for geom in cube_target_geoms]
            for cube_target_geoms in self._cube_target_geoms_list
        ]

    def set_object_pose(self, object_pos, object_rpy):
        for i in range(self._num_cubes):
            self._data.joint(f'object_joint_{i}').qpos[:3] = object_pos[i]
            self._data.joint(f'object_joint_{i}').qpos[3:] = lie.SO3.from_rpy_radians(*object_rpy[i]).wxyz.tolist()
            self._data.joint(f'object_joint_{i}').qvel[:] = np.zeros(6)
        mujoco.mj_forward(self._model, self._data)

    def initialize_episode(self):
        # Set cube colors.
        for i in range(self._num_cubes):
            for gid in self._cube_geom_ids_list[i]:
                self._model.geom(gid).rgba = self._cube_colors[i]
            for gid in self._cube_target_geom_ids_list[i]:
                self._model.geom(gid).rgba[:3] = self._cube_colors[i, :3]

        self._data.qpos[self._arm_joint_ids] = self._home_qpos
        mujoco.mj_kinematics(self._model, self._data)

        # Set object positions and orientations based on the current task.
        if self._permute_blocks:
            # Randomize the order of the cubes when there are multiple cubes.
            permutation = self.np_random.permutation(self._num_cubes)
        else:
            permutation = np.arange(self._num_cubes)
        init_xyzs = self.cur_task_info['init_xyzs'].copy()[permutation]
        goal_xyzs = self.cur_task_info['goal_xyzs'].copy()[permutation]
        self._goal_masks = self.cur_task_info['goal_masks'].copy()[permutation]

        # First, force set the current scene to the goal state to obtain the goal observation.
        saved_qpos = self._data.qpos.copy()
        saved_qvel = self._data.qvel.copy()
        self.initialize_arm()
        for i in range(self._num_cubes):
            self._data.joint(f'object_joint_{i}').qpos[:3] = goal_xyzs[i]
            self._data.joint(f'object_joint_{i}').qpos[3:] = lie.SO3.identity().wxyz.tolist()
            self._data.mocap_pos[self._cube_target_mocap_ids[i]] = goal_xyzs[i]
            self._data.mocap_quat[self._cube_target_mocap_ids[i]] = lie.SO3.identity().wxyz.tolist()
        mujoco.mj_forward(self._model, self._data)

        # Do a few random steps to make the scene stable.
        for _ in range(2):
            self.step(self.action_space.sample())

        # Save the goal observation.
        self._cur_goal_ob = (
            self.compute_oracle_observation() if self._use_oracle_rep else self.compute_observation()
        )
        if self._render_goal:
            self._cur_goal_rendered = self.render()
        else:
            self._cur_goal_rendered = None

        # Now, do the actual reset.
        self._data.qpos[:] = saved_qpos
        self._data.qvel[:] = saved_qvel
        self.initialize_arm()
        for i in range(self._num_cubes):
            # Set cube position and orientation from the task init configuration.
            obj_pos = init_xyzs[i].copy()
            # obj_pos[:2] += self.np_random.uniform(-0.01, 0.01, size=2)
            self._data.joint(f'object_joint_{i}').qpos[:3] = obj_pos
            yaw = self.np_random.uniform(0, 2 * np.pi) * 0  # Yaw randomization disabled; cubes start upright.
            obj_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
            self._data.joint(f'object_joint_{i}').qpos[3:] = obj_ori
            self._data.mocap_pos[self._cube_target_mocap_ids[i]] = goal_xyzs[i]
            self._data.mocap_quat[self._cube_target_mocap_ids[i]] = lie.SO3.identity().wxyz.tolist()

        # Forward kinematics to update site positions.
        self.pre_step()
        mujoco.mj_forward(self._model, self._data)
        self.post_step()

        self._success = False

    def _compute_end_effector_success(self):
        return_default = self.cur_task_info['return_defaults']

        if return_default:
            eff_pos = self._data.site_xpos[self._pinch_site_id].copy()
            return np.linalg.norm(eff_pos - self._neutral_arm_position) <= 0.05
        else:
            return True

    def _compute_successes(self):
        """Compute object successes."""
        cube_successes = []
        for i in range(self._num_cubes):
            obj_pos = self._data.joint(f'object_joint_{i}').qpos[:3]
            tar_pos = self._data.mocap_pos[self._cube_target_mocap_ids[i]]
            tar_mask = self._goal_masks[i]
            if np.linalg.norm(obj_pos - tar_pos) <= self._success_threshold or not tar_mask:
                cube_successes.append(True)
            else:
                cube_successes.append(False)
        
        eff_success = self._compute_end_effector_success()
        return cube_successes, eff_success
    
    def _compute_successes_easy(self):
        """Compute object successes."""
        cube_successes = []
        for i in range(self._num_cubes):
            obj_pos = self._data.joint(f'object_joint_{i}').qpos[:3]
            tar_pos = self._data.mocap_pos[self._cube_target_mocap_ids[i]]
            tar_mask = self._goal_masks[i]
            if np.linalg.norm(obj_pos - tar_pos) <= self._easy_success_threshold or not tar_mask:
                cube_successes.append(True)
            else:
                cube_successes.append(False)

        return cube_successes

    def post_step(self):
        # Check if the cubes are in the target positions.
        cube_successes, eff_success = self._compute_successes()
        self._success = all(cube_successes) and eff_success

        # Adjust the colors of the cubes based on success.
        for i in range(self._num_cubes):
            if self._visualize_info and self._goal_masks[i]:
                for gid in self._cube_target_geom_ids_list[i]:
                    self._model.geom(gid).rgba[3] = 0.2
            else:
                for gid in self._cube_target_geom_ids_list[i]:
                    self._model.geom(gid).rgba[3] = 0.0

            if self._visualize_info and cube_successes[i] and self._goal_masks[i]:
                for gid in self._cube_geom_ids_list[i]:
                    self._model.geom(gid).rgba[:3] = self._cube_success_colors[i, :3]
            else:
                for gid in self._cube_geom_ids_list[i]:
                    self._model.geom(gid).rgba[:3] = self._cube_colors[i, :3]

    def add_gripper_info(self, ob_info):
        eff_success = self._compute_end_effector_success()
        ob_info['privileged/effector_success'] = np.array([eff_success], dtype=bool)

    def add_object_info(self, ob_info):

        successes, _ = self._compute_successes()
        easy_success = self._compute_successes_easy()
        # Cube positions and orientations.
        for i in range(self._num_cubes):
            ob_info[f'privileged/block_{i}_pos'] = self._data.joint(f'object_joint_{i}').qpos[:3].copy()
            ob_info[f'privileged/block_{i}_quat'] = self._data.joint(f'object_joint_{i}').qpos[3:].copy()
            ob_info[f'privileged/block_{i}_yaw'] = np.array(
                [lie.SO3(wxyz=self._data.joint(f'object_joint_{i}').qpos[3:]).compute_yaw_radians()]
            )

            obj_pos = self._data.joint(f'object_joint_{i}').qpos[:3]
            tar_pos = self._data.mocap_pos[self._cube_target_mocap_ids[i]]
            ob_info[f'privileged/block_{i}_success'] = np.array([successes[i]], dtype=bool)
            ob_info[f'privileged/block_{i}_easy_success'] = np.array([easy_success[i]], dtype=bool)

    def compute_observation(self):
        if self._ob_type == 'pixels':
            return self.get_pixel_observation()
        else:
            xyz_center = np.array([0.425, 0.0, 0.0])
            xyz_scaler = 10.0
            gripper_scaler = 3.0

            ob_info = self.compute_ob_info()
            ob = [
                ob_info['proprio/joint_pos'],
                ob_info['proprio/joint_vel'],
                (ob_info['proprio/effector_pos'] - xyz_center) * xyz_scaler,
                np.cos(ob_info['proprio/effector_yaw']),
                np.sin(ob_info['proprio/effector_yaw']),
                ob_info['proprio/gripper_opening'] * gripper_scaler,
                ob_info['proprio/gripper_contact'],
            ]
            for i in range(self._num_cubes):
                ob.extend(
                    [
                        (ob_info[f'privileged/block_{i}_pos'] - xyz_center) * xyz_scaler,
                        ob_info[f'privileged/block_{i}_quat'],
                        np.cos(ob_info[f'privileged/block_{i}_yaw']),
                        np.sin(ob_info[f'privileged/block_{i}_yaw']),
                    ]
                )

            return np.concatenate(ob)

    def compute_oracle_observation(self):
        """Return the oracle goal representation of the current state."""
        xyz_center = np.array([0.425, 0.0, 0.0])
        xyz_scaler = 10.0

        ob_info = self.compute_ob_info()
        ob = []
        for i in range(self._num_cubes):
            ob.append((ob_info[f'privileged/block_{i}_pos'] - xyz_center) * xyz_scaler)

        return np.concatenate(ob)

    def compute_reward(self):
        if self._reward_task_id is None:
            return super().compute_reward()

        # Compute the reward based on the task.
        successes, _ = self._compute_successes()
        reward = float(sum(successes) - self._num_cubes)
        return reward
    
    def truncate_episode(self) -> bool:
        return self._data.time >= (self._total_timesteps * self._control_timestep)