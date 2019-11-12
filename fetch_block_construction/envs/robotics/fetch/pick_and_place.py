import os
from gym import utils
from fetch_block_construction.envs.robotics import fetch_env
import numpy as np

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'pick_and_place.xml')


class FetchPickAndPlaceEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', obs_type='np', render_size=42):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, obs_type=obs_type, render_size=render_size)
        utils.EzPickle.__init__(self, reward_type, obs_type, render_size)


class FetchPickAndPlaceFixedEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='dense', obs_type='np', render_size=42):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, obs_type=obs_type, render_size=render_size)
        utils.EzPickle.__init__(self, reward_type, obs_type, render_size)

        self.goal = self._sample_goal().copy()
        self.object_qpos = self.randomize_object_position()

    def reset(self, **kwargs):
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        obs = self._get_obs()
        return obs

    def randomize_object_position(self):
        object_xpos = self.initial_gripper_xpos[:2]
        while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
            object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                -self.obj_range, self.obj_range, size=2)
        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        return object_qpos

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        self.sim.data.set_joint_qpos('object0:joint', self.object_qpos)
        self.sim.forward()
        return True

