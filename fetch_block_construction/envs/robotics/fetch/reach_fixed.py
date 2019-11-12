import os
from gym import utils
from fetch_block_construction.envs.robotics import fetch_env


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'reach.xml')


class FetchReachFixedGoalEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', obs_type='np', render_size=42):
        initial_qpos = {
            'robot0:slide0': 0.4049,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=False, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, obs_type=obs_type, render_size=render_size)
        utils.EzPickle.__init__(self, reward_type=reward_type, obs_type=obs_type, render_size=render_size)

        # Call this one time so we don't have to ever call it in the reset
        self.goal = self._sample_goal().copy()

    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        obs = self._get_obs()
        return obs

