import os
from gym import utils
from fetch_block_construction.envs.robotics import fetch_env
from gym.core import Wrapper
import numpy as np

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'reach.xml')


class FixedGoalWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env.unwrapped.goal = self.env.unwrapped._sample_goal().copy()
        if self.env.unwrapped.has_object:
            self.object_qpos = self.randomize_object_position()

    def reset(self, **kwargs):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self.env.unwrapped._reset_sim()
        obs = self.env.unwrapped._get_obs()
        # return obs
        return self.env.reset(**kwargs)

    def randomize_object_position(self):
        object_xpos = self.env.unwrapped.initial_gripper_xpos[:2]
        while np.linalg.norm(object_xpos - self.env.unwrapped.initial_gripper_xpos[:2]) < 0.1:
            object_xpos = self.env.unwrapped.initial_gripper_xpos[:2] + self.env.unwrapped.np_random.uniform(
                -self.env.unwrapped.obj_range, self.env.unwrapped.obj_range, size=2)
        object_qpos = self.env.unwrapped.sim.data.get_joint_qpos('object0:joint')
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        return object_qpos

    def _reset_sim(self):
        self.env.unwrapped.sim.set_state(self.env.unwrapped.initial_state)

        # Randomize start position of object.
        if self.env.unwrapped.has_object:
            self.env.unwrapped.sim.data.set_joint_qpos('object0:joint', self.object_qpos)
        self.env.unwrapped.sim.forward()
        return True

    def step(self, action):
        return self.env.step(action)

