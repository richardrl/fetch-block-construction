import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)


for num_blocks in range(1, 25):
    for reward_type in ['sparse', 'dense', 'incremental', 'block1only']:
        for obs_type in ['dictimage', 'np', 'dictstate']:
            for render_size in [42, 84]:
                for stack_only in [True, False]:
                    for case in ["Singletower", "Pyramid", "Multitower", "All"]:
                        initial_qpos = {
                            'robot0:slide0': 0.405,
                            'robot0:slide1': 0.48,
                            'robot0:slide2': 0.0,
                            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
                        }

                        for i in range(num_blocks):
                            initial_qpos[F"object{i}:joint"] = [1.25, 0.53, .4 + i*.06, 1., 0., 0., 0.]
                        kwargs = {
                            'reward_type': reward_type,
                            'initial_qpos': initial_qpos,
                            'num_blocks': num_blocks,
                            'obs_type': obs_type,
                            'render_size': render_size,
                            'stack_only': stack_only,
                            'case': case
                        }

                        register(
                            id='FetchBlockConstruction_{}Blocks_{}Reward_{}Obs_{}Rendersize_{}Stackonly_{}Case-v1'.format(*[kwarg.title() if isinstance(kwarg, str) else kwarg for kwarg in [num_blocks, reward_type, obs_type, render_size, stack_only, case]]),
                            entry_point='fetch_block_construction.envs.robotics:FetchBlockConstructionEnv',
                            kwargs=kwargs,
                            max_episode_steps=50 * num_blocks,
                        )