import gym
import numpy as np
from gym.wrappers.monitor import Monitor
import fetch_block_construction
env = gym.make('FetchBlockConstruction_17Blocks_IncrementalReward_DictstateObs_42Rendersize_TrueStackonly_AllCase-v1')
env = Monitor(env, directory="videos", force=True, video_callable=lambda x: x)

env.env._max_episode_steps = 50
# env.env.seed(0)

env.reset()
env.env.stack_only = True

step=0
while True:
    obs, done =env.reset(), False
    while not done:
        # env.render()
        action = np.asarray([0, 0, 0, 0])
        step_results = env.step(action)
        obs, reward, done, info = step_results
        print("Reward: {} Info: {}".format(reward, info))
        if done:
            step = 0
        step+=1
        print(step)