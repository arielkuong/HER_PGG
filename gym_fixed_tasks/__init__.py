from gym.envs.registration import registry, register, make, spec


# Hand environments

# Hand egg rotate

register(
    id='HandEggRotateFixedTaskEasy-v0',
    entry_point='gym_customized.envs:HandEggEnv',
    kwargs={'target_position': 'ignore', 'target_rotation': 'fixed_easy', 'reward_type': 'sparse'},
    max_episode_steps=100,
)

register(
    id='HandEggRotateFixedTaskEasy-dense-v0',
    entry_point='gym_customized.envs:HandEggEnv',
    kwargs={'target_position': 'ignore', 'target_rotation': 'fixed_easy', 'reward_type': 'dense'},
    max_episode_steps=100,
)

register(
    id='HandEggRotateFixedTaskHard-v0',
    entry_point='gym_customized.envs:HandEggEnv',
    kwargs={'target_position': 'ignore', 'target_rotation': 'fixed', 'reward_type': 'sparse'},
    max_episode_steps=100,
)

register(
    id='HandEggRotateFixedTaskHard-dense-v0',
    entry_point='gym_customized.envs:HandEggEnv',
    kwargs={'target_position': 'ignore', 'target_rotation': 'fixed', 'reward_type': 'dense'},
    max_episode_steps=100,
)


# Hand block rotate
register(
    id='HandManipulateBlockRotateZ-customized-v0',
    entry_point='gym_customized.envs:HandBlockEnv',
    kwargs={'target_position': 'ignore', 'target_rotation': 'z',  'reward_type': 'sparse'},
    max_episode_steps=100,
)

register(
    id='HandBlockRotateFixedTaskEasy-v0',
    entry_point='gym_customized.envs:HandBlockEnv',
    kwargs={'target_position': 'ignore', 'target_rotation': 'fixed_easy', 'reward_type': 'sparse'},
    max_episode_steps=100,
)

register(
    id='HandBlockRotateFixedTaskEasy-dense-v0',
    entry_point='gym_customized.envs:HandBlockEnv',
    kwargs={'target_position': 'ignore', 'target_rotation': 'fixed_easy', 'reward_type': 'sparse'},
    max_episode_steps=100,
)

register(
    id='HandBlockRotateFixedTaskHard-dense-v0',
    entry_point='gym_customized.envs:HandBlockEnv',
    kwargs={'target_position': 'ignore', 'target_rotation': 'fixed', 'reward_type': 'dense'},
    max_episode_steps=100,
)

# Fetch environments

register(
    id='FetchPushFixedTask-v1',
    entry_point='gym_customized.envs:FetchPushEnv_FixedTask',
    max_episode_steps=50,
)

register(
    id='FetchPushFixedTask-dense-v1',
    entry_point='gym_customized.envs:FetchPushEnv_FixedTask',
    kwargs={'reward_type': 'dense'},
    max_episode_steps=50,
)

register(
    id='FetchPickAndPlaceFixedTask-v1',
    entry_point='gym_customized.envs:FetchPickAndPlaceEnv_FixedTask',
    max_episode_steps=50,
)

register(
    id='FetchPickAndPlaceFixedTask-dense-v1',
    entry_point='gym_customized.envs:FetchPickAndPlaceEnv_FixedTask',
    kwargs={'reward_type': 'dense'},
    max_episode_steps=50,
)

register(
    id='FetchSlideFixedTask-v1',
    entry_point='gym_customized.envs:FetchSlideEnv_FixedTask',
    max_episode_steps=50,
)

register(
    id='FetchSlideFixedTask-dense-v1',
    entry_point='gym_customized.envs:FetchSlideEnv_FixedTask',
    kwargs={'reward_type': 'dense'},
    max_episode_steps=50,
)
