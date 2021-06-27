from gym.envs.registration import register

register(
    id='bipedal-highjump-v0',
    entry_point='custom_env.bipedal_walker_derivative_classes:BipedalHighJump',
    max_episode_steps=2000,
    reward_threshold=100,
)

register(
    id='bipedal-highjumpfixed-v0',
    entry_point='custom_env.bipedal_walker_derivative_classes:BipedalHighJumpFixed',
    max_episode_steps=2000,
    reward_threshold=100,
)

register(
    id='bipedal-longjump-v0',
    entry_point='custom_env.bipedal_walker_derivative_classes:BipedalLongJump',
    max_episode_steps=2000,
    reward_threshold=100,
)

register(
    id='bipedal-shortsprint-v0',
    entry_point='custom_env.bipedal_walker_derivative_classes:BipedalShortSprint',
    max_episode_steps=2000,
    reward_threshold=100,
)

register(
    id='bipedal-mediumrun-v0',
    entry_point='custom_env.bipedal_walker_derivative_classes:BipedalMediumRun',
    max_episode_steps=2000,
    reward_threshold=100,
)

register(
    id='bipedal-marathon-v0',
    entry_point='custom_env.bipedal_walker_derivative_classes:BipedalMarathon',
    max_episode_steps=2000,
    reward_threshold=100,
)

register(
    id='bipedal-hurdles-v0',
    entry_point='custom_env.bipedal_walker_derivative_classes:BipedalHurdles',
    max_episode_steps=2000,
    reward_threshold=100,
)

register(
    id='bipedal-sparsehurdles-v0',
    entry_point='custom_env.bipedal_walker_derivative_classes:BipedalSparseHurdles',
    max_episode_steps=2000,
    reward_threshold=100,
)

register(
    id='bipedal-nofall-v0',
    entry_point='custom_env.bipedal_walker_derivative_classes:BipedalHurdlesNoFall',
    max_episode_steps=2000,
    reward_threshold=100,
)


register(
    id='leftright-discrete-v0',
    entry_point='custom_env.left_right_envs:DiscreteLeftRightEnv',
    max_episode_steps=50,
    reward_threshold=100,
)

register(
    id='leftright-continuous-v0',
    entry_point='custom_env.left_right_envs:ContinuousLeftRightEnv',
    max_episode_steps=50,
    reward_threshold=100,
)

register(
    id='acrobot-custom-v0',
    entry_point='custom_env.acrobot_py3:Acrobot',
    max_episode_steps=500,
    reward_threshold=-100.0,
)

register(
    id='pendulum-custom-v0',
    entry_point='custom_env.pendulum:PendulumEnv',
    max_episode_steps=500,
    reward_threshold=-100.0,
)

register(
    id='mountaincar-custom-v0',
    entry_point='custom_env.mountain_car:MountainCarEnv',
    max_episode_steps=200,
    reward_threshold=-110.0,
)

register(
    id='gridworld-v0',
    entry_point='custom_env.GridWorld:GridWorld',
    max_episode_steps=50,
    reward_threshold=100,
)

register(
    id='corner-gridworld-v0',
    entry_point='custom_env.GridWorld:CornerGridWorld',
    max_episode_steps=50,
    reward_threshold=100,
)

register(
    id='bipedal-walker-continuous-v0',
    entry_point='custom_env.bipedal_walker_continuous:BipedalWalkerContinuous',
    max_episode_steps=2000,
    reward_threshold=300,
)