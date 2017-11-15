from gym.envs.registration import register

register(
    id='SingleVideoEnv-v0',
    entry_point='SVE.SVE:SingleVideoEnv',
)