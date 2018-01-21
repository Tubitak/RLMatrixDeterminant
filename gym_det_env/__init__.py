from gym.envs.registration import register

register(
    id='GymDet-v0',
    entry_point='gym_det_env.env:Determinant_v0',
)

register(
    id='GymDet-v1',
    entry_point='gym_det_env.env:Determinant_v1',
)

register(
    id='GymDet-v2',
    entry_point='gym_det_env.env:Determinant_v2',
)
