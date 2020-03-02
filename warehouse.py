import typing

import gym
import numpy as np


N_AGENTS: int = 10
N_PICKUP_POINTS: int = 10
N_DELIVERY_POINTS: int = 10
AREA_DIMENSION: float = 10.0


class Warehouse(gym.Env):
    metadata = {"render.modes": ["human"]}
    reward_range = (-np.inf, -np.inf)
    spec = None
    action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    observation_space = gym.spaces.Dict(
        {
            "self_position": gym.spaces.Box(
                low=AREA_DIMENSION, high=AREA_DIMENSION, shape=(2,), dtype=np.float32,
            ),
            "pickup_points": gym.spaces.Dict(
                {
                    "availability": gym.spaces.MultiBinary(N_PICKUP_POINTS),
                    "positions": gym.spaces.Box(
                        low=AREA_DIMENSION,
                        high=AREA_DIMENSION,
                        shape=(N_PICKUP_POINTS, 2),
                        dtype=np.float32,
                    ),
                }
            ),
            "delivery_points": gym.spaces.Dict(
                {
                    "availability": gym.spaces.MultiBinary(N_DELIVERY_POINTS),
                    "positions": gym.spaces.Box(
                        low=AREA_DIMENSION,
                        high=AREA_DIMENSION,
                        shape=(N_DELIVERY_POINTS, 2),
                        dtype=np.float32,
                    ),
                }
            ),
        }
    )

    def __init__(self) -> None:
        super(Warehouse, self).__init__()

    def step(
        self, action: gym.spaces.Box
    ) -> typing.Tuple[gym.spaces.Dict, float, bool, typing.Dict[str, str]]:
        return self.observation_space.sample(), 0.0, False, {}

    def reset(self) -> gym.spaces.Dict:
        return self.observation_space.sample()

    def render(self, mode: str = "human") -> None:
        if mode != "human":
            super(Warehouse, self).render(mode=mode)

        pass

    def close(self) -> None:
        pass
