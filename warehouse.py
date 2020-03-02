import typing

import numpy as np
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv


N_AGENTS: int = 10
N_PICKUP_POINTS: int = 10
N_DELIVERY_POINTS: int = 10
AREA_DIMENSION: float = 10.0


class Warehouse(MultiAgentEnv):
    metadata = {"render.modes": ["human"]}
    reward_range = (-np.inf, -np.inf)
    spec = None
    action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    observation_space = spaces.Dict(
        {
            "self_position": spaces.Box(
                low=AREA_DIMENSION, high=AREA_DIMENSION, shape=(2,), dtype=np.float32,
            ),
            "pickup_points": spaces.Dict(
                {
                    "availability": spaces.MultiBinary(N_PICKUP_POINTS),
                    "positions": spaces.Box(
                        low=AREA_DIMENSION,
                        high=AREA_DIMENSION,
                        shape=(N_PICKUP_POINTS, 2),
                        dtype=np.float32,
                    ),
                }
            ),
            "delivery_points": spaces.Dict(
                {
                    "availability": spaces.MultiBinary(N_DELIVERY_POINTS),
                    "positions": spaces.Box(
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

        self._counter = 1

    def reset(self) -> typing.Dict[str, spaces.Dict]:
        self._counter = 1

        return {f"{i}": self.observation_space.sample() for i in range(N_AGENTS)}

    def step(
        self, action_dict: typing.Dict[str, spaces.Box]
    ) -> typing.Tuple[
        typing.Dict[str, spaces.Dict],
        typing.Dict[str, float],
        typing.Dict[str, bool],
        typing.Dict[str, typing.Dict[str, str]],
    ]:
        self._counter += 1

        observations = {
            f"{i}": self.observation_space.sample() for i in range(N_AGENTS)
        }
        rewards = {f"{i}": 0.0 for i in range(N_AGENTS)}
        dones = {f"{i}": False for i in range(N_AGENTS)}
        dones["__all__"] = self._counter % 10 == 0
        infos = {f"{i}": {"test": "test"} for i in range(N_AGENTS)}
        return observations, rewards, dones, infos

    def render(self, mode: str = "human") -> None:
        if mode != "human":
            super(Warehouse, self).render(mode=mode)

        pass

    def close(self) -> None:
        pass
