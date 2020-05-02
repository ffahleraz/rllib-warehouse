from typing import Dict, Tuple

import numpy as np
import gym

from warehouse.core import Warehouse


__all__ = [
    "WarehouseSmall",
    "WarehouseSmallRandom",
    "WarehouseMedium",
    "WarehouseMediumRandom",
    "WarehouseLarge",
    "WarehouseLargeRandom",
]


class WarehouseSmall(Warehouse):
    max_num_agents = 4

    def __init__(self, num_agents: int) -> None:
        self.max_num_agents = 4
        assert 1 <= num_agents <= self.max_num_agents
        super(WarehouseSmall, self).__init__(
            num_agents=num_agents,
            num_requests=4,
            area_dimension=12,
            pickup_racks_arrangement=[4, 8],
            episode_duration=240,
            pickup_wait_duration=240,
        )


class WarehouseMedium(Warehouse):
    max_num_agents = 9

    def __init__(self, num_agents: int) -> None:
        assert 1 <= num_agents <= self.max_num_agents
        super(WarehouseMedium, self).__init__(
            num_agents=num_agents,
            num_requests=9,
            area_dimension=16,
            pickup_racks_arrangement=[4, 8, 12],
            episode_duration=320,
            pickup_wait_duration=320,
        )


class WarehouseLarge(Warehouse):
    max_num_agents = 16

    def __init__(self, num_agents: int) -> None:
        assert 1 <= num_agents <= self.max_num_agents
        super(WarehouseLarge, self).__init__(
            num_agents=num_agents,
            num_requests=16,
            area_dimension=20,
            pickup_racks_arrangement=[4, 8, 12, 16],
            episode_duration=400,
            pickup_wait_duration=400,
        )


class WarehouseSmallRandom(WarehouseSmall):
    def __init__(self) -> None:
        super(WarehouseSmallRandom, self).__init__(num_agents=self._get_random_num_agents())

    def reset(self) -> Dict[str, gym.spaces.Dict]:
        super(WarehouseSmallRandom, self).__init__(num_agents=self._get_random_num_agents())
        return super(WarehouseSmallRandom, self).reset()

    def _get_random_num_agents(self) -> int:
        return np.random.randint(1, WarehouseSmallRandom.max_num_agents + 1)


class WarehouseMediumRandom(WarehouseMedium):
    def __init__(self) -> None:
        super(WarehouseMediumRandom, self).__init__(num_agents=self._get_random_num_agents())

    def reset(self) -> Dict[str, gym.spaces.Dict]:
        super(WarehouseMediumRandom, self).__init__(num_agents=self._get_random_num_agents())
        return super(WarehouseMediumRandom, self).reset()

    def _get_random_num_agents(self) -> int:
        return np.random.randint(1, WarehouseMediumRandom.max_num_agents + 1)


class WarehouseLargeRandom(WarehouseLarge):
    def __init__(self) -> None:
        super(WarehouseLargeRandom, self).__init__(num_agents=self._get_random_num_agents())

    def reset(self) -> Dict[str, gym.spaces.Dict]:
        super(WarehouseLargeRandom, self).__init__(num_agents=self._get_random_num_agents())
        return super(WarehouseLargeRandom, self).reset()

    def _get_random_num_agents(self) -> int:
        return np.random.randint(1, WarehouseLargeRandom.max_num_agents + 1)
