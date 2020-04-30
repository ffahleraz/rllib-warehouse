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
    def __init__(self, num_agents: int) -> None:
        assert 1 <= num_agents <= 4
        super(WarehouseSmall, self).__init__(
            num_agents=num_agents,
            num_requests=4,
            area_dimension=12,
            pickup_racks_arrangement=[4, 8],
            episode_duration=240,
            pickup_wait_duration=24,
        )


class WarehouseSmallRandom(WarehouseSmall):
    def __init__(self) -> None:
        super(WarehouseSmallRandom, self).__init__(num_agents=self._get_random_num_agents())

    def reset(self) -> Dict[str, gym.spaces.Dict]:
        super(WarehouseSmallRandom, self).__init__(num_agents=self._get_random_num_agents())
        return super(WarehouseSmallRandom, self).reset()

    def _get_random_num_agents(self) -> int:
        return np.random.randint(1, 5)


class WarehouseMedium(Warehouse):
    def __init__(self, num_agents: int) -> None:
        assert 1 <= num_agents <= 9
        super(WarehouseMedium, self).__init__(
            num_agents=num_agents,
            num_requests=9,
            area_dimension=16,
            pickup_racks_arrangement=[4, 8, 12],
            episode_duration=320,
            pickup_wait_duration=32,
        )


class WarehouseMediumRandom(WarehouseMedium):
    def __init__(self) -> None:
        super(WarehouseMediumRandom, self).__init__(num_agents=self._get_random_num_agents())

    def reset(self) -> Dict[str, gym.spaces.Dict]:
        super(WarehouseMediumRandom, self).__init__(num_agents=self._get_random_num_agents())
        return super(WarehouseMediumRandom, self).reset()

    def _get_random_num_agents(self) -> int:
        return np.random.randint(1, 10)


class WarehouseLarge(Warehouse):
    def __init__(self, num_agents: int) -> None:
        assert 1 <= num_agents <= 16
        super(WarehouseLarge, self).__init__(
            num_agents=num_agents,
            num_requests=16,
            area_dimension=20,
            pickup_racks_arrangement=[4, 8, 12, 16],
            episode_duration=400,
            pickup_wait_duration=40,
        )


class WarehouseLargeRandom(WarehouseLarge):
    def __init__(self) -> None:
        super(WarehouseLargeRandom, self).__init__(num_agents=self._get_random_num_agents())

    def reset(self) -> Dict[str, gym.spaces.Dict]:
        super(WarehouseLargeRandom, self).__init__(num_agents=self._get_random_num_agents())
        return super(WarehouseLargeRandom, self).reset()

    def _get_random_num_agents(self) -> int:
        return np.random.randint(1, 17)
