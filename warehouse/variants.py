from warehouse.core import Warehouse


__all__ = ["WarehouseSmall", "WarehouseMedium", "WarehouseLarge"]


class WarehouseSmall(Warehouse):
    def __init__(self) -> None:
        self.num_agents: int = 2
        self.num_requests: int = 2
        self.area_dimension: int = 8

        super(WarehouseSmall, self).__init__(
            num_agents=self.num_agents,
            num_requests=self.num_requests,
            area_dimension=self.area_dimension,
            pickup_racks_arrangement=[4],
            episode_duration=600,
            pickup_wait_duration=40,
        )


class WarehouseMedium(Warehouse):
    def __init__(self) -> None:
        self.num_agents: int = 4
        self.num_requests: int = 4
        self.area_dimension: int = 12

        super(WarehouseMedium, self).__init__(
            num_agents=self.num_agents,
            num_requests=self.num_requests,
            area_dimension=self.area_dimension,
            pickup_racks_arrangement=[4, 8],
            episode_duration=600,
            pickup_wait_duration=40,
        )


class WarehouseLarge(Warehouse):
    def __init__(self) -> None:
        self.num_agents: int = 16
        self.num_requests: int = 16
        self.area_dimension: int = 20

        super(WarehouseLarge, self).__init__(
            num_agents=self.num_agents,
            num_requests=self.num_requests,
            area_dimension=self.area_dimension,
            pickup_racks_arrangement=[4, 8, 12, 16],
            episode_duration=600,
            pickup_wait_duration=40,
        )
