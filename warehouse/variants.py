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
            agent_init_positions=[[2, 2], [6, 6]],
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
            agent_init_positions=[[2, 2], [2, 10], [10, 2], [10, 10]],
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
            agent_init_positions=[
                [2, 2],
                [2, 6],
                [2, 10],
                [2, 14],
                [2, 18],
                [18, 2],
                [18, 6],
                [18, 10],
                [18, 14],
                [18, 18],
                [6, 2],
                [10, 2],
                [14, 2],
                [6, 18],
                [10, 18],
                [14, 18],
            ],
            pickup_racks_arrangement=[4, 8, 12, 16],
            episode_duration=600,
            pickup_wait_duration=40,
        )
