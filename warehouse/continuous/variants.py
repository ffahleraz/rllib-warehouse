from warehouse.continuous.core import WarehouseContinuous


__all__ = ["WarehouseContinuousSmall"]


class WarehouseContinuousSmall(WarehouseContinuous):
    def __init__(self) -> None:
        self.num_agents: int = 2
        self.num_requests: int = 2
        self.world_dimension: float = 10.0
        self.border_width: float = 1.0

        super(WarehouseContinuousSmall, self).__init__(
            num_agents=self.num_agents,
            num_requests=self.num_requests,
            world_dimension=self.world_dimension,
            border_width=self.border_width,
            agent_init_positions=[
                [3.0, 3.0],
                [self.world_dimension - 3.0, self.world_dimension - 3.0,],
            ],
            pickup_racks_arrangement=[5.0],
            episode_duration_s=160,
            pickup_wait_duration_s=40,
        )
