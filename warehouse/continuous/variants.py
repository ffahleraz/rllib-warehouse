from warehouse.continuous.core import WarehouseContinuous


__all__ = ["WarehouseContinuousSmall"]


class WarehouseContinuousSmall(WarehouseContinuous):
    NUM_AGENTS: int = 2
    NUM_REQUESTS: int = 2
    WORLD_DIMENSION: float = 10.0
    BORDER_WIDTH: float = 1.0

    def __init__(self) -> None:
        super(WarehouseContinuousSmall, self).__init__(
            num_agents=WarehouseContinuousSmall.NUM_AGENTS,
            num_requests=WarehouseContinuousSmall.NUM_REQUESTS,
            world_dimension=WarehouseContinuousSmall.WORLD_DIMENSION,
            border_width=WarehouseContinuousSmall.BORDER_WIDTH,
            agent_init_positions=[
                [3.0, 3.0],
                [
                    WarehouseContinuousSmall.WORLD_DIMENSION - 3.0,
                    WarehouseContinuousSmall.WORLD_DIMENSION - 3.0,
                ],
            ],
            pickup_racks_arrangement=[5.0],
            episode_duration_s=160,
            pickup_wait_duration_s=40,
        )
