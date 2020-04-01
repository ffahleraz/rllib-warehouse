from warehouse.discrete.core import WarehouseDiscrete


__all__ = ["WarehouseDiscreteSmall"]


class WarehouseDiscreteSmall(WarehouseDiscrete):
    NUM_AGENTS: int = 2
    NUM_REQUESTS: int = 2
    AREA_DIMENSION: int = 8

    def __init__(self) -> None:
        super(WarehouseDiscreteSmall, self).__init__(
            num_agents=WarehouseDiscreteSmall.NUM_AGENTS,
            num_requests=WarehouseDiscreteSmall.NUM_REQUESTS,
            area_dimension=WarehouseDiscreteSmall.AREA_DIMENSION,
            agent_init_positions=[
                [2, 2],
                [
                    WarehouseDiscreteSmall.AREA_DIMENSION - 2,
                    WarehouseDiscreteSmall.AREA_DIMENSION - 2,
                ],
            ],
            pickup_racks_arrangement=[4],
            episode_duration=400,
            pickup_wait_duration=40,
        )
