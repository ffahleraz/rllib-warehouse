from warehouse.core import Warehouse


__all__ = [
    "Warehouse2",
    "Warehouse4",
    "Warehouse6",
    "Warehouse8",
    "Warehouse10",
    "Warehouse12",
    "Warehouse14",
    "Warehouse16",
]


class Warehouse2(Warehouse):
    def __init__(self) -> None:
        super(Warehouse2, self).__init__(
            num_agents=2,
            num_requests=2,
            area_dimension=20,
            pickup_racks_arrangement=[4, 8, 12, 16],
            episode_duration=400,
            pickup_wait_duration=40,
        )


class Warehouse4(Warehouse):
    def __init__(self) -> None:
        super(Warehouse4, self).__init__(
            num_agents=4,
            num_requests=4,
            area_dimension=20,
            pickup_racks_arrangement=[4, 8, 12, 16],
            episode_duration=400,
            pickup_wait_duration=40,
        )


class Warehouse6(Warehouse):
    def __init__(self) -> None:
        super(Warehouse6, self).__init__(
            num_agents=6,
            num_requests=6,
            area_dimension=20,
            pickup_racks_arrangement=[4, 8, 12, 16],
            episode_duration=400,
            pickup_wait_duration=40,
        )


class Warehouse8(Warehouse):
    def __init__(self) -> None:
        super(Warehouse8, self).__init__(
            num_agents=8,
            num_requests=8,
            area_dimension=20,
            pickup_racks_arrangement=[4, 8, 12, 16],
            episode_duration=400,
            pickup_wait_duration=40,
        )


class Warehouse10(Warehouse):
    def __init__(self) -> None:
        super(Warehouse10, self).__init__(
            num_agents=10,
            num_requests=10,
            area_dimension=20,
            pickup_racks_arrangement=[4, 8, 12, 16],
            episode_duration=400,
            pickup_wait_duration=40,
        )


class Warehouse12(Warehouse):
    def __init__(self) -> None:
        super(Warehouse12, self).__init__(
            num_agents=12,
            num_requests=12,
            area_dimension=20,
            pickup_racks_arrangement=[4, 8, 12, 16],
            episode_duration=400,
            pickup_wait_duration=40,
        )


class Warehouse14(Warehouse):
    def __init__(self) -> None:
        super(Warehouse14, self).__init__(
            num_agents=14,
            num_requests=14,
            area_dimension=20,
            pickup_racks_arrangement=[4, 8, 12, 16],
            episode_duration=400,
            pickup_wait_duration=40,
        )


class Warehouse16(Warehouse):
    def __init__(self) -> None:
        super(Warehouse16, self).__init__(
            num_agents=16,
            num_requests=16,
            area_dimension=20,
            pickup_racks_arrangement=[4, 8, 12, 16],
            episode_duration=400,
            pickup_wait_duration=40,
        )
