import time
import typing

import numpy as np
import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import Box2D
from Box2D.b2 import (
    world,
    circleShape,
    polygonShape,
    dynamicBody,
)

# Engineering notes:
#   - Each agent is identified by int[0, NUM_AGENTS) in string type.
#   - Zero coordinate for the env is on the bottom left, this is then transformed to
#     top left for rendering.
#   - Pickup point states:
#       - -2: inactive
#       - -1: active, waiting pickup
#       - [0, NUM_AGENTS): pickup valid, timer started, number indicates carrier agent
#   - Delivery point states:
#       - -2: inactive
#       - -1: active, waiting pickup
#       - [0, NUM_AGENTS): carrier agent
#   - Environment layout:
#       |B|x| | |x|x| | |x|x| | |x|x| | |x|B|

# Environment
AREA_DIMENSION_M: float = 16.0
BORDER_WIDTH_M: float = 1.0
WORLD_DIMENSION_M: float = AREA_DIMENSION_M + 2 * BORDER_WIDTH_M
AGENT_RADIUS: float = 0.3
PICKUP_RACKS_ARRANGEMENT: typing.List[float] = [5.0, 9.0, 13.0]
FRAMES_PER_SECOND: int = 20

NUM_AGENTS: int = (len(PICKUP_RACKS_ARRANGEMENT) + 1) ** 2
NUM_PICKUP_POINTS: int = 4 * len(PICKUP_RACKS_ARRANGEMENT) ** 2
NUM_DELIVERY_POINTS: int = 4 * int(AREA_DIMENSION_M)
NUM_REQUESTS: int = 20

COLLISION_REWARD: float = -1.0
PICKUP_REWARD_MULTIPLIER: float = 100.0
DELIVERY_REWARD_MULTIPLIER: float = 100.0

# MAX_PICKUP_WAIT_TIME: int = 400
# MAX_DELIVERY_WAIT_TIME: int = 400

AGENT_COLLISION_EPSILON: float = 0.05
PICKUP_POSITION_EPSILON: float = 5  # 0.3
DELIVERY_POSITION_EPSILON: float = 5  # 0.3

# Rendering
B2_VEL_ITERS: int = 10
B2_POS_ITERS: int = 10
PIXELS_PER_METER: int = 30
VIEWPORT_DIMENSION_PX: int = int(WORLD_DIMENSION_M) * PIXELS_PER_METER

AGENT_COLOR: typing.Tuple[float, float, float] = (0.0, 0.0, 0.0)
BORDER_COLOR: typing.Tuple[float, float, float] = (0.5, 0.5, 0.5)
PICKUP_POINT_COLOR: typing.Tuple[float, float, float] = (0.8, 0.8, 0.8)
DELIVERY_POINT_COLOR: typing.Tuple[float, float, float] = (0.8, 0.8, 0.8)


class Warehouse(MultiAgentEnv):
    def __init__(self) -> None:
        super(Warehouse, self).__init__()

        self.metadata = {
            "render.modes": ["human"],
            "video.frames_per_second": FRAMES_PER_SECOND,
        }
        self.reward_range = (-np.inf, -np.inf)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "self_position": gym.spaces.Box(
                    low=BORDER_WIDTH_M,
                    high=WORLD_DIMENSION_M - BORDER_WIDTH_M,
                    shape=(2,),
                    dtype=np.float32,
                ),
                "self_availability": gym.spaces.MultiBinary(1),
                "self_delivery_target": gym.spaces.Box(
                    low=np.array([0.0, 0.0, 0.0]),
                    high=np.array([WORLD_DIMENSION_M, WORLD_DIMENSION_M, np.inf]),
                    dtype=np.float32,
                ),
                "other_positions": gym.spaces.Box(
                    low=BORDER_WIDTH_M,
                    high=WORLD_DIMENSION_M - BORDER_WIDTH_M,
                    shape=(NUM_AGENTS - 1, 2),
                    dtype=np.float32,
                ),
                "other_availabilities": gym.spaces.MultiBinary(NUM_AGENTS - 1),
                "other_delivery_targets": gym.spaces.Box(
                    low=np.repeat(
                        np.array([0.0, 0.0, 0.0])[np.newaxis, :,], NUM_AGENTS - 1, axis=0
                    ),
                    high=np.repeat(
                        np.array([WORLD_DIMENSION_M, WORLD_DIMENSION_M, np.inf])[np.newaxis, :,],
                        NUM_AGENTS - 1,
                        axis=0,
                    ),
                    dtype=np.float32,
                ),
                "requests": gym.spaces.Box(
                    low=np.repeat(
                        np.array([0.0, 0.0, 0.0, 0.0, 0.0])[np.newaxis, :,], NUM_REQUESTS, axis=0,
                    ),
                    high=np.repeat(
                        np.array(
                            [
                                WORLD_DIMENSION_M,
                                WORLD_DIMENSION_M,
                                WORLD_DIMENSION_M,
                                WORLD_DIMENSION_M,
                                np.inf,
                            ]
                        )[np.newaxis, :,],
                        NUM_REQUESTS,
                        axis=0,
                    ),
                    dtype=np.float32,
                ),
            }
        )

        self._viewer: gym.Viewer = None

        self._world = world(gravity=(0, 0), doSleep=False)
        self._agent_bodies: typing.List[dynamicBody] = []
        self._border_bodies: typing.List[dynamicBody] = []

        self._agent_positions: np.ndarray = None
        self._agent_delivering_states: np.ndarray = None

        # self._pickup_points: np.ndarray = None
        # self._delivery_points: np.ndarray = None

        # self._pickup_point_states: np.ndarray = None
        # self._pickup_point_wait_times: np.ndarray = None
        # self._pickup_point_delivery_target_idx: np.ndarray = None
        # self._delivery_point_states: np.ndarray = None
        # self._delivery_point_wait_times: np.ndarray = None

        self._pickup_point_positions: np.ndarray = None
        self._delivery_point_positions: np.ndarray = None

        self._waiting_pickup_point_delivery_target_idxs: np.ndarray = None
        self._waiting_pickup_point_elapsed_times: np.ndarray = None

        self._served_pickup_point_delivery_target_idxs: np.ndarray = None
        self._served_pickup_point_elapsed_times: np.ndarray = None

        print(self.observation_space.sample())

    def reset(self) -> typing.Dict[str, gym.spaces.Dict]:
        # Init agents
        self._agent_bodies = []
        racks_diff = (PICKUP_RACKS_ARRANGEMENT[1] - PICKUP_RACKS_ARRANGEMENT[0]) / 2
        arrangement = [
            PICKUP_RACKS_ARRANGEMENT[0] - racks_diff,
            *[x + racks_diff for x in PICKUP_RACKS_ARRANGEMENT],
        ]
        agent_positions: typing.List[typing.List[float]] = []
        for x in arrangement:
            for y in arrangement:
                body = self._world.CreateDynamicBody(position=(x, y))
                _ = body.CreateCircleFixture(radius=AGENT_RADIUS, density=1.0, friction=0.0)
                self._agent_bodies.append(body)
                agent_positions.append([x, y])
        self._agent_positions = np.array(agent_positions, dtype=np.float32)
        self._agent_delivering_states = np.zeros(NUM_AGENTS, dtype=np.bool)

        # Init borders
        self._border_bodies = [
            self._world.CreateStaticBody(
                position=(WORLD_DIMENSION_M / 2, BORDER_WIDTH_M / 2),
                shapes=polygonShape(box=(WORLD_DIMENSION_M / 2, BORDER_WIDTH_M / 2)),
            ),
            self._world.CreateStaticBody(
                position=(WORLD_DIMENSION_M / 2, WORLD_DIMENSION_M - BORDER_WIDTH_M / 2,),
                shapes=polygonShape(box=(WORLD_DIMENSION_M / 2, BORDER_WIDTH_M / 2)),
            ),
            self._world.CreateStaticBody(
                position=(BORDER_WIDTH_M / 2, WORLD_DIMENSION_M / 2,),
                shapes=polygonShape(box=(BORDER_WIDTH_M / 2, WORLD_DIMENSION_M / 2)),
            ),
            self._world.CreateStaticBody(
                position=(WORLD_DIMENSION_M - BORDER_WIDTH_M / 2, WORLD_DIMENSION_M / 2,),
                shapes=polygonShape(box=(BORDER_WIDTH_M / 2, WORLD_DIMENSION_M / 2)),
            ),
        ]

        # Init pickup point positions
        pickup_point_positions = []
        for x in PICKUP_RACKS_ARRANGEMENT:
            for y in PICKUP_RACKS_ARRANGEMENT:
                pickup_point_positions.extend(
                    [
                        [x - 0.5, y - 0.5],
                        [x + 0.5, y - 0.5],
                        [x + 0.5, y + 0.5],
                        [x - 0.5, y + 0.5],
                    ]
                )
        self._pickup_point_positions = np.array(pickup_point_positions, dtype=np.float32)

        # Init delivery point positions
        delivery_point_positions = []
        for val in range(int(AREA_DIMENSION_M)):
            delivery_point_positions.extend(
                [
                    [BORDER_WIDTH_M + val + 0.5, BORDER_WIDTH_M + 0.5],
                    [BORDER_WIDTH_M + val + 0.5, WORLD_DIMENSION_M - BORDER_WIDTH_M - 0.5,],
                    [BORDER_WIDTH_M + 0.5, BORDER_WIDTH_M + val + 0.5],
                    [WORLD_DIMENSION_M - BORDER_WIDTH_M - 0.5, BORDER_WIDTH_M + val + 0.5,],
                ]
            )
        self._delivery_point_positions = np.array(delivery_point_positions, dtype=np.float32)

        self._waiting_pickup_point_elapsed_times = np.zeros(NUM_PICKUP_POINTS, dtype=np.float32)
        self._waiting_pickup_point_delivery_target_idxs = np.full(
            NUM_PICKUP_POINTS, -1, dtype=np.int32
        )
        self._waiting_pickup_point_delivery_target_idxs[
            np.random.choice(NUM_PICKUP_POINTS, NUM_REQUESTS, replace=False,)
        ] = np.random.choice(NUM_DELIVERY_POINTS, NUM_REQUESTS, replace=False,)

        self._served_pickup_point_elapsed_times = np.zeros(NUM_PICKUP_POINTS, dtype=np.float32)
        self._served_pickup_point_delivery_target_idxs = np.full(
            NUM_PICKUP_POINTS, -1, dtype=np.int32
        )

        # self._pickup_point_states = np.full(NUM_PICKUP_POINTS, -2, dtype=np.int32)
        # self._pickup_point_wait_times = np.zeros(NUM_PICKUP_POINTS, dtype=np.int32)
        # active_pickup_point_idxs = np.random.choice(NUM_PICKUP_POINTS, NUM_REQUESTS, replace=False,)
        # self._pickup_point_states[active_pickup_point_idxs] = -1

        # self._delivery_point_states = np.full(NUM_PICKUP_POINTS, -2, dtype=np.int32)
        # self._delivery_point_wait_times = np.zeros(NUM_PICKUP_POINTS, dtype=np.int32)
        # new_pickup_point_idxs = np.random.choice(NUM_PICKUP_POINTS, NUM_REQUESTS, replace=False,)
        # self._pickup_point_states[new_pickup_point_idxs] = -1

        # inactive_delivery_point_idxs = np.where(self._delivery_point_states == -2)[0]
        # new_delivery_point_idxs = np.random.choice(
        #     inactive_delivery_point_idxs,
        #     NUM_REQUESTS - NUM_DELIVERY_POINTS + inactive_delivery_point_idxs.shape[0],
        #     replace=False,
        # )
        # self._delivery_point_states[new_delivery_point_idxs] = -1
        # self._delivery_point_wait_times[new_delivery_point_idxs] = 0

        return {
            str(i): {
                "self_position": self._agent_positions[i],
                "other_positions": self._agent_positions,
                "pickup_positions": np.zeros((NUM_REQUESTS, 2)),
                "delivery_positions": np.zeros((NUM_REQUESTS, 2)),
            }
            for i in range(NUM_AGENTS)
        }

    def step(
        self, action_dict: typing.Dict[str, np.ndarray]
    ) -> typing.Tuple[
        typing.Dict[str, gym.spaces.Dict],
        typing.Dict[str, float],
        typing.Dict[str, bool],
        typing.Dict[str, typing.Dict[str, str]],
    ]:
        # Update agent velocities
        for key, value in action_dict.items():
            self._agent_bodies[int(key)].linearVelocity = value.tolist()

        # Step simulation
        self._world.Step(1.0 / FRAMES_PER_SECOND, 10, 10)

        # Update agent positions
        for idx, body in enumerate(self._agent_bodies):
            self._agent_positions[idx][0] = body.position[0]
            self._agent_positions[idx][1] = body.position[1]

        # Detect agent each-other collisions
        agent_eachother_distances = np.linalg.norm(
            np.repeat(self._agent_positions[:, np.newaxis, :], NUM_AGENTS, axis=1)
            - np.repeat(self._agent_positions[np.newaxis, :, :], NUM_AGENTS, axis=0),
            axis=2,
        )
        agent_eachother_collision_counts = (
            np.count_nonzero(
                agent_eachother_distances < 2 * AGENT_RADIUS + AGENT_COLLISION_EPSILON, axis=1
            )
            - 1
        )
        temp_rewards = agent_eachother_collision_counts * COLLISION_REWARD

        # Detect pickups
        agent_and_pickup_distances = np.linalg.norm(
            np.repeat(self._agent_positions[:, np.newaxis, :], NUM_PICKUP_POINTS, axis=1)
            - np.repeat(self._pickup_points[np.newaxis, :, :], NUM_AGENTS, axis=0),
            axis=2,
        )
        active_pickup_point_idxs = np.where(self._pickup_point_states > -2)[0]
        carrier_agent_idxs, valid_pickup_idxs = np.where(
            (agent_and_pickup_distances < PICKUP_POSITION_EPSILON)[:, active_pickup_point_idxs]
        )
        self._pickup_point_states[valid_pickup_idxs] = carrier_agent_idxs

        # TODO: Assign reward for pickup

        inactive_pickup_point_idxs = np.where(self._pickup_point_states == -2)[0]
        new_pickup_point_idxs = np.random.choice(
            inactive_pickup_point_idxs,
            NUM_REQUESTS - NUM_PICKUP_POINTS + inactive_pickup_point_idxs.shape[0],
            replace=False,
        )
        self._pickup_point_states[new_pickup_point_idxs] = -1
        self._pickup_point_wait_times[new_pickup_point_idxs] = 0

        # Detect deliveries

        # inactive_delivery_point_idxs = np.where(self._delivery_point_states == -2)[0]
        # new_delivery_point_idxs = np.random.choice(
        #     inactive_delivery_point_idxs,
        #     NUM_REQUESTS - NUM_DELIVERY_POINTS + inactive_delivery_point_idxs.shape[0],
        #     replace=False,
        # )
        # self._delivery_point_states[new_delivery_point_idxs] = -1
        # self._delivery_point_wait_times[new_delivery_point_idxs] = 0

        observations = {
            str(i): {
                "self_position": self._agent_positions[i],
                "pickup_positions": np.zeros((NUM_REQUESTS, 2)),
                "delivery_positions": np.zeros((NUM_REQUESTS, 2)),
            }
            for i in range(NUM_AGENTS)
        }
        rewards = {f"{i}": 0.0 for i in range(NUM_AGENTS)}
        dones = {f"{i}": False for i in range(NUM_AGENTS)}
        dones["__all__"] = False
        infos = {f"{i}": {"test": "test"} for i in range(NUM_AGENTS)}
        return observations, rewards, dones, infos

    def render(self, mode: str = "human") -> None:
        from gym.envs.classic_control import rendering

        if mode != "human":
            super(Warehouse, self).render(mode=mode)

        if self._viewer is None:
            self._viewer = rendering.Viewer(VIEWPORT_DIMENSION_PX, VIEWPORT_DIMENSION_PX)

        for body in self._border_bodies:
            for fixture in body.fixtures:
                self._viewer.draw_polygon(
                    [fixture.body.transform * v * PIXELS_PER_METER for v in fixture.shape.vertices],
                    color=BORDER_COLOR,
                )

        for point in self._pickup_points:
            self._viewer.draw_polygon(
                [
                    ((point[0] - 0.4) * PIXELS_PER_METER, (point[1] - 0.4) * PIXELS_PER_METER,),
                    ((point[0] + 0.4) * PIXELS_PER_METER, (point[1] - 0.4) * PIXELS_PER_METER,),
                    ((point[0] + 0.4) * PIXELS_PER_METER, (point[1] + 0.4) * PIXELS_PER_METER,),
                    ((point[0] - 0.4) * PIXELS_PER_METER, (point[1] + 0.4) * PIXELS_PER_METER,),
                ],
                color=PICKUP_POINT_COLOR,
            )

        for point in self._delivery_points:
            self._viewer.draw_polygon(
                [
                    ((point[0] - 0.4) * PIXELS_PER_METER, (point[1] - 0.4) * PIXELS_PER_METER,),
                    ((point[0] + 0.4) * PIXELS_PER_METER, (point[1] - 0.4) * PIXELS_PER_METER,),
                    ((point[0] + 0.4) * PIXELS_PER_METER, (point[1] + 0.4) * PIXELS_PER_METER,),
                    ((point[0] - 0.4) * PIXELS_PER_METER, (point[1] + 0.4) * PIXELS_PER_METER,),
                ],
                color=DELIVERY_POINT_COLOR,
            )

        for body in self._agent_bodies:
            for fixture in body.fixtures:
                self._viewer.draw_circle(
                    fixture.shape.radius * PIXELS_PER_METER, 30, color=AGENT_COLOR
                ).add_attr(
                    rendering.Transform(
                        translation=fixture.body.transform * fixture.shape.pos * PIXELS_PER_METER
                    )
                )

        self._viewer.render()

    def close(self) -> None:
        pass
