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
#   - Environment layout:
#       |B|x| | |x|x| | |x|x| | |x|x| | |x|B|
#   - States:
#       - self._waiting_pickup_point_targets = np.zeros(NUM_PICKUP_POINTS, dtype=np.int32):
#           - -1: not waiting (inactive)
#           - [0, NUM_DELIVERY_POINTS): target delivery point idx
#       - self._waiting_pickup_point_timers = np.zeros(NUM_PICKUP_POINTS, dtype=np.float32):
#           - -1.0: not waiting (inactive)
#           - [0.0, oo): elapsed time since active
#       - self._served_pickup_point_server_agents = np.zeros(NUM_PICKUP_POINTS, dtype=np.int32):
#           - -1: not served or not waiting
#           - [0, NUM_AGENTS): picker agent idx
#       - self._served_pickup_point_targets = np.zeros(NUM_PICKUP_POINTS, dtype=np.int32):
#           - -1: not served or not waiting
#           - [0, NUM_DELIVERY_POINTS): target delivery point idx
#       - self._served_pickup_point_timers = np.zeros(NUM_PICKUP_POINTS, dtype=np.float32):
#           - -1.0: not served or not waiting
#           - [0.0, oo): elapsed time since served
#   - Game mechanism:
#       - There will always be NUM_REQUESTS requests for pickups in which all the pickup points are
#         unique but the delivery points may be not
#       - On each pickup, a new pickup request will be created with a random pickup point (different
#         from the existing requests, but may be the same as a pickup point that is already being
#         served) and a random delivery point (may be the same as existing requests or a delivery
#         point that is already being served)

# Environment
AREA_DIMENSION_M: float = 12.0
BORDER_WIDTH_M: float = 1.0
WORLD_DIMENSION_M: float = AREA_DIMENSION_M + 2 * BORDER_WIDTH_M
AGENT_RADIUS: float = 0.4
PICKUP_RACKS_ARRANGEMENT: typing.List[float] = [5.0, 9.0]
FRAMES_PER_SECOND: int = 10

NUM_AGENTS: int = 4
NUM_PICKUP_POINTS: int = 4 * len(PICKUP_RACKS_ARRANGEMENT) ** 2
NUM_DELIVERY_POINTS: int = 4 * int(AREA_DIMENSION_M - 4)
NUM_REQUESTS: int = 4

COLLISION_REWARD: float = -10.0
PICKUP_BASE_REWARD: float = 200.0
PICKUP_TIME_REWARD_MULTIPLIER: float = 1.0
DELIVERY_BASE_REWARD: float = 200.0
DELIVERY_TIME_REWARD_MULTIPLIER: float = 1.0

MAX_EPISODE_TIME: int = 400 * FRAMES_PER_SECOND
MAX_PICKUP_WAIT_TIME: float = 40.0 * FRAMES_PER_SECOND
MAX_DELIVERY_WAIT_TIME: float = 40.0 * FRAMES_PER_SECOND

AGENT_COLLISION_EPSILON: float = 0.05
PICKUP_POSITION_EPSILON: float = 0.3
DELIVERY_POSITION_EPSILON: float = 0.3

# Rendering
B2_VEL_ITERS: int = 10
B2_POS_ITERS: int = 10
PIXELS_PER_METER: int = 30
VIEWPORT_DIMENSION_PX: int = int(WORLD_DIMENSION_M) * PIXELS_PER_METER

AGENT_COLORS: typing.List[typing.Tuple[float, float, float]] = [
    (0.5, 0.5, 0.5),
    (0.8, 0.8, 0.8),
    (0.0, 0.0, 1.0),
]
BORDER_COLOR: typing.Tuple[float, float, float] = (0.5, 0.5, 0.5)
PICKUP_POINT_COLORS: typing.List[typing.Tuple[float, float, float]] = [
    (0.8, 0.8, 0.8),
    (0.0, 0.0, 1.0),
]
DELIVERY_POINT_COLORS: typing.List[typing.Tuple[float, float, float]] = [
    (0.8, 0.8, 0.8),
    (0.0, 0.0, 1.0),
]


class Warehouse(MultiAgentEnv):
    def __init__(self) -> None:
        super(Warehouse, self).__init__()

        self.metadata = {
            "render.modes": ["human"],
            "video.frames_per_second": FRAMES_PER_SECOND,
        }
        self.reward_range = (-np.inf, np.inf)
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
                    high=np.array([WORLD_DIMENSION_M, WORLD_DIMENSION_M, MAX_DELIVERY_WAIT_TIME]),
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
                        np.array([WORLD_DIMENSION_M, WORLD_DIMENSION_M, MAX_DELIVERY_WAIT_TIME])[
                            np.newaxis, :,
                        ],
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
                                MAX_PICKUP_WAIT_TIME,
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

        self._agent_positions = np.zeros((NUM_AGENTS, 2), dtype=np.float32)
        self._agent_availabilities = np.zeros(NUM_AGENTS, dtype=np.int8)

        self._pickup_point_positions = np.zeros((NUM_PICKUP_POINTS, 2), dtype=np.float32)
        self._delivery_point_positions = np.zeros((NUM_DELIVERY_POINTS, 2), dtype=np.float32)

        self._waiting_pickup_point_targets = np.zeros(NUM_PICKUP_POINTS, dtype=np.int32)
        self._waiting_pickup_point_timers = np.zeros(NUM_PICKUP_POINTS, dtype=np.float32)

        self._served_pickup_point_server_agents = np.zeros(NUM_PICKUP_POINTS, dtype=np.int32)
        self._served_pickup_point_targets = np.zeros(NUM_PICKUP_POINTS, dtype=np.int32)
        self._served_pickup_point_timers = np.zeros(NUM_PICKUP_POINTS, dtype=np.float32)

        self._episode_time: int = 0

    def reset(self) -> typing.Dict[str, gym.spaces.Dict]:
        self._episode_time = 0

        # Init agents
        racks_diff = (PICKUP_RACKS_ARRANGEMENT[1] - PICKUP_RACKS_ARRANGEMENT[0]) / 2
        arrangement = [
            (PICKUP_RACKS_ARRANGEMENT[0] - racks_diff, PICKUP_RACKS_ARRANGEMENT[0] + racks_diff),
            (PICKUP_RACKS_ARRANGEMENT[0] + racks_diff, PICKUP_RACKS_ARRANGEMENT[0] - racks_diff),
            (PICKUP_RACKS_ARRANGEMENT[0] + racks_diff, PICKUP_RACKS_ARRANGEMENT[1] + racks_diff),
            (PICKUP_RACKS_ARRANGEMENT[1] + racks_diff, PICKUP_RACKS_ARRANGEMENT[0] + racks_diff),
        ]

        agent_positions: typing.List[typing.List[float]] = []
        self._agent_bodies = []
        for x, y in arrangement:
            body = self._world.CreateDynamicBody(position=(x, y))
            _ = body.CreateCircleFixture(radius=AGENT_RADIUS, density=1.0, friction=0.0)
            self._agent_bodies.append(body)
            agent_positions.append([x, y])

        self._agent_positions = np.array(agent_positions, dtype=np.float32)
        self._agent_availabilities = np.ones(NUM_AGENTS, dtype=np.int8)

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
        for val in range(2, int(AREA_DIMENSION_M) - 2):
            delivery_point_positions.extend(
                [
                    [BORDER_WIDTH_M + val + 0.5, BORDER_WIDTH_M + 0.5],
                    [BORDER_WIDTH_M + val + 0.5, WORLD_DIMENSION_M - BORDER_WIDTH_M - 0.5,],
                    [BORDER_WIDTH_M + 0.5, BORDER_WIDTH_M + val + 0.5],
                    [WORLD_DIMENSION_M - BORDER_WIDTH_M - 0.5, BORDER_WIDTH_M + val + 0.5,],
                ]
            )
        self._delivery_point_positions = np.array(delivery_point_positions, dtype=np.float32)

        # Init waiting request states
        self._waiting_pickup_point_targets = np.full(NUM_PICKUP_POINTS, -1, dtype=np.int32)
        self._waiting_pickup_point_timers = np.full(NUM_PICKUP_POINTS, -1.0, dtype=np.float32)

        new_waiting_pickup_points = np.random.choice(
            NUM_PICKUP_POINTS, NUM_REQUESTS, replace=False,
        )
        self._waiting_pickup_point_targets[new_waiting_pickup_points] = np.random.choice(
            NUM_DELIVERY_POINTS, NUM_REQUESTS, replace=False,
        )
        self._waiting_pickup_point_timers[new_waiting_pickup_points] = MAX_PICKUP_WAIT_TIME

        # Init served request states
        self._served_pickup_point_server_agents = np.full(NUM_PICKUP_POINTS, -1, dtype=np.int32)
        self._served_pickup_point_targets = np.full(NUM_PICKUP_POINTS, -1, dtype=np.int32)
        self._served_pickup_point_timers = np.full(NUM_PICKUP_POINTS, -1.0, dtype=np.float32)

        # Compute observations
        other_delivery_targets = (
            self.observation_space["other_delivery_targets"].high
            - self.observation_space["other_delivery_targets"].low
        ) / 2
        other_delivery_targets[:, -1] = np.zeros(NUM_AGENTS - 1)

        waiting_pickup_points_mask = self._waiting_pickup_point_targets > -1
        requests = self._pickup_point_positions[waiting_pickup_points_mask]
        requests = np.hstack(
            (
                requests,
                self._delivery_point_positions[
                    self._waiting_pickup_point_targets[waiting_pickup_points_mask]
                ],
                self._waiting_pickup_point_timers[waiting_pickup_points_mask][:, np.newaxis],
            ),
        )

        return {
            str(i): {
                "self_position": self._agent_positions[i],
                "self_availability": self._agent_availabilities[np.newaxis, i],
                "self_delivery_target": other_delivery_targets[0],
                "other_positions": np.delete(self._agent_positions, i, axis=0),
                "other_availabilities": np.delete(self._agent_availabilities, i, axis=0),
                "other_delivery_targets": other_delivery_targets,
                "requests": requests,
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
        self._episode_time += 1

        # Update agent velocities
        for key, value in action_dict.items():
            self._agent_bodies[int(key)].linearVelocity = value.tolist()

        # Step simulation
        self._world.Step(1.0 / FRAMES_PER_SECOND, 10, 10)

        # Update agent positions
        for idx, body in enumerate(self._agent_bodies):
            self._agent_positions[idx][0] = body.position[0]
            self._agent_positions[idx][1] = body.position[1]

        # Detect agent each-other collisions and calculate rewards
        agent_and_agent_distances = np.linalg.norm(
            np.repeat(self._agent_positions[np.newaxis, :, :], NUM_AGENTS, axis=0)
            - np.repeat(self._agent_positions[:, np.newaxis, :], NUM_AGENTS, axis=1),
            axis=2,
        )
        agent_collision_counts = (
            np.count_nonzero(
                agent_and_agent_distances < 2 * AGENT_RADIUS + AGENT_COLLISION_EPSILON, axis=1
            )
            - 1
        )
        agent_rewards = agent_collision_counts * COLLISION_REWARD

        # Decrement timers
        self._waiting_pickup_point_timers[self._waiting_pickup_point_targets > -1] -= 1.0
        self._served_pickup_point_timers[self._served_pickup_point_targets > -1] -= 1.0

        # Remove expired pickup and deliveries
        expired_waiting_pickup_points_mask = self._waiting_pickup_point_timers == 0.0
        self._waiting_pickup_point_targets[expired_waiting_pickup_points_mask] = -1
        self._waiting_pickup_point_timers[expired_waiting_pickup_points_mask] = -1.0

        expired_served_pickup_points_mask = self._served_pickup_point_timers == 0.0
        self._agent_availabilities[
            self._served_pickup_point_server_agents[expired_served_pickup_points_mask]
        ] = 1
        self._served_pickup_point_server_agents[expired_served_pickup_points_mask] = -1
        self._served_pickup_point_targets[expired_served_pickup_points_mask] = -1
        self._served_pickup_point_timers[expired_served_pickup_points_mask] = -1.0

        # Detect pickups
        pickup_point_and_agent_distances = np.linalg.norm(
            np.repeat(self._agent_positions[np.newaxis, :, :], NUM_PICKUP_POINTS, axis=0)
            - np.repeat(self._pickup_point_positions[:, np.newaxis, :], NUM_AGENTS, axis=1),
            axis=2,
        )
        pickup_point_and_agent_collisions_mask = (
            pickup_point_and_agent_distances < PICKUP_POSITION_EPSILON
        )
        new_served_pickup_point_candidates_mask = np.max(
            pickup_point_and_agent_collisions_mask, axis=1
        )
        new_served_pickup_point_server_agent_candidates = np.argmax(
            pickup_point_and_agent_collisions_mask, axis=1
        )
        new_served_pickup_points_mask = (
            new_served_pickup_point_candidates_mask
            & (self._waiting_pickup_point_targets > -1)
            & (self._agent_availabilities[new_served_pickup_point_server_agent_candidates] > 0)
        )
        new_served_pickup_points_server_agents = new_served_pickup_point_server_agent_candidates[
            new_served_pickup_points_mask
        ]

        # Calculate pickup rewards
        agent_rewards[new_served_pickup_points_server_agents] += (
            PICKUP_BASE_REWARD
            + self._waiting_pickup_point_timers[new_served_pickup_points_mask]
            * PICKUP_TIME_REWARD_MULTIPLIER
        )

        # Update states
        self._served_pickup_point_server_agents[
            new_served_pickup_points_mask
        ] = new_served_pickup_points_server_agents
        self._served_pickup_point_targets[
            new_served_pickup_points_mask
        ] = self._waiting_pickup_point_targets[new_served_pickup_points_mask]
        self._served_pickup_point_timers[new_served_pickup_points_mask] = MAX_DELIVERY_WAIT_TIME

        self._agent_availabilities[new_served_pickup_points_server_agents] = 0

        self._waiting_pickup_point_targets[new_served_pickup_points_mask] = -1
        self._waiting_pickup_point_timers[new_served_pickup_points_mask] = -1.0

        # Regenerate waiting pickup points
        inactive_pickup_points = np.where(self._waiting_pickup_point_targets == -1)[0]
        new_waiting_pickup_points = np.random.choice(
            inactive_pickup_points,
            NUM_REQUESTS - NUM_PICKUP_POINTS + inactive_pickup_points.shape[0],
            replace=False,
        )
        self._waiting_pickup_point_timers[new_waiting_pickup_points] = MAX_PICKUP_WAIT_TIME

        new_waiting_pickup_point_targets = np.random.choice(
            NUM_DELIVERY_POINTS,
            NUM_REQUESTS - NUM_PICKUP_POINTS + inactive_pickup_points.shape[0],
            replace=False,
        )
        self._waiting_pickup_point_targets[
            new_waiting_pickup_points
        ] = new_waiting_pickup_point_targets

        # Detect deliveries
        delivery_point_and_agent_distances = np.linalg.norm(
            np.repeat(self._agent_positions[np.newaxis, :, :], NUM_DELIVERY_POINTS, axis=0)
            - np.repeat(self._delivery_point_positions[:, np.newaxis, :], NUM_AGENTS, axis=1),
            axis=2,
        )
        delivery_point_and_agent_collisions_mask = (
            delivery_point_and_agent_distances < DELIVERY_POSITION_EPSILON
        )
        completed_delivery_point_candidates_mask = np.max(
            delivery_point_and_agent_collisions_mask, axis=1
        )
        completed_delivery_point_server_agent_candidates = np.argmax(
            delivery_point_and_agent_collisions_mask, axis=1
        )
        served_pickup_points_mask = self._served_pickup_point_targets > -1
        completed_pickup_points_mask = (
            completed_delivery_point_candidates_mask[self._served_pickup_point_targets]
            & served_pickup_points_mask
            & (
                (self._served_pickup_point_server_agents & served_pickup_points_mask)
                == (
                    completed_delivery_point_server_agent_candidates[
                        self._served_pickup_point_targets
                    ]
                    & served_pickup_points_mask
                )
            )
        )

        # Calculate delivery rewards
        agent_rewards[self._served_pickup_point_server_agents[completed_pickup_points_mask]] += (
            DELIVERY_BASE_REWARD
            + self._served_pickup_point_timers[completed_pickup_points_mask]
            * DELIVERY_TIME_REWARD_MULTIPLIER
        )

        # Update states
        self._agent_availabilities[
            self._served_pickup_point_server_agents[completed_pickup_points_mask]
        ] = 1
        self._served_pickup_point_server_agents[completed_pickup_points_mask] = -1
        self._served_pickup_point_targets[completed_pickup_points_mask] = -1
        self._served_pickup_point_timers[completed_pickup_points_mask] = -1.0

        # Compute observations
        served_pickup_points_mask = self._served_pickup_point_targets > -1
        agent_delivery_targets = np.full(
            (NUM_AGENTS, 2),
            (
                (
                    self.observation_space["self_delivery_target"].high
                    - self.observation_space["self_delivery_target"].low
                )
                / 2
            )[0],
        )
        agent_delivery_targets[
            self._served_pickup_point_server_agents[served_pickup_points_mask]
        ] = self._delivery_point_positions[
            self._served_pickup_point_targets[served_pickup_points_mask]
        ]
        agent_delivery_targets = np.hstack((agent_delivery_targets, np.zeros((NUM_AGENTS, 1))))

        waiting_pickup_points_mask = self._waiting_pickup_point_targets > -1
        requests = self._pickup_point_positions[waiting_pickup_points_mask]
        requests = np.hstack(
            (
                requests,
                self._delivery_point_positions[
                    self._waiting_pickup_point_targets[waiting_pickup_points_mask]
                ],
                self._waiting_pickup_point_timers[waiting_pickup_points_mask][:, np.newaxis],
            ),
        )

        observations = {
            str(i): {
                "self_position": self._agent_positions[i],
                "self_availability": self._agent_availabilities[np.newaxis, i],
                "self_delivery_target": agent_delivery_targets[i],
                "other_positions": np.delete(self._agent_positions, i, axis=0),
                "other_availabilities": np.delete(self._agent_availabilities, i, axis=0),
                "other_delivery_targets": np.delete(agent_delivery_targets, 1, axis=0),
                "requests": requests,
            }
            for i in range(NUM_AGENTS)
        }

        # Compute rewards
        rewards = {f"{i}": agent_rewards[i] for i in range(NUM_AGENTS)}

        # Compute dones
        episode_done = self._episode_time >= MAX_EPISODE_TIME
        dones = {f"{i}": episode_done for i in range(NUM_AGENTS)}
        dones["__all__"] = episode_done

        return observations, rewards, dones, {f"{i}": {} for i in range(NUM_AGENTS)}

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

        for idx, point in enumerate(self._pickup_point_positions):
            color = (
                PICKUP_POINT_COLORS[1]
                if self._waiting_pickup_point_targets[idx] > -1
                else PICKUP_POINT_COLORS[0]
            )
            self._viewer.draw_polygon(
                [
                    ((point[0] - 0.4) * PIXELS_PER_METER, (point[1] - 0.4) * PIXELS_PER_METER,),
                    ((point[0] + 0.4) * PIXELS_PER_METER, (point[1] - 0.4) * PIXELS_PER_METER,),
                    ((point[0] + 0.4) * PIXELS_PER_METER, (point[1] + 0.4) * PIXELS_PER_METER,),
                    ((point[0] - 0.4) * PIXELS_PER_METER, (point[1] + 0.4) * PIXELS_PER_METER,),
                ],
                color=color,
            )

        for idx, point in enumerate(self._delivery_point_positions):
            color = (
                DELIVERY_POINT_COLORS[1]
                if np.isin(idx, self._served_pickup_point_targets)
                else DELIVERY_POINT_COLORS[0]
            )
            self._viewer.draw_polygon(
                [
                    ((point[0] - 0.4) * PIXELS_PER_METER, (point[1] - 0.4) * PIXELS_PER_METER,),
                    ((point[0] + 0.4) * PIXELS_PER_METER, (point[1] - 0.4) * PIXELS_PER_METER,),
                    ((point[0] + 0.4) * PIXELS_PER_METER, (point[1] + 0.4) * PIXELS_PER_METER,),
                    ((point[0] - 0.4) * PIXELS_PER_METER, (point[1] + 0.4) * PIXELS_PER_METER,),
                ],
                color=color,
            )
            self._viewer.draw_polygon(
                [
                    ((point[0] - 0.3) * PIXELS_PER_METER, (point[1] - 0.3) * PIXELS_PER_METER,),
                    ((point[0] + 0.3) * PIXELS_PER_METER, (point[1] - 0.3) * PIXELS_PER_METER,),
                    ((point[0] + 0.3) * PIXELS_PER_METER, (point[1] + 0.3) * PIXELS_PER_METER,),
                    ((point[0] - 0.3) * PIXELS_PER_METER, (point[1] + 0.3) * PIXELS_PER_METER,),
                ],
                color=DELIVERY_POINT_COLORS[0],
            )

        for idx, body in enumerate(self._agent_bodies):
            for fixture in body.fixtures:
                self._viewer.draw_circle(
                    fixture.shape.radius * PIXELS_PER_METER, 30, color=AGENT_COLORS[0]
                ).add_attr(
                    rendering.Transform(
                        translation=fixture.body.transform * fixture.shape.pos * PIXELS_PER_METER
                    )
                )
                self._viewer.draw_circle(
                    (fixture.shape.radius) * 3 / 4 * PIXELS_PER_METER, 30, color=AGENT_COLORS[1]
                ).add_attr(
                    rendering.Transform(
                        translation=fixture.body.transform * fixture.shape.pos * PIXELS_PER_METER
                    )
                )
                if self._agent_availabilities[idx] == 0:
                    self._viewer.draw_circle(
                        (fixture.shape.radius) / 2 * PIXELS_PER_METER, 30, color=AGENT_COLORS[2]
                    ).add_attr(
                        rendering.Transform(
                            translation=fixture.body.transform
                            * fixture.shape.pos
                            * PIXELS_PER_METER
                        )
                    )

        self._viewer.render()

    def close(self) -> None:
        pass
