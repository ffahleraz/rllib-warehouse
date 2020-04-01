import time
from typing import List, Dict, Tuple

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
#       - self._agent_delivery_targets = np.zeros(NUM_AGENTS, dtype=np.int32):
#           - -1: not delivering
#           - [0, NUM_DELIVERY_POINTS): target delivery point idx
#       - self._pickup_point_targets = np.zeros(NUM_PICKUP_POINTS, dtype=np.int32):
#           - -1: not waiting (inactive)
#           - [0, NUM_DELIVERY_POINTS): target delivery point idx
#       - self._pickup_point_timers = np.zeros(NUM_PICKUP_POINTS, dtype=np.int32):
#           - -1.0: not waiting (inactive)
#           - [0.0, oo): elapsed time since active
#   - Game mechanism:
#       - There will always be NUM_REQUESTS requests for pickups in which all the pickup points are
#         unique but the delivery points may be not
#       - On each pickup, a new pickup request will be created with a random pickup point (different
#         from the existing requests, but may be the same as a pickup point that is already being
#         served) and a random delivery point (may be the same as existing requests or a delivery
#         point that is already being served)
#   - Assumptions:
#       - NUM_REQUESTS >= NUM_AGENTS
#   - Params:
#       - WORLD is the viewport
#       - All positions are in the frame of WORLD
#       - WORLD_DIMENSION is the dimension of WORLD in meters
#       - AREA is the playable area (excluding borders)
#       - AREA_DIMENSION is the dimension of AREA in meters

# Environment
PICKUP_REWARD: float = 1.0
DELIVERY_REWARD: float = 1.0

FRAMES_PER_SECOND: int = 5
AGENT_RADIUS: float = 0.4
PICKUP_POSITION_TOLERANCE: float = 0.4
DELIVERY_POSITION_TOLERANCE: float = 0.4

# Rendering
B2_VEL_ITERS: int = 10
B2_POS_ITERS: int = 10
PIXELS_PER_METER: int = 30

AGENT_COLORS: List[Tuple[float, float, float]] = [
    (0.5, 0.5, 0.5),
    (0.8, 0.8, 0.8),
    (0.0, 0.0, 1.0),
]
BORDER_COLOR: Tuple[float, float, float] = (0.5, 0.5, 0.5)
PICKUP_POINT_COLORS: List[Tuple[float, float, float]] = [
    (0.8, 0.8, 0.8),
    (0.0, 0.0, 1.0),
]
DELIVERY_POINT_COLORS: List[Tuple[float, float, float]] = [
    (0.8, 0.8, 0.8),
    (0.0, 0.0, 1.0),
]


class WarehouseContinuous(MultiAgentEnv):
    metadata = {
        "render.modes": ["human"],
        "video.frames_per_second": FRAMES_PER_SECOND,
    }

    def __init__(
        self,
        num_agents: int,
        num_requests: int,
        world_dimension: float,
        border_width: float,
        agent_init_positions: List[List[float]],
        pickup_racks_arrangement: List[float],
        episode_duration_s: int,
        pickup_wait_duration_s: int,
    ) -> None:
        super(WarehouseContinuous, self).__init__()

        # Constants
        self._world_dimension: float = world_dimension
        self._area_dimension: float = world_dimension - 2 * border_width
        self._border_width: float = border_width
        self._agent_init_positions: List[List[float]] = agent_init_positions
        self._pickup_racks_arrangement: List[float] = pickup_racks_arrangement

        self._num_agents: int = num_agents
        self._num_pickup_points: int = 4 * len(pickup_racks_arrangement) ** 2
        self._num_delivery_points: int = 4 * int(self._area_dimension - 4)
        self._num_requests: int = num_requests

        self._episode_duration: int = episode_duration_s * FRAMES_PER_SECOND
        self._pickup_wait_duration: int = pickup_wait_duration_s * FRAMES_PER_SECOND

        self._viewport_dimension_px: int = int(self._world_dimension) * PIXELS_PER_METER

        # Specs
        self.reward_range = (0.0, 1.0)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "self_position": gym.spaces.Box(
                    low=self._border_width,
                    high=self._world_dimension - self._border_width,
                    shape=(2,),
                    dtype=np.float32,
                ),
                "self_availability": gym.spaces.MultiBinary(1),
                "self_delivery_target": gym.spaces.Box(
                    low=self._border_width,
                    high=self._world_dimension - self._border_width,
                    shape=(2,),
                    dtype=np.float32,
                ),
                "other_positions": gym.spaces.Box(
                    low=self._border_width,
                    high=self._world_dimension - self._border_width,
                    shape=(self._num_agents - 1, 2),
                    dtype=np.float32,
                ),
                "other_availabilities": gym.spaces.MultiBinary(self._num_agents - 1),
                "other_delivery_targets": gym.spaces.Box(
                    low=self._border_width,
                    high=self._world_dimension - self._border_width,
                    shape=(self._num_agents - 1, 2),
                    dtype=np.float32,
                ),
                "requests": gym.spaces.Box(
                    low=self._border_width,
                    high=self._world_dimension - self._border_width,
                    shape=(self._num_requests, 4),
                    dtype=np.float32,
                ),
            }
        )

        # States
        self._viewer: gym.Viewer = None

        self._world = world(gravity=(0, 0), doSleep=False)
        self._agent_bodies: List[dynamicBody] = []
        self._border_bodies: List[dynamicBody] = []

        self._agent_positions = np.zeros((self._num_agents, 2), dtype=np.float32)
        self._agent_delivery_targets = np.zeros(self._num_agents, dtype=np.int32)

        self._delivery_point_positions = np.zeros((self._num_delivery_points, 2), dtype=np.float32)
        self._pickup_point_positions = np.zeros((self._num_pickup_points, 2), dtype=np.float32)
        self._pickup_point_targets = np.zeros(self._num_pickup_points, dtype=np.int32)
        self._pickup_point_timers = np.zeros(self._num_pickup_points, dtype=np.int32)

        self._episode_time: int = 0

    def reset(self) -> Dict[str, gym.spaces.Dict]:
        self._episode_time = 0

        # Init agents
        agent_positions: List[List[float]] = []
        self._agent_bodies = []
        for x, y in self._agent_init_positions:
            body = self._world.CreateDynamicBody(position=(x, y))
            _ = body.CreateCircleFixture(radius=AGENT_RADIUS, density=1.0, friction=0.0)
            self._agent_bodies.append(body)
            agent_positions.append([x, y])

        self._agent_positions = np.array(agent_positions, dtype=np.float32)
        self._agent_delivery_targets = np.full(self._num_agents, -1, dtype=np.int32)

        # Init borders
        self._border_bodies = [
            self._world.CreateStaticBody(
                position=(self._world_dimension / 2, self._border_width / 2),
                shapes=polygonShape(box=(self._world_dimension / 2, self._border_width / 2)),
            ),
            self._world.CreateStaticBody(
                position=(
                    self._world_dimension / 2,
                    self._world_dimension - self._border_width / 2,
                ),
                shapes=polygonShape(box=(self._world_dimension / 2, self._border_width / 2)),
            ),
            self._world.CreateStaticBody(
                position=(self._border_width / 2, self._world_dimension / 2,),
                shapes=polygonShape(box=(self._border_width / 2, self._world_dimension / 2)),
            ),
            self._world.CreateStaticBody(
                position=(
                    self._world_dimension - self._border_width / 2,
                    self._world_dimension / 2,
                ),
                shapes=polygonShape(box=(self._border_width / 2, self._world_dimension / 2)),
            ),
        ]

        # Init pickup point positions
        pickup_point_positions = []
        for x in self._pickup_racks_arrangement:
            for y in self._pickup_racks_arrangement:
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
        for val in range(2, int(self._area_dimension) - 2):
            delivery_point_positions.extend(
                [
                    [self._border_width + val + 0.5, self._border_width + 0.5],
                    [
                        self._border_width + val + 0.5,
                        self._world_dimension - self._border_width - 0.5,
                    ],
                    [self._border_width + 0.5, self._border_width + val + 0.5],
                    [
                        self._world_dimension - self._border_width - 0.5,
                        self._border_width + val + 0.5,
                    ],
                ]
            )
        self._delivery_point_positions = np.array(delivery_point_positions, dtype=np.float32)

        # Init waiting request states
        self._pickup_point_targets = np.full(self._num_pickup_points, -1, dtype=np.int32)
        self._pickup_point_timers = np.full(self._num_pickup_points, -1, dtype=np.int32)

        new_waiting_pickup_points = np.random.choice(
            self._num_pickup_points, self._num_requests, replace=False,
        )
        self._pickup_point_targets[new_waiting_pickup_points] = np.random.choice(
            self._num_delivery_points, self._num_requests, replace=False,
        )
        self._pickup_point_timers[new_waiting_pickup_points] = self._pickup_wait_duration

        # Compute observations
        agent_availabilities = np.ones(self._num_agents, dtype=np.int8)
        agent_delivery_target_positions = np.full(
            (self._num_agents, 2), self._world_dimension / 2.0, dtype=np.float32
        )

        waiting_pickup_points_mask = self._pickup_point_targets > -1
        requests = self._pickup_point_positions[waiting_pickup_points_mask]
        requests = np.hstack(
            (
                requests,
                self._delivery_point_positions[
                    self._pickup_point_targets[waiting_pickup_points_mask]
                ],
            ),
        )

        return {
            str(i): {
                "self_position": self._agent_positions[i],
                "self_availability": agent_availabilities[np.newaxis, i],
                "self_delivery_target": agent_delivery_target_positions[i],
                "other_positions": np.delete(self._agent_positions, i, axis=0),
                "other_availabilities": np.delete(agent_availabilities, i, axis=0),
                "other_delivery_targets": np.delete(agent_delivery_target_positions, i, axis=0),
                "requests": requests,
            }
            for i in range(self._num_agents)
        }

    def step(
        self, action_dict: Dict[str, np.ndarray]
    ) -> Tuple[
        Dict[str, gym.spaces.Dict], Dict[str, float], Dict[str, bool], Dict[str, Dict[str, str]],
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

        # Remove expired pickup points
        self._pickup_point_timers[self._pickup_point_targets > -1] -= 1
        expired_waiting_pickup_points_mask = self._pickup_point_timers == 0
        self._pickup_point_targets[expired_waiting_pickup_points_mask] = -1
        self._pickup_point_timers[expired_waiting_pickup_points_mask] = -1

        # Detect pickups
        agent_and_pickup_point_distances = np.linalg.norm(
            np.repeat(self._pickup_point_positions[np.newaxis, :, :], self._num_agents, axis=0)
            - np.repeat(self._agent_positions[:, np.newaxis, :], self._num_pickup_points, axis=1),
            axis=2,
        )
        agent_and_pickup_point_collisions = (
            agent_and_pickup_point_distances < PICKUP_POSITION_TOLERANCE
        )
        new_served_pickup_point_candidates = np.argmax(agent_and_pickup_point_collisions, axis=1)
        new_delivering_agents_mask = (
            np.max(agent_and_pickup_point_collisions, axis=1)
            & (self._agent_delivery_targets == -1)
            & (self._pickup_point_targets[new_served_pickup_point_candidates] > -1)
        )
        new_served_pickup_points = new_served_pickup_point_candidates[new_delivering_agents_mask]

        self._agent_delivery_targets[new_delivering_agents_mask] = self._pickup_point_targets[
            new_served_pickup_points
        ]
        self._pickup_point_targets[new_served_pickup_points] = -1
        self._pickup_point_timers[new_served_pickup_points] = -1

        # Calculate pickup rewards
        agent_rewards = np.zeros(self._num_agents, dtype=np.float32)
        agent_rewards[new_delivering_agents_mask] += PICKUP_REWARD

        # Regenerate waiting pickup points
        inactive_pickup_points = np.where(self._pickup_point_targets == -1)[0]
        new_waiting_pickup_points = np.random.choice(
            inactive_pickup_points,
            self._num_requests - self._num_pickup_points + inactive_pickup_points.shape[0],
            replace=False,
        )
        self._pickup_point_timers[new_waiting_pickup_points] = self._pickup_wait_duration

        new_pickup_point_targets = np.random.choice(
            self._num_delivery_points,
            self._num_requests - self._num_pickup_points + inactive_pickup_points.shape[0],
            replace=False,
        )
        self._pickup_point_targets[new_waiting_pickup_points] = new_pickup_point_targets

        # Detect deliveries
        delivering_agents = np.where(self._agent_delivery_targets > -1)[0]
        delivering_agents_completion = (
            np.linalg.norm(
                self._delivery_point_positions[self._agent_delivery_targets[delivering_agents]]
                - self._agent_positions[delivering_agents],
                axis=1,
            )
            < DELIVERY_POSITION_TOLERANCE
        )
        delivered_agents = delivering_agents[delivering_agents_completion]

        self._agent_delivery_targets[delivered_agents] = -1

        # Calculate delivery rewards
        agent_rewards[delivered_agents] += DELIVERY_REWARD

        # Compute observations
        delivering_agents_mask = self._agent_delivery_targets > -1
        agent_availabilities = np.ones(self._num_agents, dtype=np.int8)
        agent_availabilities[delivering_agents_mask] = 0

        agent_delivery_target_positions = np.full(
            (self._num_agents, 2), self._world_dimension / 2.0, dtype=np.float32
        )
        agent_delivery_target_positions[delivering_agents_mask] = self._delivery_point_positions[
            self._agent_delivery_targets[delivering_agents_mask]
        ]

        waiting_pickup_points_mask = self._pickup_point_targets > -1
        requests = self._pickup_point_positions[waiting_pickup_points_mask]
        requests = np.hstack(
            (
                requests,
                self._delivery_point_positions[
                    self._pickup_point_targets[waiting_pickup_points_mask]
                ],
            ),
        )

        observations = {
            str(i): {
                "self_position": self._agent_positions[i],
                "self_availability": agent_availabilities[np.newaxis, i],
                "self_delivery_target": agent_delivery_target_positions[i],
                "other_positions": np.delete(self._agent_positions, i, axis=0),
                "other_availabilities": np.delete(agent_availabilities, i, axis=0),
                "other_delivery_targets": np.delete(agent_delivery_target_positions, 1, axis=0),
                "requests": requests,
            }
            for i in range(self._num_agents)
        }

        # Compute rewards
        rewards = {f"{i}": agent_rewards[i] for i in range(self._num_agents)}

        # Compute dones
        episode_done = self._episode_time >= self._episode_duration
        dones = {f"{i}": episode_done for i in range(self._num_agents)}
        dones["__all__"] = episode_done

        return observations, rewards, dones, {f"{i}": {} for i in range(self._num_agents)}

    def render(self, mode: str = "human") -> None:
        from gym.envs.classic_control import rendering

        if mode != "human":
            super(WarehouseContinuous, self).render(mode=mode)

        if self._viewer is None:
            self._viewer = rendering.Viewer(
                self._viewport_dimension_px, self._viewport_dimension_px
            )

        for body in self._border_bodies:
            for fixture in body.fixtures:
                self._viewer.draw_polygon(
                    [fixture.body.transform * v * PIXELS_PER_METER for v in fixture.shape.vertices],
                    color=BORDER_COLOR,
                )

        for idx, point in enumerate(self._pickup_point_positions):
            color = (
                PICKUP_POINT_COLORS[1]
                if self._pickup_point_targets[idx] > -1
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
                if np.isin(idx, self._agent_delivery_targets)
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
                if self._agent_delivery_targets[idx] > -1:
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
