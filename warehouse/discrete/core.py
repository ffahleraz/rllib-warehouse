import time
from typing import List, Dict, Tuple

import numpy as np
import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv

__all__ = ["WarehouseDiscrete"]


# Engineering notes:
#   - Each agent is identified by int[0, NUM_AGENTS) in string type.
#   - Zero coordinate for the env is on the bottom left, this is then transformed to
#     top left for rendering.
#   - Anchor points for each objects (agents, pickup & delivery points) is on the bottom left.
#   - Environment layout:
#       |B|x| | |x|x| | |x|x| | |x|x| | |x|B|
#   - Game mechanism:
#       - There will always be NUM_REQUESTS requests for pickups in which all the pickup points are
#         unique but the delivery points may be not
#       - On each pickup, a new pickup request will be created with a random pickup point (different
#         from the existing requests, but may be the same as a pickup point that is already being
#         served) and a random delivery point (may be the same as existing requests or a delivery
#         point that is already being served)
#   - Constant Assumptions:
#       - NUM_REQUESTS >= NUM_AGENTS

# Environment
MOVES: List[List[float]] = [[x, y] for x in [-1, 0, 1] for y in [-1, 0, 1]]

PICKUP_REWARD: float = 1.0
DELIVERY_REWARD: float = 1.0

EPSILON = 1e-3

# Rendering
B2_VEL_ITERS: int = 10
B2_POS_ITERS: int = 10

AGENT_RADIUS: float = 0.4
BORDER_WIDTH: float = 1.0
PIXELS_PER_METER: int = 30

AGENT_COLORS: List[Tuple[float, float, float]] = [
    (0.5, 0.5, 0.5),
    (0.8, 0.8, 0.8),
    (0.0, 0.0, 1.0),
]
PICKUP_POINT_COLORS: List[Tuple[float, float, float]] = [
    (0.8, 0.8, 0.8),
    (0.0, 0.0, 1.0),
]
DELIVERY_POINT_COLORS: List[Tuple[float, float, float]] = [
    (0.8, 0.8, 0.8),
    (0.0, 0.0, 1.0),
]
BORDER_COLOR: Tuple[float, float, float] = (0.5, 0.5, 0.5)
BACKGROUND_COLOR: Tuple[float, float, float] = (1.0, 1.0, 1.0)


class WarehouseDiscrete(MultiAgentEnv):
    metadata = {
        "render.modes": ["human"],
    }

    def __init__(
        self,
        num_agents: int,
        num_requests: int,
        area_dimension: float,
        agent_init_positions: List[List[float]],
        pickup_racks_arrangement: List[float],
        episode_duration: int,
        pickup_wait_duration: int,
    ) -> None:
        super(WarehouseDiscrete, self).__init__()

        # Constants
        self._AREA_DIMENSION: float = area_dimension
        self._AGENT_INIT_POSITIONS: List[List[float]] = agent_init_positions
        self._PICKUP_RACKS_ARRANGEMENT: List[float] = pickup_racks_arrangement

        self._NUM_AGENTS: int = num_agents
        self._NUM_PICKUP_POINTS: int = 4 * len(pickup_racks_arrangement) ** 2
        self._NUM_DELIVERY_POINTS: int = 4 * int(self._AREA_DIMENSION - 4)
        self._NUM_REQUESTS: int = num_requests

        self._EPISODE_DURATION: int = episode_duration
        self._PICKUP_WAIT_DURATION: int = pickup_wait_duration

        self._VIEWPORT_DIMENSION_PX: int = int(
            self._AREA_DIMENSION + 2 * BORDER_WIDTH
        ) * PIXELS_PER_METER

        # Specs
        self.reward_range = (0.0, 1.0)
        self.action_space = gym.spaces.Discrete(len(MOVES))
        self.observation_space = gym.spaces.Dict(
            {
                "self_position": gym.spaces.Box(
                    low=0, high=self._AREA_DIMENSION, shape=(2,), dtype=np.int32,
                ),
                "self_availability": gym.spaces.MultiBinary(1),
                "self_delivery_target": gym.spaces.Box(
                    low=0, high=self._AREA_DIMENSION, shape=(2,), dtype=np.int32,
                ),
                "other_positions": gym.spaces.Box(
                    low=0,
                    high=self._AREA_DIMENSION,
                    shape=(self._NUM_AGENTS - 1, 2),
                    dtype=np.int32,
                ),
                "other_availabilities": gym.spaces.MultiBinary(self._NUM_AGENTS - 1),
                "other_delivery_targets": gym.spaces.Box(
                    low=0,
                    high=self._AREA_DIMENSION,
                    shape=(self._NUM_AGENTS - 1, 2),
                    dtype=np.int32,
                ),
                "requests": gym.spaces.Box(
                    low=0, high=self._AREA_DIMENSION, shape=(self._NUM_REQUESTS, 4), dtype=np.int32,
                ),
            }
        )

        # States
        self._viewer: gym.Viewer = None

        self._agent_positions = np.zeros((self._NUM_AGENTS, 2), dtype=np.int32)
        self._agent_delivery_targets = np.zeros(self._NUM_AGENTS, dtype=np.int32)

        self._delivery_point_positions = np.zeros((self._NUM_DELIVERY_POINTS, 2), dtype=np.int32)
        self._pickup_point_positions = np.zeros((self._NUM_PICKUP_POINTS, 2), dtype=np.int32)
        self._pickup_point_targets = np.zeros(self._NUM_PICKUP_POINTS, dtype=np.int32)
        self._pickup_point_timers = np.zeros(self._NUM_PICKUP_POINTS, dtype=np.int32)

        self._episode_time: int = 0

    def reset(self) -> Dict[str, gym.spaces.Dict]:
        self._episode_time = 0

        # Init agents
        agent_positions: List[List[float]] = []
        for x, y in self._AGENT_INIT_POSITIONS:
            agent_positions.append([x, y])
        self._agent_positions = np.array(agent_positions, dtype=np.int32)
        self._agent_delivery_targets = np.full(self._NUM_AGENTS, -1, dtype=np.int32)

        # Init pickup point positions
        pickup_point_positions = []
        for x in self._PICKUP_RACKS_ARRANGEMENT:
            for y in self._PICKUP_RACKS_ARRANGEMENT:
                pickup_point_positions.extend(
                    [[x - 1, y - 1], [x, y - 1], [x - 1, y], [x, y],]
                )
        self._pickup_point_positions = np.array(pickup_point_positions, dtype=np.int32)

        # Init delivery point positions
        delivery_point_positions = []
        for val in range(2, int(self._AREA_DIMENSION) - 2):
            delivery_point_positions.extend(
                [
                    [val, 0],
                    [0, val],
                    [val, self._AREA_DIMENSION - 1],
                    [self._AREA_DIMENSION - 1, val],
                ]
            )
        self._delivery_point_positions = np.array(delivery_point_positions, dtype=np.int32)

        # Init waiting request states
        self._pickup_point_targets = np.full(self._NUM_PICKUP_POINTS, -1, dtype=np.int32)
        self._pickup_point_timers = np.full(self._NUM_PICKUP_POINTS, -1, dtype=np.int32)

        new_waiting_pickup_points = np.random.choice(
            self._NUM_PICKUP_POINTS, self._NUM_REQUESTS, replace=False,
        )
        self._pickup_point_targets[new_waiting_pickup_points] = np.random.choice(
            self._NUM_DELIVERY_POINTS, self._NUM_REQUESTS, replace=False,
        )
        self._pickup_point_timers[new_waiting_pickup_points] = self._PICKUP_WAIT_DURATION

        # Compute observations
        agent_availabilities = np.ones(self._NUM_AGENTS, dtype=np.int32)
        agent_delivery_target_positions = np.full(
            (self._NUM_AGENTS, 2), self._AREA_DIMENSION // 2, dtype=np.int32
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
            for i in range(self._NUM_AGENTS)
        }

    def step(
        self, action_dict: Dict[str, np.ndarray]
    ) -> Tuple[
        Dict[str, gym.spaces.Dict], Dict[str, float], Dict[str, bool], Dict[str, Dict[str, str]],
    ]:
        self._episode_time += 1

        # Update agent positions
        for key, action in action_dict.items():
            idx = int(key)
            x = self._agent_positions[idx][0] + MOVES[action][0]
            y = self._agent_positions[idx][1] + MOVES[action][1]

            if not (0 <= x < self._AREA_DIMENSION):
                x = self._agent_positions[idx][0]
            if not (0 <= y < self._AREA_DIMENSION):
                y = self._agent_positions[idx][1]

            collide = False
            for xi, yi in np.delete(self._agent_positions, idx, axis=0):
                if xi == x and yi == y:
                    collide = True
                    break

            if not collide:
                self._agent_positions[idx][0] = x
                self._agent_positions[idx][1] = y

        # Remove expired pickup points
        self._pickup_point_timers[self._pickup_point_targets > -1] -= 1
        expired_waiting_pickup_points_mask = self._pickup_point_timers == 0
        self._pickup_point_targets[expired_waiting_pickup_points_mask] = -1
        self._pickup_point_timers[expired_waiting_pickup_points_mask] = -1

        # Detect pickups
        agent_and_pickup_point_distances = np.linalg.norm(
            (
                np.repeat(self._pickup_point_positions[np.newaxis, :, :], self._NUM_AGENTS, axis=0)
                - np.repeat(
                    self._agent_positions[:, np.newaxis, :], self._NUM_PICKUP_POINTS, axis=1
                )
            ).astype(np.float32),
            axis=2,
        )
        agent_and_pickup_point_collisions = agent_and_pickup_point_distances < EPSILON
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
        agent_rewards = np.zeros(self._NUM_AGENTS, dtype=np.float32)
        agent_rewards[new_delivering_agents_mask] += PICKUP_REWARD

        # Regenerate waiting pickup points
        inactive_pickup_points = np.where(self._pickup_point_targets == -1)[0]
        new_waiting_pickup_points = np.random.choice(
            inactive_pickup_points,
            self._NUM_REQUESTS - self._NUM_PICKUP_POINTS + inactive_pickup_points.shape[0],
            replace=False,
        )
        self._pickup_point_timers[new_waiting_pickup_points] = self._PICKUP_WAIT_DURATION

        new_pickup_point_targets = np.random.choice(
            self._NUM_DELIVERY_POINTS,
            self._NUM_REQUESTS - self._NUM_PICKUP_POINTS + inactive_pickup_points.shape[0],
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
            < EPSILON
        )
        delivered_agents = delivering_agents[delivering_agents_completion]

        self._agent_delivery_targets[delivered_agents] = -1

        # Calculate delivery rewards
        agent_rewards[delivered_agents] += DELIVERY_REWARD

        # Compute observations
        delivering_agents_mask = self._agent_delivery_targets > -1
        agent_availabilities = np.ones(self._NUM_AGENTS, dtype=np.int32)
        agent_availabilities[delivering_agents_mask] = 0

        agent_delivery_target_positions = np.full(
            (self._NUM_AGENTS, 2), self._AREA_DIMENSION // 2, dtype=np.int32
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
            for i in range(self._NUM_AGENTS)
        }

        # Compute rewards
        rewards = {f"{i}": agent_rewards[i] for i in range(self._NUM_AGENTS)}

        # Compute dones
        episode_done = self._episode_time >= self._EPISODE_DURATION
        dones = {f"{i}": episode_done for i in range(self._NUM_AGENTS)}
        dones["__all__"] = episode_done

        return observations, rewards, dones, {f"{i}": {} for i in range(self._NUM_AGENTS)}

    def render(self, mode: str = "human") -> None:
        from gym.envs.classic_control import rendering

        if mode != "human":
            super(WarehouseDiscrete, self).render(mode=mode)

        if self._viewer is None:
            self._viewer = rendering.Viewer(
                self._VIEWPORT_DIMENSION_PX, self._VIEWPORT_DIMENSION_PX
            )

        # Border
        self._viewer.draw_polygon(
            [
                (0.0, 0.0),
                ((self._AREA_DIMENSION + 2 * BORDER_WIDTH) * PIXELS_PER_METER, 0.0),
                (
                    (self._AREA_DIMENSION + 2 * BORDER_WIDTH) * PIXELS_PER_METER,
                    (self._AREA_DIMENSION + 2 * BORDER_WIDTH) * PIXELS_PER_METER,
                ),
                (0.0, (self._AREA_DIMENSION + 2 * BORDER_WIDTH) * PIXELS_PER_METER),
            ],
            color=BORDER_COLOR,
        )
        self._viewer.draw_polygon(
            [
                (BORDER_WIDTH * PIXELS_PER_METER, BORDER_WIDTH * PIXELS_PER_METER),
                (
                    (self._AREA_DIMENSION + BORDER_WIDTH) * PIXELS_PER_METER,
                    BORDER_WIDTH * PIXELS_PER_METER,
                ),
                (
                    (self._AREA_DIMENSION + BORDER_WIDTH) * PIXELS_PER_METER,
                    (self._AREA_DIMENSION + BORDER_WIDTH) * PIXELS_PER_METER,
                ),
                (
                    BORDER_WIDTH * PIXELS_PER_METER,
                    (self._AREA_DIMENSION + BORDER_WIDTH) * PIXELS_PER_METER,
                ),
            ],
            color=BACKGROUND_COLOR,
        )

        # Pickup points
        for idx, point in enumerate(self._pickup_point_positions):
            color = (
                PICKUP_POINT_COLORS[1]
                if self._pickup_point_targets[idx] > -1
                else PICKUP_POINT_COLORS[0]
            )
            self._viewer.draw_polygon(
                [
                    (
                        (point[0] + BORDER_WIDTH + 0.1) * PIXELS_PER_METER,
                        (point[1] + BORDER_WIDTH + 0.1) * PIXELS_PER_METER,
                    ),
                    (
                        (point[0] + BORDER_WIDTH + 0.9) * PIXELS_PER_METER,
                        (point[1] + BORDER_WIDTH + 0.1) * PIXELS_PER_METER,
                    ),
                    (
                        (point[0] + BORDER_WIDTH + 0.9) * PIXELS_PER_METER,
                        (point[1] + BORDER_WIDTH + 0.9) * PIXELS_PER_METER,
                    ),
                    (
                        (point[0] + BORDER_WIDTH + 0.1) * PIXELS_PER_METER,
                        (point[1] + BORDER_WIDTH + 0.9) * PIXELS_PER_METER,
                    ),
                ],
                color=color,
            )

        # Delivery points
        for idx, point in enumerate(self._delivery_point_positions):
            color = (
                DELIVERY_POINT_COLORS[1]
                if np.isin(idx, self._agent_delivery_targets)
                else DELIVERY_POINT_COLORS[0]
            )
            self._viewer.draw_polygon(
                [
                    (
                        (point[0] + BORDER_WIDTH + 0.1) * PIXELS_PER_METER,
                        (point[1] + BORDER_WIDTH + 0.1) * PIXELS_PER_METER,
                    ),
                    (
                        (point[0] + BORDER_WIDTH + 0.9) * PIXELS_PER_METER,
                        (point[1] + BORDER_WIDTH + 0.1) * PIXELS_PER_METER,
                    ),
                    (
                        (point[0] + BORDER_WIDTH + 0.9) * PIXELS_PER_METER,
                        (point[1] + BORDER_WIDTH + 0.9) * PIXELS_PER_METER,
                    ),
                    (
                        (point[0] + BORDER_WIDTH + 0.1) * PIXELS_PER_METER,
                        (point[1] + BORDER_WIDTH + 0.9) * PIXELS_PER_METER,
                    ),
                ],
                color=color,
            )
            self._viewer.draw_polygon(
                [
                    (
                        (point[0] + BORDER_WIDTH + 0.2) * PIXELS_PER_METER,
                        (point[1] + BORDER_WIDTH + 0.2) * PIXELS_PER_METER,
                    ),
                    (
                        (point[0] + BORDER_WIDTH + 0.8) * PIXELS_PER_METER,
                        (point[1] + BORDER_WIDTH + 0.2) * PIXELS_PER_METER,
                    ),
                    (
                        (point[0] + BORDER_WIDTH + 0.8) * PIXELS_PER_METER,
                        (point[1] + BORDER_WIDTH + 0.8) * PIXELS_PER_METER,
                    ),
                    (
                        (point[0] + BORDER_WIDTH + 0.2) * PIXELS_PER_METER,
                        (point[1] + BORDER_WIDTH + 0.8) * PIXELS_PER_METER,
                    ),
                ],
                color=DELIVERY_POINT_COLORS[0],
            )

        # Agents
        for idx, point in enumerate(self._agent_positions):
            transform = (
                point.astype(np.float32) + np.array([0.5, 0.5], dtype=np.float32) + BORDER_WIDTH
            )
            self._viewer.draw_circle(
                AGENT_RADIUS * PIXELS_PER_METER, 30, color=AGENT_COLORS[0]
            ).add_attr(rendering.Transform(translation=transform * PIXELS_PER_METER))
            self._viewer.draw_circle(
                AGENT_RADIUS * 3 / 4 * PIXELS_PER_METER, 30, color=AGENT_COLORS[1]
            ).add_attr(rendering.Transform(translation=transform * PIXELS_PER_METER))
            if self._agent_delivery_targets[idx] > -1:
                self._viewer.draw_circle(
                    AGENT_RADIUS / 2 * PIXELS_PER_METER, 30, color=AGENT_COLORS[2]
                ).add_attr(rendering.Transform(translation=transform * PIXELS_PER_METER))

        self._viewer.render()
