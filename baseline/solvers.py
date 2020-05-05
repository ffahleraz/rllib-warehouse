import time
from typing import Dict
from abc import ABC, abstractmethod

import numpy as np

import gym


STEP_ROTATION_DEG: int = 45
STEP_ROTATION_PROB: float = 0.0


class WarehouseSolver(ABC):
    @abstractmethod
    def compute_action(
        self, observations: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        ...


class WarehouseRandomSolver(WarehouseSolver):
    def __init__(self, action_space: gym.spaces.Space) -> None:
        self._action_space = action_space

    def compute_action(
        self, observations: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        return {str(i): self._action_space.sample() for i in range(len(observations))}


class WarehouseGreedySolver(WarehouseSolver):
    def __init__(self, num_agents: int, num_requests: int, action_space: gym.Space) -> None:
        self._num_agents = num_agents
        self._num_requests = num_requests
        self._action_space = action_space

    def compute_action(
        self, observations: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        action_dict = {}

        for i in range(self._num_agents):
            agent_id = f"{i}"
            if observations[agent_id]["self_availability"][0] == 0:
                target_position = observations[agent_id]["self_delivery_target"]
            else:
                target_position = self._find_closest(
                    observations[agent_id]["self_position"], observations[agent_id]["requests"]
                )

            step = np.clip(target_position - observations[agent_id]["self_position"], -1, 1)
            future_position = observations[agent_id]["self_position"] + step

            # # Rotate step direction if it conflicts with other agent's position to avoid stuc
            # # due to collision
            # if any(np.equal(future_position, observations[agent_id]["other_positions"]).all(1)):
            #     step = self._rotate_step(step, STEP_ROTATION_DEG)

            if np.random.uniform() < STEP_ROTATION_PROB:
                action_dict[agent_id] = self._action_space.sample()
            else:
                action_idxs = step + 1
                action = action_idxs[0] * 3 + action_idxs[1]
                action_dict[agent_id] = action

        return action_dict

    def _find_closest(self, agent_position: np.ndarray, requests: np.ndarray) -> np.ndarray:
        deltas = np.absolute(
            np.repeat(agent_position[np.newaxis, :], self._num_requests, axis=0) - requests[:, 0:2],
        )
        distances = np.sum(deltas, axis=1)
        return requests[np.argmin(distances)][0:2]

    def _rotate_step(self, step: np.ndarray, angle_deg: float) -> np.ndarray:
        theta = np.radians(angle_deg)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        return np.rint(np.dot(R, step)).astype(np.int32)
