import time
import argparse
from typing import Dict, Deque, List
from collections import deque

import numpy as np

from warehouse import WarehouseSmall, WarehouseMedium, WarehouseLarge


STEP_ROTATION_DEG: int = 45


class WarehouseGridSolver:
    def __init__(self, num_agents: int, num_requests: int) -> None:
        self._num_agents = num_agents
        self._num_requests = num_requests

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

            # Rotate step direction if it conflicts with other agent's position to avoid stuc
            # due to collision
            if any(np.equal(future_position, observations[agent_id]["other_positions"]).all(1)):
                step = self._rotate_step(step, STEP_ROTATION_DEG)

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


def main(env_variant: str) -> None:
    step_time_buffer: Deque[float] = deque([], maxlen=10)
    render_time_buffer: Deque[float] = deque([], maxlen=10)

    if env_variant == "small":
        env = WarehouseSmall()
    elif env_variant == "medium":
        env = WarehouseMedium()
    else:
        env = WarehouseLarge()

    solver = WarehouseGridSolver(num_agents=env.num_agents, num_requests=env.num_requests)

    observations = env.reset()
    for _, observation in observations.items():
        assert env.observation_space.contains(observation)

    acc_rewards = [0.0 for i in range(env.num_agents)]
    done = False
    step_count = 0
    while not done:
        action_dict = solver.compute_action(observations)

        start_time = time.time()
        observations, rewards, dones, infos = env.step(action_dict=action_dict)
        step_time_buffer.append(1.0 / (time.time() - start_time))
        env.render()
        render_time_buffer.append(1.0 / (time.time() - start_time))

        for _, observation in observations.items():
            assert env.observation_space.contains(observation)

        acc_rewards = [acc_rewards[i] + rewards[f"{i}"] for i in range(env.num_agents)]
        done = dones["__all__"]

        print(f"\n=== Step {step_count} ===")
        print("Rewards:", *acc_rewards)
        print(
            f"Step avg FPS: {sum(step_time_buffer) / len(step_time_buffer)}, render avg FPS: {sum(render_time_buffer) / len(render_time_buffer)}"
        )

        step_count += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "env_variant", type=str, choices=["small", "medium", "large"], help="environment variant"
    )
    args = parser.parse_args()
    main(env_variant=args.env_variant)
