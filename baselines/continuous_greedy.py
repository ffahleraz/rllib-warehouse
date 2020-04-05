import time
import argparse
from typing import Dict, Deque, List
from collections import deque

import numpy as np

from warehouse import WarehouseContinuousSmall


class WarehouseContinuousSolver:
    def __init__(self, num_agents: int, num_requests: int) -> None:
        self._num_agents = num_agents
        self._num_requests = num_requests
        self._agent_pickup_targets = [-1] * num_agents

    def compute_action(
        self, observations: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        action_dict = {}

        for i in range(self._num_agents):
            agent_id = f"{i}"
            if observations[agent_id]["self_availability"][0] == 0:
                self._agent_pickup_targets[i] = -1
                target = observations[agent_id]["self_delivery_target"]
            else:
                if self._agent_pickup_targets[i] == -1:
                    for j in range(self._num_requests):
                        if j not in self._agent_pickup_targets:
                            self._agent_pickup_targets[i] = j
                            break
                target = observations[agent_id]["requests"][self._agent_pickup_targets[i]][0:2]

            action = target - observations[agent_id]["self_position"]
            action /= np.linalg.norm(action)
            action_dict[agent_id] = action

        return action_dict


def main(env_variant: str) -> None:
    step_time_buffer: Deque[float] = deque([], maxlen=10)
    render_time_buffer: Deque[float] = deque([], maxlen=10)

    if env_variant == "small":
        env = WarehouseContinuousSmall()

    solver = WarehouseContinuousSolver(num_agents=env.num_agents, num_requests=env.num_requests)

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
    parser.add_argument("env_variant", type=str, choices=["small"], help="environment variant")
    args = parser.parse_args()
    main(env_variant=args.env_variant)
