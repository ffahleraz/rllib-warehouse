import time
from typing import Dict, Deque, List
from collections import deque

import numpy as np

import discrete
from discrete import WarehouseDiscrete


NUM_AGENTS: int = discrete.NUM_AGENTS
NUM_REQUESTS: int = discrete.NUM_REQUESTS
ROTATE_ACTION_PROB: float = 0.1  # To avoid stuck due to collision


class WarehouseDiscreteSolver:
    def __init__(self) -> None:
        self._agent_pickup_targets = [-1] * NUM_AGENTS

    def compute_action(
        self, observations: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        action_dict = {}

        for i in range(NUM_AGENTS):
            agent_id = f"{i}"
            if observations[agent_id]["self_availability"][0] == 0:
                self._agent_pickup_targets[i] = -1
                target = observations[agent_id]["self_delivery_target"]
            else:
                if self._agent_pickup_targets[i] == -1:
                    for j in range(NUM_REQUESTS):
                        if j not in self._agent_pickup_targets:
                            self._agent_pickup_targets[i] = j
                            break
                target = observations[agent_id]["requests"][self._agent_pickup_targets[i]][0:2]

            action_idxs = np.clip(target - observations[agent_id]["self_position"], -1, 1) + 1

            # Randomly rotate action to avoid stuck due to collision
            if np.random.uniform() < ROTATE_ACTION_PROB:
                action_idxs[0] = (action_idxs[0] + 1) % 3
            if np.random.uniform() < ROTATE_ACTION_PROB:
                action_idxs[1] = (action_idxs[1] + 1) % 3

            action = action_idxs[0] * 3 + action_idxs[1]
            action_dict[agent_id] = action

        return action_dict


if __name__ == "__main__":
    step_time_buffer: Deque[float] = deque([], maxlen=10)
    render_time_buffer: Deque[float] = deque([], maxlen=10)

    env = WarehouseDiscrete()
    solver = WarehouseDiscreteSolver()
    observations = env.reset()

    acc_rewards = [0.0, 0.0]
    done = False
    while not done:
        action_dict = solver.compute_action(observations)
        start_time = time.time()
        observations, rewards, dones, infos = env.step(action_dict=action_dict)
        step_time_buffer.append(1.0 / (time.time() - start_time))
        env.render()
        render_time_buffer.append(1.0 / (time.time() - start_time))

        acc_rewards = [acc_rewards[i] + rewards[f"{i}"] for i in range(NUM_AGENTS)]
        done = dones["__all__"]

        print("\n=== Status ===")
        print("Rewards:", *acc_rewards)
        print(
            f"Step FPS: {sum(step_time_buffer) / len(step_time_buffer)}, render FPS: {sum(render_time_buffer) / len(render_time_buffer)}"
        )
