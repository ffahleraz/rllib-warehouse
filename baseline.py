import typing
import time

import numpy as np

import warehouse
from warehouse import Warehouse


NUM_AGENTS: int = warehouse.NUM_AGENTS
NUM_REQUESTS: int = warehouse.NUM_REQUESTS


class WarehouseSolver:
    def __init__(self) -> None:
        self._agent_pickup_targets = [-1] * NUM_AGENTS

    def compute_action(
        self, observations: typing.Dict[str, typing.Dict[str, np.ndarray]]
    ) -> typing.Dict[str, np.ndarray]:
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

            action = target - observations[agent_id]["self_position"]
            action /= np.linalg.norm(action)
            action_dict[agent_id] = action

        return action_dict


if __name__ == "__main__":
    env = Warehouse()
    solver = WarehouseSolver()

    observations = env.reset()
    assert env.observation_space.contains(observations["0"])

    acc_rewards = [0.0, 0.0]
    done = False
    while not done:
        start_time = time.time()
        observations, rewards, dones, infos = env.step(
            action_dict=solver.compute_action(observations)
        )
        step_fps = 1.0 / (time.time() - start_time)
        env.render()
        render_fps = 1.0 / (time.time() - start_time)

        acc_rewards = [acc_rewards[i] + rewards[f"{i}"] for i in range(NUM_AGENTS)]
        done = dones["__all__"]

        print("\n=== Status ===")
        print("Rewards:", *acc_rewards)
        print(f"Step FPS: {step_fps}, render FPS: {render_fps}")
