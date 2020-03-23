import typing

import numpy as np

import warehouse
from warehouse import Warehouse


NUM_AGENTS: int = warehouse.NUM_AGENTS


if __name__ == "__main__":
    env = Warehouse()
    observations = env.reset()
    done = False
    while not done:
        action_dict = {f"{i}": np.array([-1.0, -1.0], dtype=np.float32) for i in range(NUM_AGENTS)}

        if observations["0"]["self_availability"][0] == 0:
            target = observations["0"]["self_delivery_target"][0:2]
        else:
            target = observations["0"]["requests"][0, 0:2]
        action = target - observations["0"]["self_position"]
        action /= np.linalg.norm(action)
        action_dict["0"] = action

        observations, rewards, dones, infos = env.step(action_dict=action_dict)
        done = dones["__all__"]
        env.render()
