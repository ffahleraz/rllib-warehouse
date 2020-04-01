import time
from typing import Deque
from collections import deque

from discrete import WarehouseDiscrete


if __name__ == "__main__":
    step_time_buffer: Deque[float] = deque([], maxlen=10)
    render_time_buffer: Deque[float] = deque([], maxlen=10)

    env = WarehouseDiscrete()
    observations = env.reset()
    for _, observation in observations.items():
        assert env.observation_space.contains(observation)

    done = False
    while not done:
        action_dict = {str(i): env.action_space.sample() for i in range(len(observations))}
        print(env.action_space.sample())

        start_time = time.time()
        observations, rewards, dones, infos = env.step(action_dict=action_dict)
        step_time_buffer.append(1.0 / (time.time() - start_time))
        env.render()
        render_time_buffer.append(1.0 / (time.time() - start_time))

        for _, observation in observations.items():
            assert env.observation_space.contains(observation)

        done = dones["__all__"]
        print(
            f"Step FPS: {sum(step_time_buffer) / len(step_time_buffer)}, render FPS: {sum(render_time_buffer) / len(render_time_buffer)}"
        )
