import time
import argparse
from typing import Deque
from collections import deque

from warehouse import WarehouseHardSmall, WarehouseHardMedium, WarehouseHardLarge


def main(env_variant: str) -> None:
    step_time_buffer: Deque[float] = deque([], maxlen=10)
    render_time_buffer: Deque[float] = deque([], maxlen=10)

    if env_variant == "small":
        env = WarehouseHardSmall()
    elif env_variant == "medium":
        env = WarehouseHardMedium()
    else:
        env = WarehouseHardLarge()

    observations = env.reset()
    for _, observation in observations.items():
        assert env.observation_space.contains(observation)

    done = False
    while not done:
        action_dict = {str(i): env.action_space.sample() for i in range(len(observations))}

        start_time = time.time()
        observations, rewards, dones, infos = env.step(action_dict=action_dict)
        step_time_buffer.append(1.0 / (time.time() - start_time))
        env.render()
        render_time_buffer.append(1.0 / (time.time() - start_time))

        for _, observation in observations.items():
            assert env.observation_space.contains(observation)

        done = dones["__all__"]
        print(
            f"Step avg FPS: {sum(step_time_buffer) / len(step_time_buffer)}, render avg FPS: {sum(render_time_buffer) / len(render_time_buffer)}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "env_variant", type=str, choices=["small", "medium", "large"], help="environment variant"
    )
    args = parser.parse_args()
    main(env_variant=args.env_variant)
