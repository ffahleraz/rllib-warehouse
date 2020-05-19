import time
import argparse
from typing import Dict, Deque, Type
from collections import deque

from warehouse import (
    WarehouseSmall,
    WarehouseMedium,
    WarehouseLarge,
)

from solvers import WarehouseRandomGreedySolver


def main(env_size: str, num_agents: int, random_action_prob: float, render: bool) -> None:
    step_time_buffer: Deque[float] = deque([], maxlen=10)
    think_time_buffer: Deque[float] = deque([], maxlen=10)
    render_time_buffer: Deque[float] = deque([], maxlen=10)

    env_type_map: Dict[str, Type] = {
        "small": WarehouseSmall,
        "medium": WarehouseMedium,
        "large": WarehouseLarge,
    }
    env_type = env_type_map[env_size]
    env = env_type(num_agents)

    solver = WarehouseRandomGreedySolver(
        num_agents=env.num_agents,
        num_requests=env.num_requests,
        random_action_prob=random_action_prob,
        action_space=env.action_space,
    )

    observations = env.reset()
    for _, observation in observations.items():
        assert env.observation_space.contains(observation)

    acc_rewards = [0.0 for i in range(env.num_agents)]
    done = False
    step_count = 0
    while not done:
        if render:
            start_time = time.time()
            env.render(animate=True)
            render_time_buffer.append(
                (1.0 / (time.time() - start_time)) * env.animate_frames_per_step
            )

        start_time = time.time()
        action_dict = solver.compute_action(observations)
        think_time_buffer.append(1.0 / (time.time() - start_time))

        start_time = time.time()
        observations, rewards, dones, infos = env.step(action_dict=action_dict)
        step_time_buffer.append(1.0 / (time.time() - start_time))

        for _, observation in observations.items():
            assert env.observation_space.contains(observation)

        acc_rewards = [acc_rewards[i] + rewards[f"{i}"] for i in range(env.num_agents)]
        done = dones["__all__"]

        if render:
            print(f"\n=== Step {step_count} ===")
            print("Rewards:", *acc_rewards)
            print(f"Total: {sum(acc_rewards)}, Per Agent: {sum(acc_rewards) / len(acc_rewards)}")
            print(
                f"Step avg FPS: {sum(step_time_buffer) / len(step_time_buffer)}, think avg FPS: {sum(think_time_buffer) / len(think_time_buffer)}, render avg FPS: {sum(render_time_buffer) / len(render_time_buffer)} / {env.animate_frames_per_step * env.animate_steps_per_second}"
            )

        step_count += 1

    print(f"\n=== Done ({step_count} steps) ===")
    print("Rewards:", *acc_rewards)
    print(f"Total: {sum(acc_rewards)}, Per Agent: {sum(acc_rewards) / len(acc_rewards)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "env_size", type=str, choices=["small", "medium", "large"], help="environment size"
    )
    parser.add_argument("num_agents", type=int, help="number of agents")
    parser.add_argument(
        "random_action_prob",
        type=float,
        help="probability of the solver to take random action [0.0, 1.0]",
    )
    parser.add_argument(
        "-r", "--render", help="render the environment on each step", action="store_true"
    )
    args = parser.parse_args()
    main(
        env_size=args.env_size,
        num_agents=args.num_agents,
        random_action_prob=args.random_action_prob,
        render=args.render,
    )
