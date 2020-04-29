import time
import argparse
from typing import Deque
from collections import deque

from warehouse import Warehouse, WarehouseSmall, WarehouseMedium, WarehouseLarge

from solvers import WarehouseSolver, WarehouseRandomSolver, WarehouseGreedySolver


def main(solver_type: str, env_variant: str) -> None:
    step_time_buffer: Deque[float] = deque([], maxlen=10)
    think_time_buffer: Deque[float] = deque([], maxlen=10)
    render_time_buffer: Deque[float] = deque([], maxlen=10)

    env: Warehouse
    if env_variant == "small":
        env = WarehouseSmall()
    elif env_variant == "medium":
        env = WarehouseMedium()
    elif env_variant == "large":
        env = WarehouseLarge()

    solver: WarehouseSolver
    if solver_type == "random":
        solver = WarehouseRandomSolver(action_space=env.action_space)
    elif solver_type == "greedy":
        solver = WarehouseGreedySolver(num_agents=env.num_agents, num_requests=env.num_requests)

    observations = env.reset()
    for _, observation in observations.items():
        assert env.observation_space.contains(observation)

    acc_rewards = [0.0 for i in range(env.num_agents)]
    done = False
    step_count = 0
    while not done:
        start_time = time.time()
        env.render()
        render_time_buffer.append(1.0 / (time.time() - start_time))

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

        print(f"\n=== Step {step_count} ===")
        print("Rewards:", *acc_rewards)
        print(
            f"Step avg FPS: {sum(step_time_buffer) / len(step_time_buffer)}, think avg FPS: {sum(think_time_buffer) / len(think_time_buffer)}, render avg FPS: {sum(render_time_buffer) / len(render_time_buffer)}"
        )

        step_count += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("solver_type", type=str, choices=["random", "greedy"], help="solver type")
    parser.add_argument(
        "env_variant", type=str, choices=["small", "medium", "large"], help="environment variant"
    )
    args = parser.parse_args()
    main(solver_type=args.solver_type, env_variant=args.env_variant)
