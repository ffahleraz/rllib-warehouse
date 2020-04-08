import argparse
import time

import json
import glob
import os
import gym
import ray
from ray.rllib.agents.sac.sac import SACTrainer
from ray.tune.registry import register_env

from warehouse import (
    WarehouseSmall,
    WarehouseMedium,
    WarehouseLarge,
    WarehouseHardSmall,
    WarehouseHardMedium,
    WarehouseHardLarge,
)


def main(trial_dir: str, iteration: int, render: bool) -> None:
    ray.init()

    params = json.load(open(os.path.join(trial_dir, "params.json"), "rb"))

    env_map = {
        "WarehouseSmall-v0": WarehouseSmall,
        "WarehouseMedium-v0": WarehouseMedium,
        "WarehouseLarge-v0": WarehouseLarge,
        "WarehouseHardSmall-v0": WarehouseHardSmall,
        "WarehouseHardMedium-v0": WarehouseHardMedium,
        "WarehouseHardLarge-v0": WarehouseHardLarge,
    }
    for key, val in env_map.items():
        register_env(key, lambda _: val())

    checkpoint_paths = glob.glob(os.path.join(trial_dir, "checkpoint_*"))
    checkpoint_iterations = sorted(
        [int(os.path.basename(path).split("_")[1]) for path in checkpoint_paths]
    )
    iteration_choice = (
        checkpoint_iterations[-1]
        if iteration == -1
        else min(checkpoint_iterations, key=lambda x: abs(x - iteration))
    )
    restore_dir = os.path.join(
        trial_dir, f"checkpoint_{iteration_choice}", f"checkpoint-{iteration_choice}"
    )

    trainer = SACTrainer(config=params)
    trainer.restore(restore_dir)

    if iteration == -1:
        print(f"Loading the lastest checkpoint at iteration {iteration_choice}.")
    else:
        print(
            f"Checkpoint at iteration {iteration} doesn't exist, loading the closest one at {iteration_choice} instead."
        )

    env = env_map[params["env"]]()
    observations = env.reset()

    acc_rewards = [0.0 for i in range(len(observations))]
    done = False
    step_count = 0
    while not done:
        action_dict = {
            f"{i}": trainer.compute_action(observations[f"{i}"]) for i in range(env.num_agents)
        }
        observations, rewards, dones, infos = env.step(action_dict=action_dict)

        acc_rewards = [acc_rewards[i] + rewards[f"{i}"] for i in range(env.num_agents)]
        done = dones["__all__"]

        print(f"\n=== Step {step_count} ===")
        print("Rewards:", *acc_rewards)

        step_count += 1

        if render:
            env.render()
            time.sleep(0.1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "trial_dir", type=str, help="path to the folder of the saved training trial"
    )
    parser.add_argument(
        "-i",
        "--iteration",
        type=int,
        default=-1,
        help="the iteration of the checkpoint to be loaded",
    )
    parser.add_argument(
        "-r", "--render", help="render the environment on each step", action="store_true"
    )
    args = parser.parse_args()
    main(trial_dir=args.trial_dir, iteration=args.iteration, render=args.render)
