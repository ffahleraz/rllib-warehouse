import argparse

import json
import glob
import os
import gym
import ray
from ray.rllib.agents.sac.sac import SACTrainer
from ray.tune.registry import register_env

from warehouse import (
    WarehouseDiscreteSmall,
    WarehouseDiscreteMedium,
    WarehouseDiscreteLarge,
    WarehouseContinuousSmall,
)


def main(trial_dir: str, iteration: int, render: bool) -> None:
    ray.init()

    params = json.load(open(os.path.join(trial_dir, "params.json"), "rb"))

    env_map = {
        "WarehouseDiscreteSmall-v0": WarehouseDiscreteSmall,
        "WarehouseDiscreteMedium-v0": WarehouseDiscreteMedium,
        "WarehouseDiscreteLarge-v0": WarehouseDiscreteLarge,
        "WarehouseContinuousSmall-v0": WarehouseContinuousSmall,
    }
    for key in env_map:
        register_env(key, lambda _: env_map[key]())

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
    done = False
    while not done:
        action_dict = {
            f"{i}": trainer.compute_action(observations[f"{i}"]) for i in range(env.num_agents)
        }
        observations, rewards, dones, infos = env.step(action_dict=action_dict)
        done = dones["__all__"]
        env.render()


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
