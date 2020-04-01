import argparse

import ray
from ray import tune
from ray.rllib.agents.sac.sac import SACTrainer
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print

from warehouse import WarehouseContinuous


def main(experiment_name: str, restore_dir: str, num_iterations: int) -> None:
    ray.init()
    register_env("WarehouseContinuous-v0", lambda _: WarehouseContinuous())

    tune.run(
        "PPO",
        name=experiment_name,
        stop={"training_iteration": num_iterations},
        restore=restore_dir,
        checkpoint_freq=10,
        checkpoint_at_end=True,
        config={"env": "WarehouseContinuous-v0", "num_gpus": 1, "num_workers": 1,},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name", type=str, help="experiment name")
    parser.add_argument(
        "-n", "--num_iterations", type=int, required=True, help="number of training iterations"
    )
    parser.add_argument("-r", "--restore_dir", type=str, help="path to the folder to restore model")
    args = parser.parse_args()
    main(
        experiment_name=args.experiment_name,
        restore_dir=args.restore_dir,
        num_iterations=args.num_iterations,
    )
