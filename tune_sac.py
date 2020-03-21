import argparse

import ray
from ray import tune
from ray.rllib.agents.sac.sac import SACTrainer
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print

from warehouse import Warehouse


def main(experiment_name: str, restore_dir: str, num_iterations: int) -> None:
    ray.init()
    register_env("Warehouse-v0", lambda _: Warehouse())

    tune.run(
        "SAC",
        name=experiment_name,
        stop={"training_iteration": num_iterations},
        restore=restore_dir,
        checkpoint_freq=10,
        checkpoint_at_end=True,
        config={
            "env": "Warehouse-v0",
            "Q_model": {"hidden_activation": "relu", "hidden_layer_sizes": (256, 256),},
            "policy_model": {"hidden_activation": "relu", "hidden_layer_sizes": (256, 256),},
            "normalize_actions": False,
            "no_done_at_end": True,
            "timesteps_per_iteration": 4000,
            "buffer_size": int(1e5),
            "learning_starts": 20000,
            "num_gpus": 0,
            "num_workers": 2,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-num", "--num_iterations", type=int, required=True, help="number of training iterations"
    )
    parser.add_argument("-n", "--name", type=str, required=True, help="experiment name")
    parser.add_argument("-r", "--restore_dir", type=str, help="path to the folder to restore model")
    args = parser.parse_args()
    main(
        experiment_name=args.name, restore_dir=args.restore_dir, num_iterations=args.num_iterations,
    )
