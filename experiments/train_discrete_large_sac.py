import argparse

import ray
from ray import tune
from ray.rllib.agents.sac.sac import SACTrainer
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print

from warehouse import WarehouseDiscreteLarge


def main(experiment_name: str, restore_dir: str, num_iterations: int) -> None:
    ray.init()
    register_env("WarehouseDiscreteLarge-v0", lambda _: WarehouseDiscreteLarge())

    tune.run(
        "SAC",
        name=experiment_name,
        stop={"training_iteration": num_iterations},
        restore=restore_dir,
        checkpoint_freq=100,
        checkpoint_at_end=True,
        config={
            "env": "WarehouseDiscreteLarge-v0",
            "Q_model": {"hidden_activation": "relu", "hidden_layer_sizes": (1024, 512, 256, 64),},
            "policy_model": {
                "hidden_activation": "relu",
                "hidden_layer_sizes": (1024, 512, 256, 64),
            },
            "horizon": 600,
            "soft_horizon": False,
            "no_done_at_end": True,
            "normalize_actions": False,
            "timesteps_per_iteration": 600,
            "learning_starts": 6000,
            "buffer_size": 240000,
            "prioritized_replay": False,
            "target_network_update_freq": 1,
            "num_gpus": 1,
            "num_workers": 4,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name", type=str, help="experiment name")
    parser.add_argument("num_iterations", type=int, help="number of training iterations")
    parser.add_argument("-r", "--restore_dir", type=str, help="path to the folder to restore model")
    args = parser.parse_args()
    main(
        experiment_name=args.experiment_name,
        restore_dir=args.restore_dir,
        num_iterations=args.num_iterations,
    )
