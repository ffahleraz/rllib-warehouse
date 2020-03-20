import argparse

import ray
from ray.rllib.agents.sac.sac import SACTrainer
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print

from warehouse import Warehouse


def main(save_dir: str, load_dir: str, num_iterations: int) -> None:
    ray.init()

    register_env("Warehouse-v0", lambda _: Warehouse())
    trainer = SACTrainer(
        env="Warehouse-v0",
        config={
            "Q_model": {"hidden_activation": "relu", "hidden_layer_sizes": (256, 256),},
            "policy_model": {"hidden_activation": "relu", "hidden_layer_sizes": (256, 256),},
            "normalize_actions": False,
            "no_done_at_end": True,
            "timesteps_per_iteration": 4000,
            "buffer_size": int(1e4),
            "learning_starts": 20000,
            "num_gpus": 1,
            "num_workers": 2,
        },
    )
    if load_dir is not None:
        trainer.restore(load_dir)

    for i in range(num_iterations):
        print("==> Iteration Start")

        result = trainer.train()
        print(pretty_print(result))

        if i % 20 == 0:
            checkpoint = trainer.save(save_dir)
            print("==> Checkpoint saved at", checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--num_iterations", type=int, required=True, help="number of training iterations"
    )
    parser.add_argument(
        "-s", "--save", type=str, required=True, help="path to the folder to save model"
    )
    parser.add_argument("-l", "--load", type=str, help="path to the folder to load model")
    args = parser.parse_args()
    main(save_dir=args.save, load_dir=args.load, num_iterations=args.num_iterations)
