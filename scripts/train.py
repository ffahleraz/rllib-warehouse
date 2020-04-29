import argparse

import yaml
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.tune.tune import run_experiments

from warehouse import (
    Warehouse2,
    Warehouse4,
    Warehouse6,
    Warehouse8,
    Warehouse10,
    Warehouse12,
    Warehouse14,
    Warehouse16,
)


def main(config_path: str) -> None:
    ray.init()

    register_env("Warehouse2-v0", lambda _: Warehouse2())
    register_env("Warehouse4-v0", lambda _: Warehouse4())
    register_env("Warehouse6-v0", lambda _: Warehouse6())
    register_env("Warehouse8-v0", lambda _: Warehouse8())
    register_env("Warehouse10-v0", lambda _: Warehouse10())
    register_env("Warehouse12-v0", lambda _: Warehouse12())
    register_env("Warehouse14-v0", lambda _: Warehouse14())
    register_env("Warehouse16-v0", lambda _: Warehouse16())

    with open(args.config_path) as config_file:
        experiments = yaml.safe_load(config_file)

    run_experiments(experiments)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="path to the experiment config file")
    args = parser.parse_args()
    main(config_path=args.config_path)
