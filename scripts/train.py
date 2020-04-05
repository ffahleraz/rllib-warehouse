import argparse

import yaml
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.tune.tune import run_experiments

from warehouse import (
    WarehouseGridSmall,
    WarehouseGridMedium,
    WarehouseGridLarge,
    WarehouseSmall,
)


def main(config_path: str) -> None:
    ray.init()

    register_env("WarehouseGridSmall-v0", lambda _: WarehouseGridSmall())
    register_env("WarehouseGridMedium-v0", lambda _: WarehouseGridMedium())
    register_env("WarehouseGridLarge-v0", lambda _: WarehouseGridLarge())
    register_env("WarehouseSmall-v0", lambda _: WarehouseSmall())

    with open(args.config_path) as config_file:
        experiments = yaml.safe_load(config_file)

    run_experiments(experiments)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="path to the experiment config file")
    args = parser.parse_args()
    main(config_path=args.config_path)
