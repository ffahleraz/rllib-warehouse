import argparse

import yaml
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.tune.tune import run_experiments

from warehouse import (
    WarehouseSmall,
    WarehouseMedium,
    WarehouseLarge,
    WarehouseHardSmall,
    WarehouseHardMedium,
    WarehouseHardLarge,
)


def main(config_path: str) -> None:
    ray.init()

    register_env("WarehouseSmall-v0", lambda _: WarehouseSmall())
    register_env("WarehouseMedium-v0", lambda _: WarehouseMedium())
    register_env("WarehouseLarge-v0", lambda _: WarehouseLarge())
    register_env("WarehouseHardSmall-v0", lambda _: WarehouseHardSmall())
    register_env("WarehouseHardMedium-v0", lambda _: WarehouseHardMedium())
    register_env("WarehouseHardLarge-v0", lambda _: WarehouseHardLarge())

    with open(args.config_path) as config_file:
        experiments = yaml.safe_load(config_file)

    run_experiments(experiments)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="path to the experiment config file")
    args = parser.parse_args()
    main(config_path=args.config_path)
