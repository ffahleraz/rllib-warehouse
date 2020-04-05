import argparse

import yaml
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.tune.tune import run_experiments

from warehouse import (
    WarehouseDiscreteSmall,
    WarehouseDiscreteMedium,
    WarehouseDiscreteLarge,
    WarehouseContinuousSmall,
)


def main(config_path: str) -> None:
    ray.init()

    register_env("WarehouseDiscreteSmall-v0", lambda _: WarehouseDiscreteSmall())
    register_env("WarehouseDiscreteMedium-v0", lambda _: WarehouseDiscreteMedium())
    register_env("WarehouseDiscreteLarge-v0", lambda _: WarehouseDiscreteLarge())
    register_env("WarehouseContinuousSmall-v0", lambda _: WarehouseContinuousSmall())

    with open(args.config_path) as config_file:
        experiments = yaml.safe_load(config_file)

    run_experiments(experiments)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="path to the experiment config file")
    args = parser.parse_args()
    main(config_path=args.config_path)
