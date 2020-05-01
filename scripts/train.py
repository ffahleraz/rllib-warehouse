from typing import Dict, Any, Type
import argparse

import numpy as np
import yaml
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.tune.tune import run_experiments

from warehouse import (
    WarehouseSmallRandom,
    WarehouseMediumRandom,
    WarehouseLargeRandom,
)


def on_episode_end(info: Dict[str, Any]) -> None:
    episode = info["episode"]
    curr_num_agents = len(episode.agent_rewards)
    avg_agent_reward = sum([value for _, value in episode.agent_rewards.items()]) / curr_num_agents
    episode.custom_metrics["avg_agent_reward_all"] = [avg_agent_reward]
    episode.custom_metrics[f"avg_agent_reward_{curr_num_agents}"] = [avg_agent_reward]


def main(config_path: str) -> None:
    ray.init()

    env_type_map: Dict[str, Type] = {
        "WarehouseSmall-v0": WarehouseSmallRandom,
        "WarehouseMedium-v0": WarehouseMediumRandom,
        "WarehouseLarge-v0": WarehouseLargeRandom,
    }
    for key, val in env_type_map.items():
        register_env(key, lambda _: val())

    with open(args.config_path) as config_file:
        experiments = yaml.safe_load(config_file)

    experiment_name = list(experiments.keys())[0]
    experiments[experiment_name]["config"]["callbacks"] = {"on_episode_end": on_episode_end}

    run_experiments(experiments)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="path to the experiment config file")
    args = parser.parse_args()
    main(config_path=args.config_path)
