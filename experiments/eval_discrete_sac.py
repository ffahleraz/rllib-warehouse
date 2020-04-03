import argparse

import ray
from ray.rllib.agents.sac.sac import SACTrainer
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print

import warehouse
from warehouse import WarehouseDiscreteSmall, WarehouseDiscreteMedium, WarehouseDiscreteLarge


def main(env_variant: str, restore_dir: str) -> None:
    ray.init()

    if env_variant == "small":
        env = WarehouseDiscreteSmall()
    elif env_variant == "medium":
        env = WarehouseDiscreteMedium()
    else:
        env = WarehouseDiscreteLarge()

    register_env("WarehouseDiscrete-v0", lambda _: env)
    trainer = SACTrainer(env="WarehouseDiscrete-v0", config={"normalize_actions": False})
    trainer.restore(restore_dir)

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
        "env_variant", type=str, choices=["small", "medium", "large"], help="environment variant"
    )
    parser.add_argument("restore_dir", type=str, help="path to the folder to restore model")
    args = parser.parse_args()
    main(env_variant=args.env_variant, restore_dir=args.restore_dir)
