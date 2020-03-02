import ray
from ray.rllib.agents.sac.sac import SACTrainer
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print

from warehouse import Warehouse

if __name__ == "__main__":
    ray.init()

    register_env("Warehouse-v0", lambda _: Warehouse())
    trainer = SACTrainer(env="Warehouse-v0", config={"normalize_actions": False})

    for i in range(10):
        print("== Iteration", i, "==")
        print(pretty_print(trainer.train()))
