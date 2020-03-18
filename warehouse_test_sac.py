import ray
from ray.rllib.agents.sac.sac import SACTrainer
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print

from warehouse import Warehouse

if __name__ == "__main__":
    ray.init()

    register_env("Warehouse-v0", lambda _: Warehouse())
    trainer = SACTrainer(
        env="Warehouse-v0",
        config={
            "normalize_actions": False,
            "no_done_at_end": True,
            "learning_starts": 20000,
            "timesteps_per_iteration": 2000,
        },
    )

    for i in range(100):
        print("== Iteration", i, "==")
        print(pretty_print(trainer.train()))
