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
            "Q_model": {"hidden_activation": "relu", "hidden_layer_sizes": (256, 256),},
            "policy_model": {"hidden_activation": "relu", "hidden_layer_sizes": (256, 256),},
            "normalize_actions": False,
            "no_done_at_end": False,
            "learning_starts": 20000,
            "timesteps_per_iteration": 4000,
        },
    )

    for i in range(2000):
        print("==> Iteration", i)

        result = trainer.train()
        print(pretty_print(result))

        if i % 50 == 0:
            checkpoint = trainer.save("saves/0")
            print("==> Checkpoint saved at", checkpoint)
