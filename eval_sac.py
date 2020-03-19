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
            "learning_starts": 10000,
            "timesteps_per_iteration": 1000,
            # "num_workers": 2,
        },
    )
    trainer.restore(
        "/Users/ffahleraz/ray_results/SAC_Warehouse-v0_2020-03-19_00-48-38_y4x_v_g/checkpoint_201/checkpoint-201"
    )

    env = Warehouse()
    observations = env.reset()
    done = False
    while not done:
        action_dict = {f"{i}": trainer.compute_action(observations[f"{i}"]) for i in range(4)}
        observations, rewards, dones, infos = env.step(action_dict=action_dict)
        done = dones["__all__"]
        env.render()
