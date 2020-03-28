import time

from warehouse import Warehouse


if __name__ == "__main__":
    env = Warehouse()
    observations = env.reset()
    done = False
    while not done:
        action_dict = {str(i): env.action_space.sample() for i in range(len(observations))}

        start_time = time.time()
        observations, rewards, dones, infos = env.step(action_dict=action_dict)
        step_fps = 1.0 / (time.time() - start_time)
        env.render()
        render_fps = 1.0 / (time.time() - start_time)

        for _, observation in observations.items():
            assert env.observation_space.contains(observation)

        done = dones["__all__"]
        print(f"Step FPS: {step_fps}, render FPS: {render_fps}")
