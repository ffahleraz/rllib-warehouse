from warehouse import Warehouse

if __name__ == "__main__":
    env = Warehouse()
    observations = env.reset()
    done = False
    while not done:
        action_dict = {"0": env.action_space.sample()}
        observations, rewards, dones, infos = env.step(action_dict=action_dict)
        done = dones["__all__"]
        env.render()
        # print(action_dict)
        # print(observations)
