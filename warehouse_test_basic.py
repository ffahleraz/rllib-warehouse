from warehouse import Warehouse

if __name__ == "__main__":
    env = Warehouse()
    observations = env.reset()
    done = False
    while not done:
        action_dict = {
            str(i): env.action_space.sample() for i in range(len(observations))
        }
        observations, rewards, dones, infos = env.step(action_dict=action_dict)
        done = dones["__all__"]
        env.render()
        # print(action_dict)
        # print(observations)
