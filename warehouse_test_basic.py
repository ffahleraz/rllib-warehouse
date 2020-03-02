from warehouse import Warehouse

if __name__ == "__main__":
    env = Warehouse()
    observations = env.reset()
    done = False
    while not done:
        action_dict = {1: env.action_space.sample()}
        observations, rewards, dones, infos = env.step(action_dict=action_dict)
        print(action_dict)
        print(observations)
