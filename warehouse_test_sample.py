from warehouse import Warehouse

if __name__ == "__main__":
    env = Warehouse()
    # print(env.observation_space.sample())
    observations = env.reset()
    # print(observations["0"])
    done = False
    while not done:
        action_dict = {str(i): env.action_space.sample() for i in range(len(observations))}
        observations, rewards, dones, infos = env.step(action_dict=action_dict)
        done = dones["__all__"]
        env.render()
        # print(env._episode_time)
        # break
        # print(action_dict)
        # print(observations)
