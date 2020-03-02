from warehouse import Warehouse

if __name__ == "__main__":
    env = Warehouse()
    env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action=action)
        print(action)
        print(observation)
