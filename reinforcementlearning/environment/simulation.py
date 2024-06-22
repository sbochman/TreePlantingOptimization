from sampleEnv import SampleEnv
from agent import Agent

def main():
    m, n = 5, 5  # Define a smaller grid for simplicity
    env = SampleEnv(m, n)
    agent = Agent(env)

    episodes = 10
    for episode in range(episodes):
        print(f"Episode {episode + 1}")
        env.reset()
        done = False

        while not done:
            action = agent.choose_action()
            reward, done = env.step(action)
            agent.learn(action, reward)
            env.plot_grid()
            print(f"Planted at {action}, Reward: {reward}")

        print(f"Episode {episode + 1} finished\n")

if __name__ == "__main__":
    main()
