from sampleEnv import SampleEnv
from QLearningAgent import QLearningAgent

def main():
    m, n = 15, 15  # Define a grid size
    env = SampleEnv(m, n)
    agent = QLearningAgent(env)

    episodes = 10000  # Number of episodes to train
    for episode in range(episodes):
        print(f"Episode {episode + 1}")
        env.reset()
        state = env.grid.copy()
        done = False
        total_reward = 0
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            total_reward+=reward
            state = next_state
            print(f"Action: {action}, Reward: {reward}")

        print(f"Episode {episode + 1} finished\n")
        #printthe total reward for each episode
        print(f"Total Reward: {total_reward}")

    # Optional: Plot the final grid after training
    env.plot_grid(agent.getQTable())

if __name__ == "__main__":
    main()
