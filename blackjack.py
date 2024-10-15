import gymnasium as gym
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class BlackjackAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            env: The training environment
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error =[]
        self.rewards = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
            """
            Returns the best action with probability (1 - epsilon)
            otherwise a random action with probability epsilon to ensure exploration.
            """
            # with probability epsilon return a random action to explore the environment
            if np.random.random() < self.epsilon:
                return self.env.action_space.sample()
            # with probability (1 - epsilon) act greedily (exploit)
            else:
                return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def train(self, num_episodes):
        for episode in tqdm(range(num_episodes)):
            total_reward = 0
            observation, info = self.env.reset()
            done = False
            while not done:
                action = self.get_action(observation)
                next_observation, reward, done, truncated, info = self.env.step(action)
                self.update(observation, action, reward, next_observation, done)
                observation = next_observation
                total_reward += reward
            self.rewards.append(total_reward)

        # Plot the rewards
        plt.plot(self.rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Total Reward per Episode')
        plt.show()

if __name__ == "__main__":
    learning_rate = 0.01
    n_episodes = 100_000
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
    final_epsilon = 0.1
    env = gym.make("Blackjack-v1", sab=False)
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

    agent = BlackjackAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon
    )

    agent.train(n_episodes)