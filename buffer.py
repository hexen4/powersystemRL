'''
class
- Buffer
- PrioritizedReplayBuffer
- ReplayBuffer
'''
from torchrl.data import PrioritizedReplayBuffer, ListStorage
import numpy as np
import torch
from setting import *


class ExtendedPrioritizedReplayBuffer(PrioritizedReplayBuffer):
    def __init__(self, capacity = REPLAY_BUFER_SIZE, eta=0.9, **kwargs):
        """
        Extended Prioritized Replay Buffer with recent score prioritization.

        Args:
            capacity (int): Maximum buffer size.
            beta (float): Importance sampling exponent.
            eta (float): Recent score hyperparameter.
            kwargs: Additional arguments for base PrioritizedReplayBuffer.
        """
        super().__init__(alpha = ALPHA, beta = BETA, storage=ListStorage(max_size=capacity) , **kwargs) #LazyMemmapStorage?
        self.capacity = capacity
        self.rho_min = RHO_MIN
        self.eta = ETA

    def add_episodes(self, transitions,episode_counter):
        """
        Adds an entire episode to the replay buffer, calculating the recent score ρe.

        Args:
            transitions (list): List of transition tuples (state, action, reward, next_state, done).
            episode_id (int): Unique identifier for the episode.
        """
       
        #episode_length = len(transitions)

        # Calculate ρe for the episode
        rho_e = max(self.capacity * self.eta **(1000/(episode_counter+1)), self.rho_min) # TODO need counter for current stage    

        # Assign recent score to all transitions in the episode
        last_transition = transitions[-1]
        rho = rho_e + last_transition[-2]
        for transition in transitions:
            transition.append(torch.tensor(rho, dtype=torch.float32))
            super().add(transition)

    def sample(self, batch_size):
        """
        Samples a prioritized batch of transitions from the replay buffer.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            list: Sampled batch of transitions sorted by priority.
        """
        # Sample two mini-batches, H1 and H2
        H1, info1 = super().sample(batch_size, return_info=True)
        H2, info2 = super().sample(batch_size, return_info=True)

        # Combine H1 and H2
        combined = [torch.cat([h1, h2], dim=0) for h1, h2 in zip(H1, H2)]
        # Sort combined transitions by priority score (ρ)
        sorted_indices = torch.argsort(combined[-1], descending=True)
        # Select top batch_size transitions
        sorted_combined = [tensor[sorted_indices] for tensor in combined]
        top_transitions = [tensor[:batch_size] for tensor in sorted_combined]

        return top_transitions 
    
    def __len__(self):
        return len(self.storage)



# if __name__ == '__main__':
#     buffer = ExtendedPrioritizedReplayBuffer()
#     for episode_id in range(1, 19):  # Add 18 episodes
#         transitions = []
#         for t in range(10):  # Assume each episode has 10 transitions
#             state = torch.rand(5, dtype=torch.float32)  # Random tensor state of size 5
#             action = torch.rand(2, dtype=torch.float32)  # Random tensor action of size 2
#             reward = torch.tensor(np.random.rand(), dtype=torch.float32)  # Random reward tensor
#             next_state = torch.rand(5, dtype=torch.float32)  # Random tensor next state
#             done = torch.tensor(t == 9, dtype=torch.bool)  # Boolean flag as tensor
            
#             transitions.append([state, action, next_state, reward,done])

#         buffer.add_episodes(transitions, episode_id)

#     # Display stored buffer content
#     print(buffer.sample(10))