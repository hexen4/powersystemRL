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
        super().__init__(alpha = ALPHA, beta = BETA, storage=ListStorage(max_size=capacity) , **kwargs)
        self.capacity = capacity
        self.rho_min = RHO_MIN
        self.eta = ETA

    def add_episodes(self, transitions):
        """
        Adds an entire episode to the replay buffer, calculating the recent score ρe.

        Args:
            transitions (list): List of transition tuples (state, action, reward, next_state, done).
            episode_id (int): Unique identifier for the episode.
        """
        N = self.capacity
        episode_length = len(transitions)

        # Calculate ρe for the episode
        rho_e = max((N * self.eta **(1000/ episode_length)), self.rho_min) # TODO need counter for current stage    

        # Assign recent score to all transitions in the episode
        for t, transition in enumerate(transitions):
            state, action, reward, next_state,done = transition # TODO fix this line
            rho_f = reward if done else 0  # Terminal state reward
            rho = rho_e + rho_f
            transition_with_priority = (state, action, reward, next_state, done, rho)
            super().add(transition_with_priority)  # Add to base buffer

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
        combined = list(zip(H1, info1["index"])) + list(zip(H2, info2["index"]))

        # Sort combined transitions by priority score (ρ)
        sorted_combined = sorted(combined, key=lambda x: x[0][-1], reverse=True)

        # Select top batch_size transitions
        top_transitions = sorted_combined[:batch_size]

        # Extract transitions and their indices
        sampled_transitions = [t[0] for t in top_transitions]
        sampled_indices = [t[1] for t in top_transitions]

        # Update priority indices to mark as updated
        self.mark_update(torch.tensor(sampled_indices))

        return sampled_transitions
    
    def __len__(self):
        return len(self.storage)