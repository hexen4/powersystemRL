'''
class
- Buffer
- PrioritizedReplayBuffer
- ReplayBuffer
'''
from torchrl.data import PrioritizedReplayBuffer, ListStorage
import numpy as np
import torch
from .. import setting


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
        rho = rho_e + last_transition[-1]
        for transition in transitions:
            transition.append(rho)
            super().add(transition) # TODO need to convert to tensor

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
        #combined = list(zip(H1, info1["index"])) + list(zip(H2, info2["index"]))
        combined = [H1,H2]
        # Sort combined transitions by priority score (ρ)
        sorted_combined = sorted(combined, key=lambda x: x[-1], reverse=True)

        # Select top batch_size transitions
        top_transitions = sorted_combined[:batch_size]

        # # Extract transitions and their indices
        # sampled_transitions = [t[0] for t in top_transitions]
        # sampled_indices = [t[1] for t in top_transitions]

        # # Update priority indices to mark as updated
        # self.mark_update(torch.tensor(sampled_indices))

        return top_transitions
    
    def __len__(self):
        return len(self.storage)




        # TODO need to sample based on priority value? not randomly? what priority value is being set? i.e. how it is smapling


if __name__ == "main":
    buffer = ExtendedPrioritizedReplayBuffer()
    buffer.add_episodes([1,2,3,4,5,6,7,8,9,10],1)
    buffer.add_episodes([1,2,3,4,5,6,7,8,9,10],2)
    buffer.add_episodes([1,2,3,4,5,6,7,8,9,10],3)
    buffer.add_episodes([1,2,3,4,5,6,7,8,9,10],4)
    buffer.add_episodes([1,2,3,4,5,6,7,8,9,10],5)
    buffer.add_episodes([1,2,3,4,5,6,7,8,9,10],6)
    buffer.add_episodes([1,2,3,4,5,6,7,8,9,10],7)
    buffer.add_episodes([1,2,3,4,5,6,7,8,9,10],8)
    buffer.add_episodes([1,2,3,4,5,6,7,8,9,10],9)
    buffer.add_episodes([1,2,3,4,5,6,7,8,9,10],10)
    buffer.add_episodes([1,2,3,4,5,6,7,8,9,10],11)
    buffer.add_episodes([1,2,3,4,5,6,7,8,9,10],12)
    buffer.add_episodes([1,2,3,4,5,6,7,8,9,10],13)
    buffer.add_episodes([1,2,3,4,5,6,7,8,9,10],14)
    buffer.add_episodes([1,2,3,4,5,6,7,8,9,10],15)
    buffer.add_episodes([1,2,3,4,5,6,7,8,9,10],16)
    buffer.add_episodes([1,2,3,4,5,6,7,8,9,10],17)
    buffer.add_episodes([1,2,3,4,5,6,7,8,9,10],18)
