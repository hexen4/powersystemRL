import gymnasium as gym
from collections import defaultdict
import numpy as np
from tqdm import tqdm
def main():
    
    # Create the environment
    env = gym.make('CartPole-v1', render_mode = "human") #pettingzoo for multiagent
    
    # Reset the environment to the initial state
    observation, info = env.reset() #call with seed 
    observation_space = env.observation_space #returns the observation space of the environment
    for _ in range(100):
        env.render() #renders the environment
        
        # Sample a random action from the action space
        action = env.action_space.sample() #returns a random action from the action space
        
        # Take the action and observe the result
        observation, reward, done, truncated, info = env.step(action) #updates environment with actions returning the next agent observation, reward for taking action
        #terminated defined by MDP -> boolean value indicating if the episode is done
        
        #truncated is a boolean value indicating if outside the scope of MDP (e.g. time limit)
        # If the episode is done, reset the environment
        if done or truncated: 
            observation, info = env.reset() #required for calling next step
    
    env.close() #closes the environment, important when external software is used

if __name__ == "__main__":
    main()
