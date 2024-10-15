# powersystemRL

1) Implementation of https://ieeexplore.ieee.org/document/10345718 onto https://www.sciencedirect.com/science/article/pii/S2213138821002356


# 1. Understand the Existing Code Structure:
Review the repository to understand key modules such as environment setup, DRL algorithms (PPO, TD3), data handling, and reward functions.
# 2. Integrate TCSAC Algorithm:
Create a new DRL agent by adapting the existing TD3 or SAC code, introducing the triplet-critic mechanism to handle over/underestimation bias as required by the TCSAC algorithm.
# 3. Modify the Environment for Interval Optimization:
Update the environment to handle uncertainty in renewable energy sources (wind, solar). Use interval variables for upper and lower bounds to simulate energy fluctuations.
# 4. Redefine Reward Function:
Modify the reward structure to address multi-objective optimization (economic cost, network loss, system stability) by introducing a composite reward function that balances these objectives.
# 5. Comprehensive Experience Replay:
Implement comprehensive experience replay for more efficient training, adapting from standard replay buffers to prioritize diverse experiences based on the novel prioritization scheme.
# 6. Test and Validate:
Use the provided IEEE bus system or similar microgrid setups to validate your TCSAC implementation. Ensure the agent can handle real-time decision-making as required in the 2024 paper.
# 7. Iterate on Improvements:
After the initial integration, continue refining the algorithm based on performance metrics like convergence speed, stability, and handling of large state-action spaces.
