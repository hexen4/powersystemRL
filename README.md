# powersystemRL

1) Implementation of https://ieeexplore.ieee.org/document/10345718 onto https://www.sciencedirect.com/science/article/pii/S2213138821002356
skeleton code based off: https://github.com/GitX123/microgrid-ems-drl

# 1. Understand the Existing Code Structure:
Review the repository to understand key modules such as environment setup, DRL algorithms (PPO, TD3), data handling, and reward functions.
# 2. Integrate TCSAC Algorithm:
Create a new DRL agent by adapting the existing TD3, introducing the triplet-critic mechanism (within model.py) to handle over/underestimation bias as required by the TCSAC algorithm + change NN architecture. Also include entropy reg.(within TCSAC agent), and change hyperparameters.
  Ensure to include operating constraints in Actor/Agent (action constraints, reward penalisation for violating, state and environment constraints, prioritize transitions where constraint violations occur, helping the agent learn faster how to avoid such violations)
# 3. Modify the Environment for Interval Optimization:
Update the environment to handle uncertainty in renewable energy sources instead of battery management. Use interval variables for upper and lower bounds to simulate energy fluctuations.
# 4. Redefine Reward Function:
Modify the reward structure to address multi-objective optimization (economic cost, network loss, system stability) by introducing a composite reward function that balances these objectives.
# 5. Comprehensive Experience Replay:
Implement comprehensive experience replay for more efficient training, adapting from standard replay buffers to prioritize diverse experiences based on the novel prioritization scheme. (within buffer.py)
# 6. Test and Validate (need to ask Pratik for source files for IEEE bus):
Use the provided IEEE bus system or similar microgrid setups to validate your TCSAC implementation. Ensure the agent can handle real-time decision-making as required in the 2024 paper.
# 7. Iterate on Improvements:
After the initial integration, continue refining the algorithm based on performance metrics like convergence speed, stability, and handling of large state-action spaces.
