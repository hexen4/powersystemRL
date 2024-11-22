# powersystemRL

1) Implementation of https://ieeexplore.ieee.org/document/10345718 onto https://www.sciencedirect.com/science/article/pii/S2213138821002356
skeleton code based off: https://github.com/GitX123/microgrid-ems-drl

# PowerSystemRL
A Python-based repository for implementing advanced reinforcement learning (RL) algorithms for microgrid energy management. This project integrates the Triplet-Critic Soft Actor-Critic (TCSAC) algorithm to manage uncertainties in renewable energy, improve economic performance, and ensure system stability in microgrids. It also adapts concepts from demand response and renewable energy curtailment.

## Table of Contents
Features
Requirements
Installation
Usage
Training
Evaluation
Project Structure
Algorithms
Environment Details
Contributing
License
Features
Advanced RL Integration:

Implements Triplet-Critic Soft Actor-Critic (TCSAC).
Reduces value estimation bias with a triple-critic mechanism.
Incorporates entropy regularization for enhanced exploration.
Interval-Based Renewable Energy Modeling:

Simulates uncertainty in solar and wind power outputs.
Introduces interval variables for more realistic power system behavior.
Comprehensive Reward Function:

Balances economic cost, network loss, and system stability.
Penalizes renewable curtailment and state violations dynamically.
Prioritized Experience Replay:

Comprehensive replay buffer prioritizes critical transitions.
Enhances learning from constraint-violating experiences.
Multi-Objective Optimization:

Handles energy dispatch across economic, stability, and operational constraints.
Requirements
Python 3.8+
TensorFlow 2.x
Pandas, NumPy, Matplotlib
Pandapower
TensorFlow Addons
Scipy
Install the dependencies using the following command:

bash
Copy code
pip install -r requirements.txt
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/hexen4/powersystemRL.git
cd powersystemRL
Install dependencies:

bash
Copy code
pip install -r requirements.txt
(Optional) Set up a virtual environment:

bash
Copy code
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
Usage
Training
Run the main.py script to train a reinforcement learning agent. Specify the algorithm and environment parameters.

bash
Copy code
python main.py train --algorithm TCSAC --runs 5 --epochs 100
Key arguments:

--algorithm: Choose between PPO, TD3, and TCSAC.
--runs: Number of training runs.
--epochs: Number of epochs per run.
Evaluation
Test a pre-trained agent on the microgrid environment:

bash
Copy code
python main.py evaluate --algorithm TCSAC --model-path ./models/tcsac_best_model.h5
Key arguments:

--model-path: Path to the saved model file.
Project Structure
plaintext
Copy code
.
├── controllers/              # RL agent implementations (PPO, TD3, TCSAC)
├── data/                     # Renewable profiles, load, and price data
├── utils/                    # Utility functions for normalization, scaling, and plotting
├── microgrid_env.py          # Microgrid environment simulation
├── models.py                 # Neural network models for RL agents
├── buffer.py                 # Replay buffer implementations
├── constraints_rewards.py    # Constraint functions and reward logic
├── main.py                   # Entry point for training and evaluation
├── setting.py                # Hyperparameters and environment constants
├── README.md                 # Project documentation
Algorithms
Triplet-Critic Soft Actor-Critic (TCSAC)
Objective: Mitigates value estimation bias and accelerates convergence.
Features:
Triple-critic mechanism.
Entropy maximization for exploration.
Prioritized experience replay.
Proximal Policy Optimization (PPO)
Actor-Critic architecture with clipped objective functions.
Twin Delayed Deep Deterministic Policy Gradient (TD3)
Reduces overestimation bias in Q-value approximation.
Environment Details
The environment simulates a CIGRE MV Microgrid, including:

Renewables: Wind turbine and multiple PV generators.
Storage: Two battery systems with charging/discharging constraints.
Loads: Multiple demand nodes with dynamic profiles.
Key Features:

Renewable uncertainty modeled with interval variables.
Real-time decision-making for dispatch optimization.
Multi-objective optimization considering cost, stability, and loss.
Contributing
We welcome contributions to enhance this repository! To contribute:

Fork the repository.
Create a feature branch:
bash
Copy code
git checkout -b feature-name
Commit your changes:
bash
Copy code
git commit -m "Add feature-name"
Push to your branch and create a pull request.
License
This project is licensed under the MIT License. See the LICENSE file for details.
