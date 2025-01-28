# **PowerSystemRL**

A Python-based repository for implementing advanced reinforcement learning (RL) algorithms for microgrid energy management. This project leverages the **Triplet-Critic Soft Actor-Critic (TCSAC)** algorithm to manage renewable energy uncertainties, optimize costs, and ensure system stability.

---

## **Features**

### **1. Advanced RL Integration**
- Implements **Triplet-Critic Soft Actor-Critic (TCSAC)**:
  - Reduces value estimation bias with a triple-critic mechanism.
  - Incorporates entropy regularization for enhanced exploration.
- Includes other algorithms such as **PPO** and **TD3**.

### **2. Interval-Based Renewable Energy Modeling**
- Simulates uncertainties in solar and wind power outputs.
- Uses interval variables for more realistic power system behavior.

### **3. Comprehensive Reward Function**
- Balances **economic costs**, **network losses**, and **system stability**.
- Dynamically penalizes renewable energy curtailment and state violations.

### **4. Prioritized Experience Replay**
- Replay buffer prioritizes critical transitions for enhanced learning.
- Focuses on transitions violating constraints for faster convergence.

### **5. Multi-Objective Optimization**
- Manages energy dispatch across **economic**, **stability**, and **operational constraints**.

---

## **Requirements**
- **Python 3.12.5+**
- Install requirements
Install the dependencies using:

pip install -r requirements.txt
---

## **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/hexen4/powersystemRL.git
   cd powersystemRL
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```


## **Usage**

### **Training**
Run the `main.py` script to train an RL agent:
```bash
python -m TCASC_controller()
```


## **Project Structure**

```plaintext
.
├── controllers/             # RL agent implementations (PPO, TD3, TCSAC)
├── data/                    # Renewable profiles, load, and price data
├── utils/                   # Utility functions for normalization, scaling, and plotting
├── microgrid_env.py         # Microgrid environment simulation
├── models.py                # Neural network models for RL agents
├── buffer.py                # Replay buffer implementations
├── constraints_rewards.py   # Constraint functions and reward logic
├── main.py                  # Entry point for training and evaluation
├── setting.py               # Hyperparameters and environment constants
├── README.md                # Project documentation
```

## **Environment Details**
The environment simulates a *IEEE34* with the following components:
1. **Renewables**: 1 WT + 1 PV + 1 CDG
2. **Storage**: 1 future battery storage.
3. **Loads**: Dynamic profiles for 33 customers

### **Key Features**
- **Renewable Uncertainty**: Modeled with interval variables.
- **Real-Time Decision-Making**: Optimized for dispatching resources.
- **Multi-Objective Optimization**: Balances cost, stability, and loss.

---
