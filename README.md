# **PowerSystemRL**

A MATLAB-based repository for implementing advanced reinforcement learning (RL) algorithms for microgrid energy management with HILF-events.

---


## **Requirements**
- **MATLAB R2023a**
- Install add-ons
Install the following:
MATLAB                                                Version 9.14        (R2023a)
Control System Toolbox                                Version 10.13       (R2023a)
Deep Learning Toolbox                                 Version 14.6        (R2023a)
Global Optimization Toolbox                           Version 4.8.1       (R2023a)
Optimization Toolbox                                  Version 9.5         (R2023a)
Parallel Computing Toolbox                            Version 7.8         (R2023a)
Reinforcement Learning Toolbox                        Version 2.4         (R2023a)
Statistics and Machine Learning Toolbox               Version 12.5        (R2023a)


## **Usage**

### **Training**
1) Set training = 1 in relevant environment(i.e. case2/copyt_of_environment_case3) and run
2) If using SAC / TD3, run actor_generator.mlx and use case2/tanh_case3.
3) Open RL Toolbox and insert relevant hyperparameters / agents

### **Testing**

1) Set training = 0
2) Use agent_evaluator for detailed results



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
