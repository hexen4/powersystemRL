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

1) Add Case 1 to path
2) Change computer variable in main based on computer usage
3) Run main.m; agent with reward > -10 will be saved, alongside the training info, and evaluation based on saved agents





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
