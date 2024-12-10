'''
func:
- constraints + rewards
'''
import numpy as np
from utils import *
from setting import *
def cal_costgen(power_gen,prev_power_gen): 
    """
    Calculate the generation cost based on power generation.
    """
    cost = (A1 * power_gen ** 2 + A2 * power_gen + A3) + prev_power_gen
    return cost
    # if kwargs:
    #     log_cost_info(profit, kwargs['t'], source='MGO_profit',**kwargs)
def cal_costpow(market_price,power_transfer,prev_cost_pow):
    """
    Calculate the cost based on power transfer.
    """
    cost = market_price * power_transfer + prev_cost_pow
    return cost

def MGO_profit(alpha, curtailed,incentive,prev_mgo_profit):
    """
    Calculate the profit of the Microgrid Operator (MGO) based on curtailment and incentives.
    """
    profit = np.sum((alpha-incentive) * curtailed) + prev_mgo_profit
    return profit


# --- Constraint Functions ---

def power_balance_constraint(P_grid, P_gen, P_solar, P_wind, total_load, curtailed, P_loss): 
    """
    Check the power balance constraint and return a penalty if it is not satisfied.
    total_load = sum(P_demand)
    curtailed = J x 1
    P_loss = P_gen
    """
    
    total_supply = P_grid + P_gen + P_solar + P_wind
    total_demand = total_load - sum(curtailed) + P_loss 
    if not np.isclose(total_supply, total_demand, atol=1e-5): # TODO check atol
        penalty = abs(total_supply - total_demand) ** 2 # TODO check penalty term
        return PENALTY_FACTOR * penalty
    return 0 

def generation_limit_constraint(P_gen):
    """
    Ensure the generation stays within its defined limits.
    """
    if P_gen < PGEN_MIN:
        return PENALTY_FACTOR * (PGEN_MIN - P_gen) ** 2
    elif P_gen > PGEN_MAX:
        return PENALTY_FACTOR * (P_gen - PGEN_MAX) ** 2
    return 0

def ramp_rate_constraint(P_gen, P_gen_prev):
    """
    Ensure the generation ramp rate stays within limits.
    """
    delta = P_gen - P_gen_prev
    if delta > PRAMPUP:
        return PENALTY_FACTOR * (delta - PRAMPUP) ** 2
    elif delta < PRAMPDOWN:
        return PENALTY_FACTOR * (delta - PRAMPDOWN) ** 2
    return 0

def curtailment_limit_constraint(curtailed, P_active_demand, mu1=0, mu2=0.6): 
    """
    Ensure curtailments are within allowable limits.
    """
    if curtailed.shape != P_active_demand.shape:
        raise ValueError("Shapes of 'curtailed' and 'P_demand' must match.")
    min_curtailment = mu1 * P_active_demand
    max_curtailment = mu2 * P_active_demand

    below_min_penalty = np.sum((min_curtailment - curtailed)[curtailed < min_curtailment] ** 2)
    above_max_penalty = np.sum((curtailed - max_curtailment)[curtailed > max_curtailment] ** 2)

    # Total penalty
    penalty = below_min_penalty + above_max_penalty
    return penalty * PENALTY_FACTOR

def daily_curtailment_limit(curtailed, P_active_demand, prev_curtailed, prev_P_active_demand, lambda_=0.4):
    """
    P_active_demand = J x 1
    curtailed = J x 1
    prev_curtailed = J x 1
    prev_P_active_demand = J x 1
    """
    max_curtailment = lambda_ * P_active_demand + prev_P_active_demand
    total_curtailment = curtailed + prev_curtailed
    violations = total_curtailment > max_curtailment

    # Calculate penalty for violations
    penalty = np.sum((total_curtailment[violations] - max_curtailment[violations]) ** 2)
    return PENALTY_FACTOR * penalty

def indivdiual_consumer_benefit(incentive, curtailed, discomforts, prev_benefit, epsilon= EPSILON):
    """
    incentive = 1 x 1
    curtailed = J x 1
    discomforts = J x 1
    prev_benefit = J x 1
    epsilon = J x 1 (but scalar in this case, assuming everyone has same attitude)
    """

    benefit_diff = epsilon * incentive * curtailed - (1 - epsilon) * discomforts

    # Identify violations where the condition is not met
    violations = benefit_diff + prev_benefit < 0

    # Calculate the penalty for violations
    penalty = np.sum((benefit_diff[violations]) ** 2)

    return PENALTY_FACTOR * penalty

def benefit_limit_constraint(incentive, curtailed, discomforts, prev_benefit,epsilon= EPSILON):
    """
    Ensure consumers with higher rankings have greater benefits.
    incentive = 1 x 1
    curtailed = J x 1
    discomforts = J x 1
    prev_curtailed = J x 1
    prev_discomforts = J x 1
    assuming epsilon equal => no need for this 
    """


    # Calculate current and previous benefit values
    current_benefit = (epsilon * incentive * curtailed - (1 - epsilon) * discomforts) + prev_benefit

    # Identify violations where current benefit is less than the previous benefit
    violations = current_benefit < prev_benefit

    # Calculate the penalty for violations
    penalty = np.sum((current_benefit[violations] - prev_benefit[violations]) ** 2)

    return penalty * PENALTY_FACTOR

def incentive_rate_constraint(incentive, market_price, min_market_price,eta=0.4):
    """
    Ensure incentive rate remains within the allowable range.

    """
    min_market_price = min(market_price,min_market_price)
    if incentive > min_market_price:
        return PENALTY_FACTOR * (min_market_price - incentive) ** 2
    elif incentive < min_market_price * eta:
        return PENALTY_FACTOR * (incentive - eta * min_market_price) ** 2
    return 0

def budget_limit_constraint(incentives, curtailed, prev_budget, budget = MB): # TODO set this
    """
    incentives = 1x1
    curtailed = J x 1
    prev_budget = 1x1

    """
    total_cost = sum(incentives * curtailed) + prev_budget
    if total_cost > budget:
        return PENALTY_FACTOR * (total_cost - budget) ** 2
    return 0
