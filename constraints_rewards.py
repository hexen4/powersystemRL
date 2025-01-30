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

def cal_costpow(market_price,power_transfer,prev_cost_pow):
    """
    Calculate the cost based on power transfer.
    """
    cost = market_price * power_transfer + prev_cost_pow
    return cost if cost > 0 else 0

def MGO_profit(alpha, curtailed,incentive,prev_mgo_profit):
    """
    Calculate the profit of the Microgrid Operator (MGO) based on curtailment and incentives.
    """
    profit = np.sum((alpha-incentive) * curtailed) + prev_mgo_profit
    return profit


# --- Constraint Functions ---

def power_balance_constraint(P_grid, P_solar, P_wind, total_load, curtailed, P_loss): 
    """
    Check the power balance constraint and return a penalty if it is not satisfied.
    total_load = sum(P_demand)
    curtailed = J x 1
    P_loss = P_gen
    """
    
    total_supply = P_grid + P_solar + P_wind
    total_demand = total_load - sum(curtailed) + P_loss 
    if not np.isclose(total_supply, total_demand, atol=1e-5): # TODO check atol
        penalty = abs(total_supply - total_demand) 
        return PENALTY_FACTOR * penalty
    return 0 

def generation_limit_constraint(P_gen):
    """
    Ensure the generation stays within its defined limits.
    """
    if P_gen < PGEN_MIN:
        return PENALTY_FACTOR * abs(PGEN_MIN - P_gen) 
    elif P_gen > PGEN_MAX:
        return PENALTY_FACTOR * abs(P_gen - PGEN_MAX) 
    return 0

def ramp_rate_constraint(P_gen, P_gen_prev,time):
    """
    Ensure the generation ramp rate stays within limits.
    """
    if time == 0:
        return 0
    delta = P_gen - P_gen_prev
    if delta > PRAMPUP:
        return abs(PENALTY_FACTOR * (delta - PRAMPUP)) 
    elif delta < -PRAMPDOWN:
        return abs(PENALTY_FACTOR * (delta - PRAMPDOWN)) 
    return 0

def curtailment_limit_constraint(curtailed, P_active_demand, mu1=0, mu2=0.6): 
    """
    Ensure curtailments are within allowable limits.
    """

    min_curtailment = mu1 * P_active_demand
    max_curtailment = mu2 * P_active_demand

    below_min_penalty = abs(min_curtailment[curtailed < min_curtailment] - curtailed[curtailed < min_curtailment])*PENALTY_FACTOR
    above_max_penalty = abs(curtailed[curtailed > max_curtailment] - max_curtailment[curtailed > max_curtailment])*PENALTY_FACTOR

    # Total penalty
    penalty = np.sum(below_min_penalty) + np.sum(above_max_penalty)
    return penalty

def daily_curtailment_limit(curtailed, P_active_demand, prev_curtailed, prev_P_active_demand, lambda_=lambda_):
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
    penalty = np.sum(abs(total_curtailment[violations] - max_curtailment[violations])*PENALTY_FACTOR)
    return penalty

def indivdiual_consumer_benefit(incentive, curtailed, discomforts, prev_benefit, epsilon= EPSILON):
    """
    incentive = 1 x 1
    curtailed = J x 1
    discomforts = J x 1
    prev_benefit = J x 1
    epsilon = J x 1 (but scalar in this case, assuming everyone has same attitude)
    """

    benefit_diff = (epsilon * incentive * curtailed - (1 - epsilon) * discomforts) + prev_benefit

    # Identify violations where the condition is not met
    violations = benefit_diff  < 0

    # Calculate the penalty for violations
    penalty = np.sum(abs(benefit_diff[violations])*PENALTY_FACTOR)

    return penalty,benefit_diff

# def benefit_limit_constraint(incentive, curtailed, discomforts, prev_benefit,epsilon= EPSILON):
#     """
#     Ensure consumers with higher rankings have greater benefits.
#     incentive = 1 x 1
#     curtailed = J x 1
#     discomforts = J x 1
#     prev_curtailed = J x 1
#     prev_discomforts = J x 1
#     assuming epsilon equal => no need for this 
#     """


#     # Calculate current and previous benefit values
#     current_benefit = (epsilon * incentive * curtailed - (1 - epsilon) * discomforts) + prev_benefit

#     # Identify violations where current benefit is less than the previous benefit
#     violations = current_benefit < prev_benefit

#     # Calculate the penalty for violations
#     penalty = np.sum((current_benefit[violations] - prev_benefit[violations]))

#     return abs(penalty * PENALTY_FACTOR)

def incentive_rate_constraint(incentive, market_price, min_market_price,eta=0.4):
    """
    Ensure incentive rate remains within the allowable range.

    """
    min_market_price = min(market_price,min_market_price)
    if incentive < min_market_price:
        return abs(PENALTY_FACTOR * (min_market_price - incentive)) 
    elif incentive > min_market_price * eta:
        return abs(PENALTY_FACTOR * (incentive - eta * min_market_price))
    return 0

def budget_limit_constraint(incentives, curtailed, prev_budget, budget = MB): # TODO set this
    """
    incentives = 1x1
    curtailed = J x 1
    prev_budget = 1x1

    """
    total_cost = np.sum(incentives * curtailed) + prev_budget
    if total_cost > budget:
        return abs((PENALTY_FACTOR * (total_cost - budget))), total_cost
    return 0, total_cost
