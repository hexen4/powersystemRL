'''
func:
- constraints + rewards
'''
import numpy as np
from utils import *
from setting import *
# --- Cost Calculation Functions ---
# TODO need to keep track of prev values -> cummulative sum for optimization
# TODO curtailed -> numpy array
def cal_costgen(power_gen): # TODO where is powwer_gen coming from?
    """
    Calculate the generation cost based on power generation.
    """
    cost = A1 * power_gen ** 2 + A2 * power_gen + A3
    return cost
    # if kwargs:
    #     log_cost_info(profit, kwargs['t'], source='MGO_profit',**kwargs)
def cal_costpow(market_price, power_transfer):
    """
    Calculate the cost based on power transfer.
    """
    cost = market_price * power_transfer
    return cost

def MGO_profit(alpha, curtailed, incentive):
    """
    Calculate the profit of the Microgrid Operator (MGO) based on curtailment and incentives.
    """
    if not isinstance(curtailed, (np.ndarray)):
        curtailed = np.array(curtailed)
    profit = np.sum((alpha-incentive) * curtailed)
    return profit


# --- Constraint Functions ---

def power_balance_constraint(P_grid, P_gen, P_solar, P_wind, P_demand, curtailed, P_loss): # TODO check p_g = p_loss
    """
    Check the power balance constraint and return a penalty if it is not satisfied.
    """
    if not isinstance(P_demand, np.ndarray):
        P_demand = np.array(P_demand)
    if not isinstance(curtailed, np.ndarray):
        curtailed = np.array(curtailed)
    
    if len(P_demand) != len(curtailed):
        raise ValueError("P_demand and curtailments must have the same length, representing each consumer.")
    
    total_supply = P_grid + P_gen + P_solar + P_wind
    total_demand = sum(P_demand) - sum(curtailed) + P_loss # TODO sum(p_demand) = total load?
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

def curtailment_limit_constraint(curtailed, P_demand, mu1=0, mu2=0.6):
    """
    Ensure curtailments are within allowable limits.
    """
    if curtailed.shape != P_demand.shape:
        raise ValueError("Shapes of 'curtailed' and 'P_demand' must match.")
    if not isinstance(curtailed, np.ndarray):
        curtailed = np.array(curtailed)
    if not isinstance(P_demand, np.ndarray):
        P_demand = np.array(P_demand)
    min_curtailment = mu1 * P_demand
    max_curtailment = mu2 * P_demand

    below_min_penalty = np.sum((min_curtailment - curtailed)[curtailed < min_curtailment] ** 2)
    above_max_penalty = np.sum((curtailed - max_curtailment)[curtailed > max_curtailment] ** 2)

    # Total penalty
    penalty = below_min_penalty + above_max_penalty
    return penalty * PENALTY_FACTOR

def daily_curtailment_limit(curtailed, P_demand, prev_curtailed,prev_P_demand, lambda_=0.4):
    if curtailed.shape != P_demand.shape:
        raise ValueError("Shapes of 'curtailed' and 'P_demand' must match.")
    if not isinstance(curtailed, np.ndarray):
        curtailed = np.array(curtailed)
    if not isinstance(P_demand, np.ndarray):
        P_demand = np.array(P_demand)
    if not isinstance(prev_curtailed, np.ndarray):
        prev_curtailed = np.array(prev_curtailed)
    if not isinstance(prev_P_demand, np.ndarray):
        prev_P_demand = np.array(prev_P_demand)

    max_curtailment = lambda_ * (P_demand + prev_P_demand)
    total_curtailment = curtailed + prev_curtailed
    violations = total_curtailment > max_curtailment

    # Calculate penalty for violations
    penalty = np.sum((total_curtailment[violations] - max_curtailment[violations]) ** 2)
    return PENALTY_FACTOR * penalty

def consumer_incentives_constraint(incentive, curtailed, discomforts, epsilon= EPSILON):
    """
    Ensure each consumer has sufficient benefit to offset discomfort from curtailment.
    """
    if not isinstance(curtailed, np.ndarray):
        curtailed = np.array(curtailed)
    if not isinstance(discomforts, np.ndarray):
        discomforts = np.array(discomforts)
    benefit_diff = epsilon * incentive * curtailed - (1 - epsilon) * discomforts

    # Identify violations where the condition is not met
    violations = benefit_diff < 0

    # Calculate the penalty for violations
    penalty = np.sum((benefit_diff[violations]) ** 2)

    return PENALTY_FACTOR * penalty

def incentives_limit_constraint(incentive, curtailed, discomforts,prev_curtailed, prev_discomforts,epsilon= EPSILON):
    """
    Ensure consumers with higher rankings have greater benefits.
    """
    if not isinstance(curtailed, np.ndarray):
        curtailed = np.array(curtailed)
    if not isinstance(discomforts, np.ndarray):
        discomforts = np.array(discomforts)
    if not isinstance(prev_curtailed, np.ndarray):
        prev_curtailed = np.array(prev_curtailed)
    if not isinstance(prev_discomforts, np.ndarray):
        prev_discomforts = np.array(prev_discomforts)

    # Calculate current and previous benefit values
    current_benefit = epsilon * incentive * curtailed - (1 - epsilon) * discomforts
    prev_benefit = epsilon * incentive * prev_curtailed - (1 - epsilon) * prev_discomforts

    # Identify violations where current benefit is less than the previous benefit
    violations = current_benefit < prev_benefit

    # Calculate the penalty for violations
    penalty = np.sum((current_benefit[violations] - prev_benefit[violations]) ** 2)

    return penalty * PENALTY_FACTOR

def incentive_rate_constraint(incentive, market_price, eta=0.4):
    """
    Ensure incentive rate remains within the allowable range.
    """
    if incentive < market_price * eta:
        return PENALTY_FACTOR * (market_price * eta - incentive) ** 2
    elif incentive > market_price * eta:
        return PENALTY_FACTOR * (incentive - eta * market_price) ** 2
    return 0

def budget_limit_constraint(incentives, curtailed, budget = MB): # TODO set this
    """
    Ensure that the total incentive payment remains within the budget.
    """
    total_cost = sum(incentives * curtailed)
    if total_cost > budget:
        return PENALTY_FACTOR * (total_cost - budget) ** 2
    return 0
