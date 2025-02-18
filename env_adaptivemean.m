classdef env_adaptivemean < rl.env.MATLABEnvironment    
    properties
        % Weights for objectives
        w1;
        w2;
        w3;
        w4;
        H;
        market_prices; % $/MWh
        load_percent; 
        State;
        wt_KW_max;
        pv_KW_max;
        wt_KW_min;
        pv_KW_min;
        customer_ids;
        init_obs;
        IDX_POWER_GEN_MAX; % smaller as max values used
        IDX_POWER_GEN_MIN;
        IDX_MARKET_PRICE;
        IDX_MARKET_MINPRICE;
        IDX_SOLAR_MAX;
        IDX_SOLAR_MIN;
        IDX_WIND_MAX;
        IDX_WIND_MIN;
        IDX_TOTAL_LOAD;
        IDX_PREV_GENPOWER_MAX;
        IDX_PREV_GENPOWER_MIN;
        IDX_PROSUMER_PKW;
        IDX_CURTAILED_SUM ;
        IDX_PROSUMER_SUM;
        IDX_BENEFIT_SUM;
        IDX_BUDGET_SUM; % maybe include discomforts as well. maybe include pgridmax pgridmin as well
        IDX_DISCOMFORTS;
        IDX_TIME;
        Sbase;
        PENALTY_FACTOR;
        time;
        N_OBS;
        EpisodeLogs;
        AllLogs;
        lambda_
        f4;
        f3;
        f2;
        f1;
        minprice;
        
        %% --- NEW PROPERTIES FOR ADAPTIVE REWARD NORMALIZATION ---
        rewardMu;           % Running mean of rewards
        rewardVar;          % Running variance (EMA)
        rewardInitialized;  % Flag for initialization
        alphaReward;        % Smoothing factor (e.g., 0.01)
    end
    
    properties(Access = protected)
        % Termination Flag
        IsDone = false;        
    end

    methods              
        function this = env_adaptivemean()
            %% Compatibility with RL 
            ObservationInfo = rlNumericSpec([38 1], ...
                'LowerLimit', -inf, 'UpperLimit', inf);
            ObservationInfo.Name = 'Microgrid State';
            ActionInfo = rlNumericSpec([6 1], ...
                'LowerLimit', 0 , 'UpperLimit', 100); 
            ActionInfo.Name = 'Microgrid Action';
            % Call Base Class Constructor
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
            this.PENALTY_FACTOR = 1.5;  
            w1 = 1;
            w2 = 1;
            w3 = 10;
            w4 = 1;
            this.w1 = w1/(w1+w2+w3+w4);
            this.w2 = w2/(w1+w2+w3+w4);
            this.w3 = w3/(w1+w2+w3+w4);
            this.w4 = w4/(w1+w2+w3+w4);
            this.H = 24;
            this.EpisodeLogs = {};   
            this.AllLogs = {};       
            this.time = 1;
            this.f4 = 0;
            this.f3 = 0;
            this.f2 = 0;
            this.f1 = 0;
            this.lambda_ = 0.4;
            
            %% --- Initialize adaptive reward normalization variables ---
            this.alphaReward = 0.01;  % Adjust smoothing factor as needed
            this.rewardMu = 0;
            this.rewardVar = 0;
            this.rewardInitialized = false;
            
            %% Battery (commented out)
            % this.SOC_max = 4; % MWh
            % this.SOC_min = 1; % MWh
            % this.SOC_init = 2; % MWh
            % this.battery_efficiency = 95; % percent
            % this.maxcharge = 1; % MW
            
            %% Indices of state
            this.IDX_POWER_GEN_MAX            = 1;
            this.IDX_POWER_GEN_MIN            = 2;
            this.IDX_MARKET_PRICE             = 3;
            this.IDX_SOLAR_MAX                = 4;
            this.IDX_SOLAR_MIN                = 5;
            this.IDX_WIND_MAX                 = 6;
            this.IDX_WIND_MIN                 = 7;
            this.IDX_TOTAL_LOAD               = 8;
            this.IDX_PREV_GENPOWER_MAX        = 9;
            this.IDX_PREV_GENPOWER_MIN        = 10;
            this.IDX_PROSUMER_PKW             = 11:15;   % 5 consumers
            this.IDX_PROSUMER_SUM             = 16:20;
            this.IDX_CURTAILED_SUM            = 21:25;
            this.IDX_BENEFIT_SUM              = 26:30;
            this.IDX_BUDGET_SUM               = 31;
            this.IDX_MARKET_MINPRICE          = 32;
            this.IDX_DISCOMFORTS              = 33:37;
            this.IDX_TIME                     = 38;

            this.N_OBS = this.IDX_TIME;
            %% Read tables
            this.market_prices = readtable("data/Copy_of_solar_wind_data.csv").price;  
            this.load_percent = readtable("data/Copy_of_solar_wind_data.csv").hourly_load;  
            this.wt_KW_max = 1000*readtable("data/wt_profile.csv").P_wind_max;
            this.wt_KW_min = 1000*readtable("data/wt_profile.csv").P_wind_min;
            this.pv_KW_max = 1000*readtable("data/pv_profile.csv").P_solar_max;
            this.pv_KW_min = 1000*readtable("data/pv_profile.csv").P_solar_min;  
            this.customer_ids = [9,22,14,30,25];
            this.State = zeros(this.N_OBS,1);
            this.init_obs = zeros(this.N_OBS,1);

            %% Cache state t = 1
            this.Sbase = 10; % MVA

            [init_BD_max,init_LD_max,TL,CPKW_b4_action,sumload_b4_action] = ieee33(this.load_percent(1), this.pv_KW_max(1), this.wt_KW_max(1), zeros(5,1));
            nbr = size(init_LD_max,1);
            nbus = size(init_BD_max,1);
            [init_Yb_max] = Ybus(init_LD_max, nbr, nbus);
            [init_Vmag_max, init_theta_max, init_Pcalc_max, init_Qcalc_max] = NR_zero_PQVdelta(init_BD_max, init_Yb_max, nbus); 

            [init_BD_min,init_LD_min,TL,~,~] = ieee33(this.load_percent(1), this.pv_KW_min(1), this.wt_KW_min(1), zeros(5,1));
            [init_Yb_min] = Ybus(init_LD_min, nbr, nbus);
            [init_Vmag_min, init_theta_min, init_Pcalc_min, init_Qcalc_min] = NR_zero_PQVdelta(init_BD_min, init_Yb_min, nbus); 

            this.init_obs(this.IDX_POWER_GEN_MAX) = 1000*this.Sbase*single(init_BD_max(12,7) + init_Pcalc_max(12,1));
            this.init_obs(this.IDX_POWER_GEN_MIN) = 1000*this.Sbase*single(init_BD_min(12,7) + init_Pcalc_min(12,1));
            this.init_obs(this.IDX_TOTAL_LOAD)      = single(sumload_b4_action);
            this.init_obs(this.IDX_PROSUMER_PKW)      = single(CPKW_b4_action);
            this.init_obs(this.IDX_MARKET_PRICE)      = single(this.market_prices(1));
            this.init_obs(this.IDX_MARKET_MINPRICE)   = single(this.market_prices(1));
            this.init_obs(this.IDX_SOLAR_MAX)         = single(this.pv_KW_max(1));
            this.init_obs(this.IDX_WIND_MAX)          = single(this.wt_KW_max(1));
            this.init_obs(this.IDX_SOLAR_MIN)         = single(this.pv_KW_min(1));
            this.init_obs(this.IDX_WIND_MIN)          = single(this.wt_KW_min(1)); 
            this.init_obs(this.IDX_PROSUMER_SUM)      = this.lambda_ * single(this.init_obs(this.IDX_PROSUMER_PKW));
            this.init_obs(this.IDX_TIME)            = 1;
            this.init_obs(this.IDX_PREV_GENPOWER_MAX) = 0;
            this.init_obs(this.IDX_PREV_GENPOWER_MIN) = 0;
            this.init_obs(this.IDX_CURTAILED_SUM)     = zeros(5,1);
            this.init_obs(this.IDX_BENEFIT_SUM)       = zeros(5,1);
            this.init_obs(this.IDX_DISCOMFORTS)       = zeros(5,1); 
            this.init_obs(this.IDX_BUDGET_SUM)        = 0;

            this.State = this.init_obs;          
        end
    
        function next_state = update_state(this, State, Action, time)
            %% Use NR to calculate line losses and set pgen
            prev_curtailed = Action(1:end-1);
            %% max values
            [BD_max, LD_max, TL, CPKW_b4_action, sumload_b4_action] = ieee33(this.load_percent(time-1), this.pv_KW_max(time-1), this.wt_KW_max(time-1), prev_curtailed);
            nbr = size(LD_max,1);
            nbus = size(BD_max,1);
            [Yb_max] = Ybus(LD_max, nbr, nbus);
            [Vmag_max, theta_max, Pcalc_max, Qcalc_max] = NR_zero_PQVdelta(BD_max, Yb_max, nbus); % POWER FLOW
            %% using min values
            [BD_min, LD_min, TL, ~, ~] = ieee33(this.load_percent(time-1), this.pv_KW_min(time-1), this.wt_KW_min(time-1), prev_curtailed);
            [Yb_min] = Ybus(LD_min, nbr, nbus);
            [Vmag_min, theta_min, Pcalc_min, Qcalc_min] = NR_zero_PQVdelta(BD_min, Yb_min, nbus); 
            
            %% Initialize next state (s_{t+1})
            next_state = zeros(this.N_OBS, 1);
            next_state(this.IDX_POWER_GEN_MAX) = 1000*this.Sbase*single(BD_max(12,7) + Pcalc_max(12,1));
            next_state(this.IDX_POWER_GEN_MIN) = 1000*this.Sbase*single(BD_min(12,7) + Pcalc_min(12,1)); 
            next_state(this.IDX_MARKET_PRICE)  = this.market_prices(time);
            next_state(this.IDX_MARKET_MINPRICE) = min(this.market_prices(time), this.State(this.IDX_MARKET_MINPRICE));
            next_state(this.IDX_SOLAR_MAX)       = single(this.pv_KW_max(time));
            next_state(this.IDX_SOLAR_MIN)       = single(this.pv_KW_min(time));
            next_state(this.IDX_WIND_MAX)        = single(this.wt_KW_max(time));
            next_state(this.IDX_WIND_MIN)        = single(this.wt_KW_min(time));
            next_state(this.IDX_TOTAL_LOAD)      = single(sumload_b4_action); % before curtailment
            next_state(this.IDX_PREV_GENPOWER_MAX) = State(this.IDX_POWER_GEN_MAX);
            next_state(this.IDX_PREV_GENPOWER_MIN) = State(this.IDX_POWER_GEN_MIN);
            next_state(this.IDX_PROSUMER_PKW)      = single(CPKW_b4_action); % before curtailment
            next_state(this.IDX_PROSUMER_SUM)      = this.lambda_ * next_state(this.IDX_PROSUMER_PKW) + this.State(this.IDX_PROSUMER_SUM);
            next_state(this.IDX_CURTAILED_SUM)     = prev_curtailed + this.State(this.IDX_CURTAILED_SUM); % for constraint
            next_state(this.IDX_TIME)            = time;
        end

        function [Observation, Reward, IsDone] = step(this, Action)
            %% Get action limits
            min_incentive = this.State(this.IDX_MARKET_MINPRICE) * 0.4;
            max_incentive = this.State(this.IDX_MARKET_MINPRICE); % constraint 8
            max_action = [0.6 .* this.State(this.IDX_PROSUMER_PKW); max_incentive]; % constraint 4
            min_action = [zeros(5,1); min_incentive];
            Action = this.scale_action(Action, max_action, min_action);

            %% Update state 
            this.time = this.time + 1;
            Observation_old = this.update_state(this.State, Action, this.time);
            
            %% Reward + constraints
            [Reward, logStruct, Observation] = this.calculate_reward(Action, Observation_old, this.time, this.State);
            this.State = Observation;
            this.EpisodeLogs{end+1} = logStruct;        
            %% Check if episode is done  
            IsDone = (this.time >= this.H);
            if IsDone
                this.AllLogs{end+1} = this.EpisodeLogs;
                this.EpisodeLogs = {};
                this.reset();
            end
        end
       
        % Reset environment to initial state and output initial observation
        function InitialObservation = reset(this)
            InitialObservation = this.init_obs;
            this.State = InitialObservation;
            this.time = 1;
            this.f1 = 0;
            this.f2 = 0;
            this.f3 = 0;
            this.f4 = 0;
            % Optionally, you can reset the reward normalization here if desired:
            %this.rewardInitialized = false;
        end
    end
    
    %% Optional Methods (set methods' attributes accordingly)
    methods     
        function action = scale_action(this, nn_action, max_action, min_action)
            action = min_action + (nn_action/100) .* (max_action - min_action);
            action = min(max(action, min_action), max_action);
        end
        
        function discomforts = calculate_discomforts(this, xjh, pjh)
            CONSUMER_BETA = [1,2,2,3,3];
            discomforts = exp(CONSUMER_BETA' .* (xjh ./ pjh)) - 1;
        end
        
        function cost = cal_costgen(this, power_gen)
            A1 = 0.0001;  % $/kWh 
            A3 = 14.5216;
            A2 = 0.1032;  % $/kWh   
            cost = (A1 * power_gen^2 + A2 * power_gen + A3);
        end
    
        function cost = cal_costpow(this, market_price, power_transfer)
            cost = market_price * (power_transfer/1000);
            if cost <= 0
                cost = 0;
            end
        end
    
        function profit = MGO_profit(this, alpha, curtailed, incentive)
            curtailed = curtailed / 1000;
            profit = sum((alpha - incentive) .* curtailed);
        end
    
        function penalty = power_balance_constraint(this, P_grid, P_solar, P_wind, total_load, curtailed)
            total_supply = P_grid + P_solar + P_wind;
            total_demand = total_load - sum(curtailed);
            if abs(total_supply - total_demand) > 1e-5
                penalty = this.PENALTY_FACTOR * abs(total_supply - total_demand);
            else
                penalty = 0;
            end
        end
    
        function penalty = generation_limit_constraint(this, P_gen)
            PGEN_MIN = 35;
            PGEN_MAX = 300;
            if P_gen < PGEN_MIN
                penalty = this.PENALTY_FACTOR * abs(PGEN_MIN - P_gen);
            elseif P_gen > PGEN_MAX
                penalty = this.PENALTY_FACTOR * abs(P_gen - PGEN_MAX);
            else
                penalty = 0;
            end
        end
    
        function penalty = ramp_rate_constraint(this, P_gen, P_gen_prev, time)
            PRAMPUP = 70;
            PRAMPDOWN = 50;
            if time == 1
                penalty = 0;
                return;
            end
            delta = P_gen - P_gen_prev;
            if delta > PRAMPUP
                penalty = abs(this.PENALTY_FACTOR * (delta - PRAMPUP));
            elseif delta < -PRAMPDOWN
                penalty = abs(this.PENALTY_FACTOR * (delta + PRAMPDOWN));
            else
                penalty = 0;
            end
        end
    
        function penalty = daily_curtailment_limit(this, curtailed_sum, P_demand_sum, time)
            violations = curtailed_sum > P_demand_sum;
            penalty = sum(abs(curtailed_sum(violations) - P_demand_sum(violations)));
        end 
        
        function [penalty, benefit_diff] = indivdiual_consumer_benefit(this, incentive, curtailed, discomforts, prev_benefit, time)
            epsilon = [1;0.9;0.7;0.6;0.4];
            curtailed = curtailed ./ 1000;
            benefit_diff = (epsilon .* incentive .* curtailed - (1 - epsilon) .* discomforts) + prev_benefit;
            % Here we reward positive benefit differences by subtracting the sum of all differences:
            penalty = -sum(benefit_diff);
        end
   
        function [penalty, total_cost] = budget_limit_constraint(this, incentive, curtailed, prev_budget, time)
            curtailed = curtailed ./ 1000;
            budget = 500;
            total_cost = sum(incentive .* curtailed) + prev_budget;
            if total_cost > budget
               penalty = abs(total_cost - budget);
            else
                penalty = 0;
            end
        end
        
        % Reward function with Adaptive Normalization
        function [reward, logStruct, Observation_updated] = calculate_reward(this, scaled_action, next_state, time, state)
            curtailed = scaled_action(1:end-1);
            incentive = scaled_action(end);
            
            %% Constants 
            P_grid_max = state(this.IDX_TOTAL_LOAD) - state(this.IDX_WIND_MAX) - state(this.IDX_SOLAR_MAX) - sum(curtailed);
            P_grid_min = state(this.IDX_TOTAL_LOAD) - state(this.IDX_WIND_MIN) - state(this.IDX_SOLAR_MIN) - sum(curtailed);
            discomforts = this.calculate_discomforts(curtailed, state(this.IDX_PROSUMER_PKW));
            
            %% Interval optimisation
            generation_cost_max = this.cal_costgen(next_state(this.IDX_POWER_GEN_MAX)); % F2
            generation_cost_min = this.cal_costgen(next_state(this.IDX_POWER_GEN_MIN)); 
            generation_cost = (generation_cost_min + generation_cost_max) / 2;
            
            power_transfer_cost_max = this.cal_costpow(state(this.IDX_MARKET_PRICE), P_grid_max); % F1
            power_transfer_cost_min = this.cal_costpow(state(this.IDX_MARKET_PRICE), P_grid_min);
            power_transfer_cost = (power_transfer_cost_max + power_transfer_cost_min) / 2;

            % Compute MGO profit (F3)
            mgo_profit = this.MGO_profit(state(this.IDX_MARKET_PRICE), curtailed, incentive); 
                
            % Compute power balance penalty, constraint 1
            balance_penalty_max = this.power_balance_constraint(P_grid_max, state(this.IDX_SOLAR_MAX), state(this.IDX_WIND_MAX), state(this.IDX_TOTAL_LOAD), curtailed);
            balance_penalty_min = this.power_balance_constraint(P_grid_min, state(this.IDX_SOLAR_MIN), state(this.IDX_WIND_MIN), state(this.IDX_TOTAL_LOAD), curtailed);
            balance_penalty = (balance_penalty_max + balance_penalty_min) / 2;
            
            % Constraint 2
            generation_penalty_max = this.generation_limit_constraint(next_state(this.IDX_POWER_GEN_MAX));
            generation_penalty_min = this.generation_limit_constraint(next_state(this.IDX_POWER_GEN_MIN));
            generation_penalty = (generation_penalty_max + generation_penalty_min) / 2;

            % Constraint 3
            ramp_penalty_max = this.ramp_rate_constraint(next_state(this.IDX_POWER_GEN_MAX), next_state(this.IDX_PREV_GENPOWER_MAX), time);
            ramp_penalty_min = this.ramp_rate_constraint(next_state(this.IDX_POWER_GEN_MIN), next_state(this.IDX_PREV_GENPOWER_MIN), time);
            ramp_penalty = (ramp_penalty_max + ramp_penalty_min) / 2;
        
            % Constraint 5 (daily curtailment penalty)
            daily_curtailment_penalty = this.daily_curtailment_limit(next_state(this.IDX_CURTAILED_SUM), next_state(this.IDX_PROSUMER_SUM), time);
                                                                                        
            % Constraint 6 (individual consumer benefit)
            [consumer_benefit_penalty, benefit] = this.indivdiual_consumer_benefit(incentive, curtailed, discomforts, state(this.IDX_BENEFIT_SUM), time);
            
            % Constraint 9 (budget limit)
            [budget_limit_penalty, budget] = this.budget_limit_constraint(incentive, curtailed, state(this.IDX_BUDGET_SUM), time);
            
            %% Interval optimisation: combine penalties (scaled by time^1.5)
            penalties_max = time^1.5 * (balance_penalty_max + daily_curtailment_penalty + consumer_benefit_penalty + budget_limit_penalty + generation_penalty_max + ramp_penalty_max);
            penalties_min = time^1.5 * (balance_penalty_min + daily_curtailment_penalty + consumer_benefit_penalty + budget_limit_penalty + generation_penalty_min + ramp_penalty_min);
            penalties = (penalties_max + penalties_min) / 2;

            % Compute total reward
            reward = -this.w1 * generation_cost - this.w2 * power_transfer_cost + this.w3 * mgo_profit - this.w4 * penalties;
            
            %% Add sums to next_state for logging/updating state
            next_state(this.IDX_BENEFIT_SUM) = single(benefit);
            next_state(this.IDX_BUDGET_SUM) = single(budget);
            next_state(this.IDX_DISCOMFORTS) = single(discomforts);
            Observation_updated = next_state;
            
            % Log cumulative values
            this.f1 = this.f1 + power_transfer_cost;
            this.f2 = this.f2 + generation_cost;
            this.f3 = this.f3 + mgo_profit;
            this.f4 = this.f4 + penalties;
            
            logStruct = struct(...
                'P_grid_max', P_grid_max, ...
                'P_grid_min', P_grid_min, ...
                'discomforts', discomforts, ...
                'generation_cost', generation_cost, ...
                'generation_max', next_state(this.IDX_POWER_GEN_MAX),...
                'generation_min', next_state(this.IDX_POWER_GEN_MIN),...
                'generation_penalty', generation_penalty, ...
                'ramp_penalty', ramp_penalty, ...
                'power_transfer_cost', power_transfer_cost, ...
                'mgo_profit', mgo_profit, ...
                'balance_penalty', balance_penalty, ...
                'daily_curtailment_penalty', daily_curtailment_penalty, ...
                'consumer_benefit_penalty', consumer_benefit_penalty, ...
                'Benefit', benefit, ...
                'budget_limit_penalty', budget_limit_penalty, ...
                'Budget', budget, ...   
                'reward', reward, ...
                'power_transfer_cost_culm', this.f1, ...
                'generator_cost_culm', this.f2, ...
                'mgo_profit_culm', this.f3, ...
                'sum_penalties', this.f4, ...
                'action', scaled_action ...
            );
            
            %% --- Adaptive Reward Normalization ---
            % Update running mean and variance, then normalize reward.
            if ~this.rewardInitialized
                this.rewardMu = reward;
                this.rewardVar = 0;
                this.rewardInitialized = true;
            else
                this.rewardMu = this.alphaReward * reward + (1 - this.alphaReward) * this.rewardMu;
                diff = reward - this.rewardMu;
                this.rewardVar = this.alphaReward * (diff^2) + (1 - this.alphaReward) * this.rewardVar;
            end
            sigma = sqrt(this.rewardVar) + 1e-8;
            normReward = (reward - this.rewardMu) / sigma;
            reward = normReward;        
            % Optionally squash to [-1,1]:
            %normReward = tanh(normReward);
            %reward = normReward;
            
        end
    end
    
end
