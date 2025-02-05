classdef environment < rl.env.MATLABEnvironment    
    properties
        % Weights for objectives
        w1;
        w2;
        H;
        market_prices; %$/MWh
        load_percent; 
        State;
        wt_KW_max;
        pv_KW_max;
        wt_KW_min;
        pv_KW_min;
        customer_ids;
        init_obs;
        IDX_POWER_GEN_MAX; %smaller as max values used
        IDX_POWER_GEN_MIN;
        IDX_MARKET_PRICE;
        IDX_SOLAR_MAX;
        IDX_SOLAR_MIN;
        IDX_WIND_MAX;
        IDX_WIND_MIN;
        IDX_TOTAL_LOAD;
        IDX_PREV_GENPOWER_MAX;
        IDX_PREV_GENPOWER_MIN;
        IDX_PROSUMER_PKW;
        IDX_PREV_CURTAILED;
        IDX_PREV_ACTIVE_PKW;
        IDX_PREV_ACTIVE_BENEFIT;
        IDX_PREV_BUDGET;
        Sbase;
        PENALTY_FACTOR;
        time;
        N_OBS;
        EpisodeLogs;
        AllLogs;
    end
    
    properties(Access = protected)
        % Termination Flag
        IsDone = false;        
    end

    methods              
        function this = environment()
            %% compatability with RL 
            ObservationInfo = rlNumericSpec([31 1], ...
                'LowerLimit', -inf, 'UpperLimit', inf);
            ObservationInfo.Name = 'Microgrid State';
            ActionInfo = rlNumericSpec([6 1], ...
                'LowerLimit', 0 , 'UpperLimit', 1); 
            ActionInfo.Name = 'Microgrid Action';
            % Call Base Class Constructor
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
            this.PENALTY_FACTOR = 1e6;
            this.w1 = 0.5;
            this.w2 = 0.5;
            this.H = 24;
            this.EpisodeLogs = {};   
            this.AllLogs = {};       
            this.time = 1;
            %% indices of state
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
            this.IDX_PREV_CURTAILED           = 16:20;   % 5 consumers
            this.IDX_PREV_ACTIVE_PKW          = 21:25;   % 5 consumers
            this.IDX_PREV_ACTIVE_BENEFIT      = 26:30;   % 5 consumers
            this.IDX_PREV_BUDGET              = 31;
            this.N_OBS = this.IDX_PREV_BUDGET;
            %% read tables
            this.market_prices = readtable("data/solar_wind_data.csv").price;  
            this.load_percent = readtable("data/solar_wind_data.csv").hourly_load;  
            this.wt_KW_max = 1000*readtable("data/wt_profile.csv").P_wind_max;
            this.wt_KW_min = 1000*readtable("data/wt_profile.csv").P_wind_min; %everything in kW
            this.pv_KW_max = 1000*readtable("data/pv_profile.csv").P_solar_max;
            this.pv_KW_min = 1000*readtable("data/pv_profile.csv").P_solar_min;  
            this.customer_ids = [9,22,14,30,25] + 1; %SS line included in customers -> +1
            this.State = zeros(this.N_OBS,1);
            this.init_obs = zeros(this.N_OBS,1);

            %% cache state t = 1
            this.Sbase = 10; %MVA
            this.init_obs(this.IDX_MARKET_PRICE) = single(this.market_prices(1));
            this.init_obs(this.IDX_SOLAR_MAX)       = single(this.pv_KW_max(1));
            this.init_obs(this.IDX_WIND_MAX)        = single(this.wt_KW_max(1));
            this.init_obs(this.IDX_SOLAR_MIN)       = single(this.pv_KW_min(1));
            this.init_obs(this.IDX_WIND_MIN)        = single(this.wt_KW_min(1)); 
            this.State = this.init_obs;
        end
    
        function next_state = update_state(this, State, Action, time, power_transfer_cost, ....
                mgo_profit, prev_benefit, prev_budget, prev_gen_cost)
            
            curtailed = Action(1:end-1);
            %% max values
            [BD_max,LD_max,TL,CPKW_b4_action,sumload_b4_action]= ieee33(this.load_percent(time), this.pv_KW_max(time),this.wt_KW_max(time),curtailed);
            nbr=size(LD_max,1);
            nbus=size(BD_max,1);
            [Yb_max] = Ybus(LD_max,nbr,nbus);
            [Vmag_max, theta_max, Pcalc_max, Qcalc_max]= NR_zero_PQVdelta(BD_max,Yb_max,nbus); %POWER FLOW
            %% using min values
            [BD_min,LD_min,TL,~,~]= ieee33(this.load_percent(time), this.pv_KW_min(time),this.wt_KW_min(time),curtailed);
            [Yb_min] = Ybus(LD_min,nbr,nbus);
            [Vmag_min, theta_min, Pcalc_min, Qcalc_min]= NR_zero_PQVdelta(BD_min,Yb_min,nbus); %POWER FLOW
            
            %% Initialize next state (s_{t+1})
            next_state = zeros(this.N_OBS, 1);
            next_state(this.IDX_POWER_GEN_MAX) = 1000*this.Sbase*single(Pcalc_max(12,1));
            next_state(this.IDX_POWER_GEN_MIN) = 1000*this.Sbase*single(Pcalc_min(12,1));
            next_state(this.IDX_MARKET_PRICE) = this.market_prices(time);
            next_state(this.IDX_SOLAR_MAX) = single(this.pv_KW_max(time));
            next_state(this.IDX_SOLAR_MIN) = single(this.pv_KW_min(time));
            next_state(this.IDX_WIND_MAX) = single(this.wt_KW_max(time));
            next_state(this.IDX_WIND_MIN) = single(this.wt_KW_min(time));
            next_state(this.IDX_TOTAL_LOAD) = single(sumload_b4_action); %before curtailment
            next_state(this.IDX_PREV_GENPOWER_MAX) = State(this.IDX_POWER_GEN_MAX);
            next_state(this.IDX_PREV_GENPOWER_MIN) = State(this.IDX_POWER_GEN_MIN);
            next_state(this.IDX_PROSUMER_PKW) = single(CPKW_b4_action); %before curtailment
            next_state(this.IDX_PREV_CURTAILED) = single(curtailed); %for constraint
            next_state(this.IDX_PREV_ACTIVE_PKW) = State(this.IDX_PROSUMER_PKW);
            next_state(this.IDX_PREV_ACTIVE_BENEFIT) = single(prev_benefit);
            next_state(this.IDX_PREV_BUDGET) = single(prev_budget);

        end

        function [Observation,Reward,IsDone] = step(this,Action)
        %% run pf for t = 1 with action
        if this.time == 1
            [BD_max,LD_max,TL,CPKW_b4_action,sumload_b4_action]= ieee33(this.load_percent(this.time), this.pv_KW_max(this.time),this.wt_KW_max(this.time),Action(1:end-1));
            nbr=size(LD_max,1);
            nbus=size(BD_max,1);
            [Yb_max] = Ybus(LD_max,nbr,nbus);
            [Vmag_max, theta_max, Pcalc_max, Qcalc_max]= NR_zero_PQVdelta(BD_max,Yb_max,nbus); %POWER FLOW
            [BD_min,LD_min,TL,~,~]= ieee33(this.load_percent(this.time), this.pv_KW_min(this.time),this.wt_KW_min(this.time),Action(1:end-1));
            [Yb_min] = Ybus(LD_min,nbr,nbus);
            [Vmag_min, theta_min, Pcalc_min, Qcalc_min]= NR_zero_PQVdelta(BD_min,Yb_min,nbus); %POWER FLOW
            
            this.State(this.IDX_POWER_GEN_MAX) = 1000*this.Sbase*single(Pcalc_max(12,1));
            this.State(this.IDX_POWER_GEN_MIN) = 1000*this.Sbase*single(Pcalc_min(12,1));
            this.State(this.IDX_TOTAL_LOAD)      = single(sumload_b4_action);
            this.State(this.IDX_PROSUMER_PKW)      = single(CPKW_b4_action);
        end
        %% Get action limits
        max_incentive = 0.4*this.market_prices(this.time); %constraint 8
        max_action = [0.6.*this.State(this.IDX_PROSUMER_PKW); max_incentive]; %constraint 4
        Action = this.scale_action(Action,max_action);
        %% Reward + constraints
        [Reward, power_transfer_cost, mgo_profit, prev_benefit, prev_budget, generation_cost,logStruct] = ...
        this.calculate_reward(Action, this.State, this.time);

        %% Update state 
        this.time = this.time + 1;
        Observation = this.update_state(this.State, Action, this.time, ...
                               power_transfer_cost, mgo_profit, prev_benefit, prev_budget, generation_cost);
        this.State = Observation;
        this.EpisodeLogs{end+1} = logStruct;        
        %% Check if episode is done  
        IsDone = (this.time >= this.H);
        if IsDone
            this.AllLogs{end+1} = this.EpisodeLogs;
            this.EpisodeLogs = {};
            this.reset()
        end

    end
       
        % Reset environment to initial state and output initial observation
        function InitialObservation = reset(this)
            
            InitialObservation = this.init_obs;
            this.State = InitialObservation;
            this.time = 1;
        end
    end
    %% Optional Methods (set methods' attributes accordingly)
    methods     
        function action = scale_action(this,nn_action, max_action)
            min_action = zeros(6,1);
            action = nn_action .* (max_action - min_action);
            action = min( max(action, min_action), max_action );
        end
        % (optional) Properties validation through set methods
        function set.State(this,state)
            validateattributes(state,{'numeric'},{31},'','State');
            this.State = double(state(:));
            notifyEnvUpdated(this);
        end
        function discomforts = calculate_discomforts(this,xjh,pjh)
             CONSUMER_BETA = [1,2,2,3,3];
             discomforts = exp(CONSUMER_BETA' .* (xjh ./ pjh)) - 1;
        end
        function cost = cal_costgen(this,power_gen)
            % Calculate the generation cost based on power generation.
            % The coefficients here are defined locally.
            A1 = 0.0001;  % $/kWh 
            A3 = 14.5216;
            A2 = 0.1032;  % $/kWh
            cost = (A1 * power_gen^2 + A2 * power_gen + A3);
        end
    
        function cost = cal_costpow(this,market_price, power_transfer)
            % Calculate the cost based on power transfer.
            cost = market_price * (power_transfer/1000);
            if cost <= 0
                cost = 0;
            end
        end
    
        function profit = MGO_profit(this,alpha, curtailed, incentive)
            % Calculate the profit of the Microgrid Operator (MGO) based on
            % curtailment and incentives.
            curtailed = curtailed ./ 1000;
            profit = sum((alpha - incentive) .* curtailed);
        end
    
        function penalty = power_balance_constraint(this,P_grid, P_solar, P_wind, total_load, curtailed)
            total_supply = P_grid + P_solar + P_wind;
            total_demand = total_load - sum(curtailed);
            if abs(total_supply - total_demand) > 1e-5
                penalty = this.PENALTY_FACTOR * abs(total_supply - total_demand);
            else
                penalty = 0;
            end
        end
    
        function penalty = generation_limit_constraint(this,P_gen)
            % Ensure the generation stays within its defined limits.
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
    
        function penalty = ramp_rate_constraint(this,P_gen, P_gen_prev, time)
            PRAMPUP = 70;
            PRAMPDOWN = 50;
            if time == 0
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
    
    
        function penalty = daily_curtailment_limit(this,curtailed, P_active_demand, prev_curtailed, prev_P_active_demand, time)
            lambda_ = 0.4;
            max_curtailment = lambda_ .* P_active_demand + prev_P_active_demand;
            total_curtailment = curtailed + prev_curtailed;
            violations = total_curtailment > max_curtailment;
            if time == 23
                penalty = sum(this.PENALTY_FACTOR * abs(total_curtailment(violations) - max_curtailment(violations)));
            else
                penalty = time*sum(abs(total_curtailment(violations) - max_curtailment(violations)));
            end
        end 
        function [penalty, benefit_diff] = indivdiual_consumer_benefit(this,incentive, curtailed, discomforts, prev_benefit,time)
            epsilon = 0.5;
            curtailed = curtailed ./ 1000;
            benefit_diff = (epsilon * incentive .* curtailed - (1 - epsilon) .* discomforts) + prev_benefit;
            violations = benefit_diff < 0;
            if time == 23
                penalty = sum(this.PENALTY_FACTOR * abs(benefit_diff(violations)));
            else 
                penalty = time*sum(abs(benefit_diff(violations)));
            end
            
        end
   
    
        function [penalty, total_cost] = budget_limit_constraint(this,incentive, curtailed, prev_budget,time)
            % Check that the total cost does not exceed the specified budget.
            curtailed = curtailed ./ 1000;
            budget = 500;
            total_cost = sum(incentive .* curtailed) + prev_budget;
            if total_cost > budget
                if time ~= 23
                   penalty = time*abs(total_cost - budget);
                else 
                   penalty = abs(this.PENALTY_FACTOR * (total_cost - budget));
                end
            else
                penalty = 0;
            end
        end
        % Reward function

        function [reward, power_transfer_cost, mgo_profit, prev_benefit, ...
                prev_budget,generation_cost,logStruct] = calculate_reward(this, scaled_action, State, time)

            curtailed = scaled_action(1:end-1);
            incentive = scaled_action(end);
            
            %% constants 
            P_grid_max = State(this.IDX_TOTAL_LOAD) - State(this.IDX_WIND_MAX) - State(this.IDX_SOLAR_MAX) - sum(curtailed);
            P_grid_min = State(this.IDX_TOTAL_LOAD) - State(this.IDX_WIND_MIN) - State(this.IDX_SOLAR_MIN) - sum(curtailed);
            discomforts = this.calculate_discomforts(curtailed, State(this.IDX_PROSUMER_PKW));
            %% interval optimisation
            generation_cost_max = this.cal_costgen(State(this.IDX_POWER_GEN_MAX)); %F2
            generation_cost_min = this.cal_costgen(State(this.IDX_POWER_GEN_MIN)); 
            generation_cost = (generation_cost_min + generation_cost_max) / 2;
            
            power_transfer_cost_max = this.cal_costpow(State(this.IDX_MARKET_PRICE), P_grid_max); %F1
            power_transfer_cost_min = this.cal_costpow(State(this.IDX_MARKET_PRICE), P_grid_min);
            power_transfer_cost = (power_transfer_cost_max + power_transfer_cost_min) / 2;

            % Compute MGO profit
            mgo_profit = this.MGO_profit(State(this.IDX_MARKET_PRICE), curtailed, incentive); %F3
                
            % Compute power balance penalty, constraint 1
            balance_penalty_max = this.power_balance_constraint(P_grid_max, State(this.IDX_SOLAR_MAX), State(this.IDX_WIND_MAX), State(this.IDX_TOTAL_LOAD), curtailed);
            balance_penalty_min = this.power_balance_constraint(P_grid_min, State(this.IDX_SOLAR_MIN), State(this.IDX_WIND_MIN), State(this.IDX_TOTAL_LOAD), curtailed);
            balance_penalty = (balance_penalty_max + balance_penalty_min) / 2;
            
            % Constraint 2
            generation_penalty_max = this.generation_limit_constraint(State(this.IDX_POWER_GEN_MAX));
            generation_penalty_min = this.generation_limit_constraint(State(this.IDX_POWER_GEN_MIN));
            generation_penalty = (generation_penalty_max + generation_penalty_min) / 2;

            %Constraint 3
            ramp_penalty_max = this.ramp_rate_constraint(State(this.IDX_POWER_GEN_MAX), State(this.IDX_PREV_GENPOWER_MAX),time);
            ramp_penalty_min = this.ramp_rate_constraint(State(this.IDX_POWER_GEN_MIN), State(this.IDX_PREV_GENPOWER_MIN),time);
            ramp_penalty = (ramp_penalty_max + ramp_penalty_min) / 2;
        
            %Constraint 5, applied at the end of the episode with
            %intermediate penalties
            daily_curtailment_penalty = this.daily_curtailment_limit(curtailed, State(this.IDX_PROSUMER_PKW), State(this.IDX_PREV_CURTAILED), State(this.IDX_PREV_ACTIVE_PKW),time);
                                                                                        
            %Constraint 6, applied at the end of the episode with
            %intermediate penalties
            [consumer_benefit_penalty, prev_benefit] = this.indivdiual_consumer_benefit(incentive, curtailed, discomforts, State(this.IDX_PREV_ACTIVE_BENEFIT),time);
                                                     
            % Constraint 9
            [budget_limit_penalty, prev_budget] = this.budget_limit_constraint(incentive, curtailed, State(this.IDX_PREV_BUDGET),time);
        
            %Compute total reward
            reward = -this.w1 * (generation_cost + power_transfer_cost) + this.w2 * mgo_profit...
            - balance_penalty  - daily_curtailment_penalty - consumer_benefit_penalty ...
                  - budget_limit_penalty -generation_penalty - ramp_penalty;
            logStruct = struct(...
            'P_grid_max', P_grid_max, ...
            'P_grid_min', P_grid_min, ...
            'discomforts', discomforts, ...
            'generation_cost', generation_cost, ...
            'generation_max', State(this.IDX_POWER_GEN_MAX),...
            'generation_min', State(this.IDX_POWER_GEN_MIN),...
            'generation_penalty', generation_penalty, ...
            'ramp_penalty', ramp_penalty, ...
            'power_transfer_cost', power_transfer_cost, ...
            'mgo_profit', mgo_profit, ...
            'balance_penalty', balance_penalty, ...
            'daily_curtailment_penalty', daily_curtailment_penalty, ...
            'consumer_benefit_penalty', consumer_benefit_penalty, ...
            'Benefit', prev_benefit, ...
            'budget_limit_penalty', budget_limit_penalty, ...
            'Budget', prev_budget, ...
            'reward', reward, ...
            'action', scaled_action ...
);

        end
    end
    
end
