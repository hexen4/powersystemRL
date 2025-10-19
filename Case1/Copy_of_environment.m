classdef Copy_of_environment < rl.env.MATLABEnvironment
    properties
        %% Weights for objectives
        w1;
        w2;
        w3;
        w4;
        %% variables
        H;
        market_prices; %$/MWh
        load_resi;
        load_comm;
        load_indu;
        State;
        wt_KW_max;
        f1min;
        f1max;
        f2min;
        f2max;
        f3min;
        f3max;
        pv_KW_max;
        wt_KW_min;
        pv_KW_min;
        time;
        %% constants
        customer_ids_residential;
        customer_ids
        customer_ids_commercial;
        load_percent;
        customer_ids_industrial;
        init_obs;
        n_cust;
        Sbase;
        PENALTY_FACTOR;
        Zbase;
        N_OBS;
        discomforts;
        lambda_;
        %% state init.
        IDX_POWER_GEN_MAX; %smaller as max values used
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
        IDX_BUDGET_SUM; %maybe include discomforts as well. maybe include pgridmax pgridmin as well
        IDX_DISCOMFORTS;
        IDX_TIME;
        EpisodeLogs;
        AllLogs;
        f4;
        f3;
        f2;
        f1;
        minprice;
        reconfiguration;
        training 
    end
    
    properties(Access = protected)
        % Termination Flag
        IsDone = false;  
        
    end

    methods              
        function this = Copy_of_environment(trainingMode)
            %% compatability with RL 
            warning('off', 'MATLAB:table:ModifiedAndSavedVarnames');
            ObservationInfo = rlNumericSpec([173 1], ...
                'LowerLimit', -inf, 'UpperLimit', inf);
            ObservationInfo.Name = 'Microgrid State';
            ActionInfo = rlNumericSpec([33 1], ...
                'LowerLimit', 0 , 'UpperLimit', 1); 
            ActionInfo.Name = 'Microgrid Action';
            % Call Base Class Constructor
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
            if nargin > 0
                this.training = trainingMode;
            else
                this.training = 1;  % default value
            end
            this.reconfiguration = 1;
            this.PENALTY_FACTOR = 1;  
            w1 = 1;
            w2 = 1;
            w3 = 10;
            this.w1 = w1/(w1+w2+w3);
            this.w2 = w2/(w1+w2+w3);
            this.w3 = w3/(w1+w2+w3);
            this.H = 24;
            this.EpisodeLogs = {};   
            this.AllLogs = {};       
            this.f1max=0;
            this.f2min=0;
            this.f2max=0;
            this.f3min=0;
            this.f3max=0;
            this.time = 1;
            this.f4 = 0;
            this.f3 = 0;
            this.f2= 0;
            this.f1 = 0;
            this.lambda_ = 0.4;
            this.n_cust = 32;
            this.Sbase = 10; %MVA
            this.Zbase = 121/10;
            %% indices of state
            this.IDX_POWER_GEN_MAX            = 1; %matching prev line losses
            this.IDX_POWER_GEN_MIN            = 2; %matching prev line losses (i.e. at t = 2, power_gen = line losses at t = 1)
            this.IDX_MARKET_PRICE             = 3;
            this.IDX_SOLAR_MAX                = 4;
            this.IDX_SOLAR_MIN                = 5;
            this.IDX_WIND_MAX                 = 6;
            this.IDX_WIND_MIN                 = 7;
            this.IDX_TOTAL_LOAD               = 8;
            this.IDX_PREV_GENPOWER_MAX        = 9;
            this.IDX_PREV_GENPOWER_MIN        = 10;
            this.IDX_PROSUMER_PKW             = 11:42;   
            this.IDX_PROSUMER_SUM             = 43:74;   % t
            this.IDX_CURTAILED_SUM            = 75:106;   % t
            this.IDX_BENEFIT_SUM              = 107:138;   % t
            this.IDX_BUDGET_SUM              = 139; % t
            this.IDX_MARKET_MINPRICE = 140;
            this.IDX_DISCOMFORTS = 141:172;
            this.IDX_TIME = 173;
            this.N_OBS = this.IDX_TIME;
            %% read tables
            if this.training == 0
                this.market_prices = 0.9*readtable("data/Copy_of_solar_wind_data.csv").price;  
                this.load_percent = 0.9*readtable("data/Copy_of_solar_wind_data.csv").hourly_load;  
                this.load_resi = 0.9*readtable("data/Copy_of_solar_wind_data.csv").residential;  
                this.load_comm = 0.9*readtable("data/Copy_of_solar_wind_data.csv").commercial;  
                this.load_indu = 0.9*readtable("data/Copy_of_solar_wind_data.csv").industrial;  
                this.wt_KW_max = 0.9*1000*readtable("data/wt_profile.csv").P_wind_max;
                this.wt_KW_min = 0.9*1000*readtable("data/wt_profile.csv").P_wind_min; %everything in kW
                this.pv_KW_max = 0.9*1000*readtable("data/pv_profile.csv").P_solar_max;
                this.pv_KW_min = 0.9*1000*readtable("data/pv_profile.csv").P_solar_min;  
            else
                this.market_prices = readtable("data/Copy_of_solar_wind_data.csv").price;  
                this.load_percent = readtable("data/Copy_of_solar_wind_data.csv").hourly_load;  
                this.load_resi = readtable("data/Copy_of_solar_wind_data.csv").residential;  
                this.load_comm = readtable("data/Copy_of_solar_wind_data.csv").commercial;  
                this.load_indu = readtable("data/Copy_of_solar_wind_data.csv").industrial;  
                this.wt_KW_max = 1000*readtable("data/wt_profile.csv").P_wind_max;
                this.wt_KW_min = 1000*readtable("data/wt_profile.csv").P_wind_min; %everything in kW
                this.pv_KW_max = 1000*readtable("data/pv_profile.csv").P_solar_max;
                this.pv_KW_min = 1000*readtable("data/pv_profile.csv").P_solar_min;  
            end
            this.customer_ids = [9,22,14,30,25] ; %SS line included in customers -> +1
            this.customer_ids_residential =[2,3,4,6,11,12,13,15,18,21,22,25,30,31,33];
            this.customer_ids_commercial =[5,10,14,19,20,24,27,29,32];
            this.customer_ids_industrial =[7,8,9,16,17,23,26,28];
            this.discomforts =[repmat(0.33, 1, numel(this.customer_ids_residential)), ...
            repmat(0.66, 1, numel(this.customer_ids_commercial)), ...
            repmat(1.00, 1, numel(this.customer_ids_industrial))];
            this.State = zeros(this.N_OBS,1);
            this.init_obs = zeros(this.N_OBS,1);
    
           
            %% cache state t = 1
            this.Sbase = 10; %MVA
            %% reconfiguration
            if this.reconfiguration == 1
                [LD_new_max,~] = reconfiguration_func(this.load_resi(1), ...
                this.load_comm(1),this.load_indu(1), this.pv_KW_max(1), ...
                this.wt_KW_max(1),zeros(this.n_cust,1),this.training);
                [LD_new_min,~] = reconfiguration_func(this.load_resi(1), ...
                this.load_comm(1),this.load_indu(1), this.pv_KW_min(1), ...
                this.wt_KW_min(1),zeros(this.n_cust,1),this.training);

                LD_new_max(:,4:5)=LD_new_max(:,4:5)*this.Zbase;
                LD_new_min(:,4:5)=LD_new_min(:,4:5)*this.Zbase;

                [init_BD_max,init_LD_max,TL,CPKW_b4_action,sumload_b4_action]= ieee33(this.load_resi(1), ...
                this.load_comm(1),this.load_indu(1),this.pv_KW_max(1),this.wt_KW_max(1),zeros(this.n_cust,1),LD_new_max);
                [init_BD_min,init_LD_min,TL,~,~]= ieee33(this.load_resi(1), ...
                this.load_comm(1),this.load_indu(1), this.pv_KW_min(1),this.wt_KW_min(1),zeros(this.n_cust,1),LD_new_min);
            
            else
                [init_BD_max,init_LD_max,TL,CPKW_b4_action,sumload_b4_action]= ieee33(this.load_resi(1), ...
                    this.load_comm(1),this.load_indu(1),this.pv_KW_max(1),this.wt_KW_max(1),zeros(this.n_cust,1));
                [init_BD_min,init_LD_min,TL,~,~]= ieee33(this.load_resi(1), ...
                this.load_comm(1),this.load_indu(1), this.pv_KW_min(1),this.wt_KW_min(1),zeros(this.n_cust,1));
            end
            %% power flow
            nbr=size(init_LD_max,1);
            nbus=size(init_BD_max,1);
            [init_Yb_max] = Ybus(init_LD_max,nbr,nbus);
            [init_Vmag_max, init_theta_max, init_Pcalc_max, init_Qcalc_max]= NR_zero_PQVdelta(init_BD_max,init_Yb_max,nbus); 
            [init_Yb_min] = Ybus(init_LD_min,nbr,nbus);
            [init_Vmag_min, init_theta_min, init_Pcalc_min, init_Qcalc_min]= NR_zero_PQVdelta(init_BD_min,init_Yb_min,nbus); 
            
            %% save into initial state
            this.init_obs(this.IDX_POWER_GEN_MAX) = 1000*this.Sbase*single(init_BD_max(12,7) + init_Pcalc_max(12,1)); 
            this.init_obs(this.IDX_POWER_GEN_MIN) = 1000*this.Sbase*single(init_BD_min(12,7) + init_Pcalc_min(12,1)); 
            this.init_obs(this.IDX_TOTAL_LOAD)      = single(sumload_b4_action);
            this.init_obs(this.IDX_PROSUMER_PKW)      = single(CPKW_b4_action);
            this.init_obs(this.IDX_MARKET_PRICE) = single(this.market_prices(1));
            this.init_obs(this.IDX_MARKET_MINPRICE) = single(this.market_prices(1));
            this.init_obs(this.IDX_SOLAR_MAX)       = single(this.pv_KW_max(1));
            this.init_obs(this.IDX_WIND_MAX)        = single(this.wt_KW_max(1));
            this.init_obs(this.IDX_SOLAR_MIN)       = single(this.pv_KW_min(1));
            this.init_obs(this.IDX_WIND_MIN)        = single(this.wt_KW_min(1)); 
            this.init_obs(this.IDX_PROSUMER_SUM) = this.lambda_ * single(this.init_obs(this.IDX_PROSUMER_PKW));
            this.init_obs(this.IDX_TIME) = 1;
            this.init_obs(this.IDX_PREV_GENPOWER_MAX) = 0;
            this.init_obs(this.IDX_PREV_GENPOWER_MIN) = 0;
            this.init_obs(this.IDX_CURTAILED_SUM) = zeros(this.n_cust,1);
            this.init_obs(this.IDX_BENEFIT_SUM) = zeros(this.n_cust,1);
            this.init_obs(this.IDX_DISCOMFORTS) = zeros(this.n_cust,1); 
            this.init_obs(this.IDX_BUDGET_SUM) = 0;
            %others are 0 either due to 0 curtailment or no prev gen power
            this.State = this.init_obs;          
        end
    
        function next_state = update_state(this, State, Action, time)
            %%% whole point of NR is to calculate line losses and set pgen
            prev_curtailed = Action(1:end-1);
            if this.reconfiguration == 1
                [LD_new_max,~] = reconfiguration_func(this.load_resi(time-1), ...
                this.load_comm(time-1),this.load_indu(time-1), this.pv_KW_max(time-1),this.wt_KW_max(time-1),prev_curtailed,this.training);
                [LD_new_min,~] = reconfiguration_func(this.load_resi(time-1), ...
                this.load_comm(time-1),this.load_indu(time-1), this.pv_KW_min(time-1),this.wt_KW_min(time-1),prev_curtailed,this.training);


                LD_new_max(:,4:5)=LD_new_max(:,4:5)*this.Zbase;
                LD_new_min(:,4:5)=LD_new_min(:,4:5)*this.Zbase;

                [BD_max,LD_max,TL,CPKW_b4_action,sumload_b4_action]= ieee33(this.load_resi(time-1), ...
                this.load_comm(time-1),this.load_indu(time-1), this.pv_KW_max(time-1),this.wt_KW_max(time-1),prev_curtailed,LD_new_max);
                [BD_min,LD_min,TL,~,~]= ieee33(this.load_resi(time-1), ...
                this.load_comm(time-1),this.load_indu(time-1), this.pv_KW_min(time-1),this.wt_KW_min(time-1),prev_curtailed,LD_new_min);
            
            else
                [BD_max,LD_max,TL,CPKW_b4_action,sumload_b4_action]= ieee33(this.load_resi(time-1), ...
                this.load_comm(time-1),this.load_indu(time-1), this.pv_KW_max(time-1),this.wt_KW_max(time-1),prev_curtailed);
                [BD_min,LD_min,TL,~,~]= ieee33(this.load_resi(time-1), ...
                this.load_comm(time-1),this.load_indu(time-1), this.pv_KW_min(time-1),this.wt_KW_min(time-1),prev_curtailed);
            end
            
            %% power flow
            nbr=size(LD_max,1);
            nbus=size(BD_max,1);
            [Yb_max] = Ybus(LD_max,nbr,nbus);
            [Vmag_max, theta_max, Pcalc_max, Qcalc_max]= NR_zero_PQVdelta(BD_max,Yb_max,nbus); %POWER FLOW
            [Yb_min] = Ybus(LD_min,nbr,nbus);
            [Vmag_min, theta_min, Pcalc_min, Qcalc_min]= NR_zero_PQVdelta(BD_min,Yb_min,nbus); 
            % if any(abs(1-Vmag_min)> 0.05) || any(abs(1-Vmag_max)> 0.05)
            %     display(time)
            % end
            %% Initialize next state (s_{t+1})
            next_state = zeros(this.N_OBS, 1);
            next_state(this.IDX_POWER_GEN_MAX) = 1000*this.Sbase*single(BD_max(12,7) + Pcalc_max(12,1)); %pgen to match prev time
            next_state(this.IDX_POWER_GEN_MIN) = 1000*this.Sbase*single(BD_min(12,7) +Pcalc_min(12,1)); 
            next_state(this.IDX_MARKET_PRICE) = this.market_prices(time);
            next_state(this.IDX_MARKET_MINPRICE) = min(this.market_prices(time),this.State(this.IDX_MARKET_MINPRICE));
            next_state(this.IDX_SOLAR_MAX) = single(this.pv_KW_max(time));
            next_state(this.IDX_SOLAR_MIN) = single(this.pv_KW_min(time));
            next_state(this.IDX_WIND_MAX) = single(this.wt_KW_max(time));
            next_state(this.IDX_WIND_MIN) = single(this.wt_KW_min(time));
            next_state(this.IDX_TOTAL_LOAD) = single(sumload_b4_action); %before curtailment
            next_state(this.IDX_PREV_GENPOWER_MAX) = State(this.IDX_POWER_GEN_MAX);
            next_state(this.IDX_PREV_GENPOWER_MIN) = State(this.IDX_POWER_GEN_MIN);
            next_state(this.IDX_PROSUMER_PKW) = single(CPKW_b4_action); %before curtailment
            next_state(this.IDX_PROSUMER_SUM) = this.lambda_ * next_state(this.IDX_PROSUMER_PKW) + this.State(this.IDX_PROSUMER_SUM);
            next_state(this.IDX_CURTAILED_SUM) = prev_curtailed + this.State(this.IDX_CURTAILED_SUM); %for constraint
            next_state(this.IDX_TIME) = time;

        end

        function [Observation,Reward,IsDone] = step(this,Action)
        %% Get action limits

        min_incentive = this.State(this.IDX_MARKET_MINPRICE)*0.3;
        max_incentive = this.State(this.IDX_MARKET_MINPRICE); %constraint 8
        max_action = [0.6.*this.State(this.IDX_PROSUMER_PKW); max_incentive]; %constraint 4
        min_action = [zeros(this.n_cust,1);min_incentive];
        Action = this.scale_action(Action,max_action,min_action);

        %% Update state 
        this.time = this.time + 1;
        Observation_old = this.update_state(this.State, Action, this.time);
        
        %% Reward + constraints
        [Reward, logStruct,Observation] = ...
        this.calculate_reward(Action, Observation_old, this.time, this.State); %R(s',a, s)
        this.State = Observation;
              
        %% Check if episode is done  
        IsDone = (this.time >= this.H);
        if this.training == 0
            this.EpisodeLogs{end+1} = logStruct;
        end
        if IsDone
            if this.training == 1
                this.reset();
            end
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
            this.AllLogs{end+1} = this.EpisodeLogs;
            this.EpisodeLogs = {};
        end
    end
    %% Optional Methods (set methods' attributes accordingly)
    methods     
        function action = scale_action(this,nn_action, max_action, min_action)
            action = min_action + (nn_action) .* (max_action - min_action);
            action = min( max(action, min_action), max_action );
        end
        %(optional) Properties validation through set methods
        % function set.State(this,state)
        %     validateattributes(state,{'numeric'},{32},'','State');
        %     this.State = double(state(:));
        %     notifyEnvUpdated(this);
        % end
        function discomforts = calculate_discomforts(this,xjh,pjh)
             discomforts = exp(this.discomforts' .* (xjh ./ pjh)) - 1;
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
            curtailed = curtailed / 1000;
            profit = sum((alpha - incentive) .* curtailed);
        end
    
        function penalty = power_balance_constraint(this,P_grid, P_solar, P_wind, total_load, curtailed)
            total_supply = P_grid + P_solar + P_wind;
            total_demand = total_load - sum(curtailed);
            if abs(total_supply - total_demand) > 1e-5
                penalty = abs(total_supply - total_demand);
            else
                penalty = 0;
            end
        end
    
        function penalty = generation_limit_constraint(this,P_gen)
            % Ensure the generation stays within its defined limits.
            PGEN_MIN = 35;
            PGEN_MAX = 300;
            if P_gen < PGEN_MIN
                penalty = abs(PGEN_MIN - P_gen);
            elseif P_gen > PGEN_MAX
                penalty = abs(P_gen - PGEN_MAX);
            else
                penalty = 0;
            end
        end
    
        function penalty = ramp_rate_constraint(this,P_gen, P_gen_prev, time)
            PRAMPUP = 70;
            PRAMPDOWN = 50;
            if time == 1
                penalty = 0;
                return;
            end
            delta = P_gen - P_gen_prev;
            if delta > PRAMPUP
                penalty = abs((delta - PRAMPUP));
            elseif delta < -PRAMPDOWN
                penalty = abs((delta + PRAMPDOWN));
            else
                penalty = 0;
            end
        end
    
    
        function penalty = daily_curtailment_limit(this,curtailed_sum, P_demand_sum, time)

            violations = curtailed_sum > P_demand_sum;

            %penalty = sum(this.PENALTY_FACTOR^time * (abs(curtailed_sum(violations) - P_demand_sum(violations))));
            penalty = sum((abs(curtailed_sum(violations) - P_demand_sum(violations))));

        end 
        function [penalty, benefit_diff] = indivdiual_consumer_benefit(this,incentive, curtailed, discomforts, prev_benefit,time)
            epsilon = this.discomforts';
            curtailed = curtailed ./ 1000;
            benefit_diff = (epsilon .* incentive .* curtailed - (1 - epsilon) .* discomforts) + prev_benefit;

            violations = benefit_diff < 0;
            non_violations = ~violations;
            
            %penalty = sum(this.PENALTY_FACTOR^time * (-benefit_diff(violations)) + sum(this.PENALTY_FACTOR^time * benefit_diff(non_violations)));
            %penalty = sum(abs(benefit_diff(violations)));
            penalty = -benefit_diff; % TODO have a look
            end
   
    
        function [penalty, total_cost] = budget_limit_constraint(this,incentive, curtailed, prev_budget,time)
            % Check that the total cost does not exceed the specified budget.
            curtailed = curtailed ./ 1000;
            budget = 3200;
            total_cost = sum(incentive .* curtailed) + prev_budget;
            if total_cost > budget

               penalty = (abs(total_cost - budget));

            else
                penalty = 0;
            end
        end
        % Reward function

        function [reward,logStruct, Observation_updated] = calculate_reward(this, scaled_action, next_state, time, state)

            curtailed = scaled_action(1:end-1);
            incentive = scaled_action(end);
            
            %% constants 
            P_grid_max = state(this.IDX_TOTAL_LOAD) - state(this.IDX_WIND_MAX) - state(this.IDX_SOLAR_MAX) - sum(curtailed);
            P_grid_min = state(this.IDX_TOTAL_LOAD) - state(this.IDX_WIND_MIN) - state(this.IDX_SOLAR_MIN) - sum(curtailed);
            discomforts = this.calculate_discomforts(curtailed, state(this.IDX_PROSUMER_PKW));
            %% interval optimisation
            generation_cost_max = this.cal_costgen(next_state(this.IDX_POWER_GEN_MAX)); %F2
            generation_cost_min = this.cal_costgen(next_state(this.IDX_POWER_GEN_MIN)); 
            generation_cost = (generation_cost_min + generation_cost_max) / 2;
            
            power_transfer_cost_max = this.cal_costpow(state(this.IDX_MARKET_PRICE), P_grid_max); %F1
            power_transfer_cost_min = this.cal_costpow(state(this.IDX_MARKET_PRICE), P_grid_min);
            power_transfer_cost = (power_transfer_cost_max + power_transfer_cost_min) / 2;

            % Compute MGO profit
            mgo_profit = this.MGO_profit(state(this.IDX_MARKET_PRICE), curtailed, incentive); %F3
                
            % Compute power balance penalty, constraint 1
            balance_penalty_max = this.power_balance_constraint(P_grid_max, state(this.IDX_SOLAR_MAX), state(this.IDX_WIND_MAX), state(this.IDX_TOTAL_LOAD), curtailed);
            balance_penalty_min = this.power_balance_constraint(P_grid_min, state(this.IDX_SOLAR_MIN), state(this.IDX_WIND_MIN), state(this.IDX_TOTAL_LOAD), curtailed);
            balance_penalty = (balance_penalty_max + balance_penalty_min) / 2;
            
            % Constraint 2
            generation_penalty_max = this.generation_limit_constraint(next_state(this.IDX_POWER_GEN_MAX));
            generation_penalty_min = this.generation_limit_constraint(next_state(this.IDX_POWER_GEN_MIN));
            generation_penalty = (generation_penalty_max + generation_penalty_min) / 2;

            %Constraint 3
            ramp_penalty_max = this.ramp_rate_constraint(next_state(this.IDX_POWER_GEN_MAX), next_state(this.IDX_PREV_GENPOWER_MAX),time);
            ramp_penalty_min = this.ramp_rate_constraint(next_state(this.IDX_POWER_GEN_MIN), next_state(this.IDX_PREV_GENPOWER_MIN),time);
            ramp_penalty = (ramp_penalty_max + ramp_penalty_min) / 2;
        
            %Constraint 5, applied at the end of the episode with
            %intermediate penalties
            daily_curtailment_penalty = this.daily_curtailment_limit(next_state(this.IDX_CURTAILED_SUM), next_state(this.IDX_PROSUMER_SUM),time);
                                                                                        
            %Constraint 6, applied at the end of the episode with
            %intermediate penalties
            [consumer_benefit_penalty, benefit] = this.indivdiual_consumer_benefit(incentive, curtailed, discomforts, state(this.IDX_BENEFIT_SUM),time);
            

            % Constraint 9
            [budget_limit_penalty, budget] = this.budget_limit_constraint(incentive, curtailed, state(this.IDX_BUDGET_SUM),time);
            %% interval optimisation
            consumer_benefit_limit = max(0,consumer_benefit_penalty); %clip at -30 for maximum benefit and minimise 
            penalties_max = (time/2) * (balance_penalty_max  + daily_curtailment_penalty...
                + budget_limit_penalty + generation_penalty_max + ramp_penalty_max) + sum(consumer_benefit_limit) ;
            penalties_min = (time/2) * (balance_penalty_min  + daily_curtailment_penalty...
                + budget_limit_penalty + generation_penalty_min + ramp_penalty_min) + sum(consumer_benefit_limit);
            penalties = (penalties_max + penalties_min) / 2;

            %Compute total reward
            reward = -this.w1 *generation_cost - this.w2*power_transfer_cost + this.w3 * mgo_profit...
            -  penalties;
           % reward = reward / 10;
            %% add sums to next_state
           
            next_state(this.IDX_BENEFIT_SUM) = single(benefit);
            next_state(this.IDX_BUDGET_SUM) = single(budget);
            next_state(this.IDX_DISCOMFORTS) = single(discomforts);
            Observation_updated = next_state;
            %log
            if this.training == 0
                this.f1 = this.f1 + power_transfer_cost;
                this.f2 = this.f2 + generation_cost;
                this.f3 = this.f3 + mgo_profit;
                this.f4 = this.f4 + penalties;
                this.f1min = this.f1min +power_transfer_cost_min;
                this.f1max = this.f1max +power_transfer_cost_max;
                this.f2min = this.f2min + generation_cost_min;
                this.f2max = this.f2max + generation_cost_max;
                this.f3min = this.f3min + mgo_profit;
                this.f3max = this.f3max + mgo_profit;
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
                'consumer_benefit_penalty', sum(consumer_benefit_limit), ...
                'Benefit', benefit, ...
                'budget_limit_penalty', budget_limit_penalty, ...
                'Budget', budget, ...   
                'reward', reward, ...
                'power_transfer_cost_culm', this.f1, ...
                'generator_cost_culm', this.f2, ...
                'mgo_profit_culm', this.f3, ...
                'sum_penalties', this.f4, ...
                "penalties", penalties,...
                'f1min', this.f1min, ...
                'f1max',this.f1max, ...
                'f2min',this.f2min, ...
                'f2max',this.f2max, ...
                'f3min',this.f3min, ...
                'f3max',this.f3max, ...
                'action', scaled_action ...
            );
            else
                logStruct = 0;
            end
        end
    end
    
end