classdef tanh_case3 < rl.env.MATLABEnvironment    
    %1) do i need to include battery economic cost?
    %Qs what happens if isolated bus? -> what to do if gen penalties?
properties
    % Weights for objectives
    w1;
    w2;
    w3;
    w4;
    w5;
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
    IDX_SOC;
    IDX_PGRID_MAX;
    IDX_PGRID_MIN;
    Sbase;
    Vbase;
    Zbase;
    PENALTY_FACTOR;
    time;
    N_OBS;
    EpisodeLogs;
    AllLogs;
    lambda_
    f5;
    f4;
    f3;
    f2;
    f1;
    minprice;
    SOC_min;
    Pbatmax;
    SOC_max;
    start_SOC;
    ref_f1;
    IDX_BROKEN;
    ref_f2;
    ref_f3;
    mingen;

end

properties(Access = protected)
    % Termination Flag
    IsDone = false;     
    training = 1
    %hello
end

methods              
    function this = tanh_case3()
        %% compatability with RL 
        ObservationInfo = rlNumericSpec([47 1], ...
            'LowerLimit', -inf, 'UpperLimit', inf);
        ObservationInfo.Name = 'Microgrid State';
        ActionInfo = rlNumericSpec([10 1], ...
            'LowerLimit', -1 , 'UpperLimit', 1); 
        ActionInfo.Name = 'Microgrid Action';
        % actions are discrete not continuous
        % Call Base Class Constructor
        this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
        this.PENALTY_FACTOR = 10;  
        w1 = 2;
        w2 = 1;
        w3 = 20;
        w4 = 0.5;
        w5 = 1;
        this.w1 = w1/(w1+w2+w3+w4+w5);
        this.w2 = w2/(w1+w2+w3+w4+w5);
        this.w3 = w3/(w1+w2+w3+w4+w5);
        this.w4 = w4/(w1+w2+w3+w4+w5);
        this.w5 = w5/(w1+w2+w3+w4+w5);
        this.H = 24;
        this.EpisodeLogs = {};   
        this.AllLogs = {};       
        this.time = 1;
        this.f5 = 0;
        this.f4 = 0;
        this.f3 = 0;
        this.f2= 0;
        this.f1 = 0;
        this.lambda_ = 0.4;
        this.Vbase = 11; %kV
        this.SOC_min = 20;
        this.SOC_max = 200;
        this.Pbatmax = 40;
        this.start_SOC = 90;
        % decision variables -> charging OR discharging 
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
        this.IDX_PROSUMER_PKW             = 11:15;   % 5 consumers
        this.IDX_PROSUMER_SUM             = 16:20;   % t
        this.IDX_CURTAILED_SUM            = 21:25;   % t
        this.IDX_BENEFIT_SUM              = 26:30;   % t
        this.IDX_BUDGET_SUM              = 31; % t
        this.IDX_MARKET_MINPRICE = 32;
        this.IDX_DISCOMFORTS = 33:37;
        this.IDX_TIME = 38;
        this.IDX_SOC = 39:42;
        this.IDX_PGRID_MAX = 43;
        this.IDX_PGRID_MIN = 44;
        this.IDX_BROKEN = 45:47;
        this.N_OBS = this.IDX_BROKEN(end);
        %% read tables
        this.market_prices = readtable("data/Copy_of_solar_wind_data.csv").price;  
        this.load_percent = readtable("data/Copy_of_solar_wind_data.csv").hourly_load;  
        this.wt_KW_max = 1000*readtable("data/wt_profile.csv").P_wind_max;
        this.wt_KW_min = 1000*readtable("data/wt_profile.csv").P_wind_min; %everything in kW
        this.pv_KW_max = 1000*readtable("data/pv_profile.csv").P_solar_max;
        this.pv_KW_min = 1000*readtable("data/pv_profile.csv").P_solar_min;  
        this.customer_ids = [9,22,14,30,25] ; %SS line included in customers -> +1
        this.State = zeros(this.N_OBS,1);
        this.init_obs = zeros(this.N_OBS,1);
        this.mingen = 10;
        %% cache state t = 1
        this.Sbase = 10; %MVA
        this.Zbase = 121/10;
        this.ref_f1 = 4393.36;
        this.ref_f2 = 650.25;
        this.ref_f3  = 370.74;

        [LD_new,~] = reconfiguration_func(this.load_percent(1), this.pv_KW_max(1),this.wt_KW_max(1),zeros(5,1),zeros(4,1),1);
        LD_new(:,4:5)=LD_new(:,4:5)*this.Zbase;
        [init_BD_max,init_LD_max,CPKW_b4_action,sumload_b4_action]= ieee33(this.load_percent(1), this.pv_KW_max(1),this.wt_KW_max(1),zeros(5,1),zeros(4,1),LD_new);
        
        nbr=size(init_LD_max,1);
        nbus=size(init_BD_max,1);
        [init_Yb_max] = Ybus(init_LD_max,nbr,nbus);
        [init_Vmag_max, init_theta_max, init_Pcalc_max, init_Qcalc_max]= NR_zero_PQVdelta(init_BD_max,init_Yb_max,nbus); 

        [init_BD_min,init_LD_min,~,~]= ieee33(this.load_percent(1), this.pv_KW_min(1),this.wt_KW_min(1),zeros(5,1),zeros(4,1),LD_new);
        [init_Yb_min] = Ybus(init_LD_min,nbr,nbus);
        [init_Vmag_min, init_theta_min, init_Pcalc_min, init_Qcalc_min]= NR_zero_PQVdelta(init_BD_min,init_Yb_min,nbus); 

        this.init_obs(this.IDX_POWER_GEN_MAX) = 1000*this.Sbase*single(init_BD_max(12,7) + init_Pcalc_max(12,1)); %
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
        this.init_obs(this.IDX_CURTAILED_SUM) = zeros(5,1);
        this.init_obs(this.IDX_BENEFIT_SUM) = zeros(5,1);
        this.init_obs(this.IDX_DISCOMFORTS) = zeros(5,1); 
        this.init_obs(this.IDX_BUDGET_SUM) = 0;
        this.init_obs(this.IDX_SOC) = ones(4,1) * this.start_SOC; %kwh
        %others are 0 either due to 0 curtailment or no prev gen power
        this.init_obs(this.IDX_PGRID_MAX) = this.init_obs(this.IDX_TOTAL_LOAD) - this.init_obs(this.IDX_WIND_MAX) - this.init_obs(this.IDX_SOLAR_MAX);
        this.init_obs(this.IDX_PGRID_MIN) = this.init_obs(this.IDX_TOTAL_LOAD) - this.init_obs(this.IDX_WIND_MIN) - this.init_obs(this.IDX_SOLAR_MIN);
        this.State = this.init_obs;          

    end

    function [next_state,resilience_metrics] = update_state(this, State, Action, time)
        %%% whole point of NR is to calculate line losses and set pgen
        prev_curtailed = Action(1:5);
        bat_powers = Action(7:end);
        %% perform reconfiguration
        [LD_nodisaster,~] = reconfiguration_func(this.load_percent(time-1), this.pv_KW_max(time-1), ...
            this.wt_KW_max(time-1),prev_curtailed,bat_powers,0);
        [LD_disaster,rowsToRemove] = reconfiguration_func(this.load_percent(time-1), this.pv_KW_max(time-1), ...
            this.wt_KW_max(time-1),prev_curtailed,bat_powers,time);
        LD_disaster(:,4:5)=LD_disaster(:,4:5)*this.Zbase;
        LD_nodisaster(:,4:5)=LD_nodisaster(:,4:5)*this.Zbase;
        %% max values
        [BD_maxnodisaster,LD_maxnodisaster,CPKW_b4_action,sumload_b4_action]= ieee33(this.load_percent(time-1), ...
            this.pv_KW_max(time-1),this.wt_KW_max(time-1),prev_curtailed,zeros(4,1),LD_nodisaster);
        nbr=size(LD_maxnodisaster,1);
        nbus=size(BD_maxnodisaster,1);
        [Yb_max] = Ybus(LD_maxnodisaster,nbr,nbus);
        [Vmag_max_nodisaster, theta_max, Pcalc_max_nodisaster, Qcalc_max]= NR_zero_PQVdelta(BD_maxnodisaster,Yb_max,nbus); %POWER FLOW
        
        [BD_maxdisaster,LD_maxdisaster,~,~]= ieee33(this.load_percent(time-1), ...
            this.pv_KW_max(time-1),this.wt_KW_max(time-1),prev_curtailed,bat_powers,LD_disaster);
        [Yb_maxdisaster] = Ybus(LD_maxdisaster,nbr,nbus);
        %[Vmag_max_disaster, theta_max_disaster, Pcalc_max_disaster, Qcalc_max_disaster]= NR_zero_PQVdelta(BD_maxdisaster,Yb_maxdisaster,nbus); %POWER FLOW     
        
        [Vmag_max_disaster, theta_max_disaster, Pcalc_max_disaster, Qcalc_max_disaster,P_GRID_MAX]= gen_cap(this,BD_maxdisaster,Yb_maxdisaster,nbus,this.time,this.State(this.IDX_POWER_GEN_MAX));
        %% using min values
        [BD_minnodisaster,LD_minnodisaster,~,~]= ieee33(this.load_percent(time-1), this.pv_KW_min(time-1), ...
            this.wt_KW_min(time-1),prev_curtailed,zeros(4,1),LD_nodisaster);
        [Yb_minnodisaster] = Ybus(LD_minnodisaster,nbr,nbus);
        [Vmag_min_nodisaster, theta_min, Pcalc_min_nodisaster, Qcalc_min]= NR_zero_PQVdelta(BD_minnodisaster,Yb_minnodisaster,nbus); 
        
        [BD_mindisaster,LD_mindisaster,~,~]= ieee33(this.load_percent(time-1), this.pv_KW_min(time-1),...
        this.wt_KW_min(time-1),prev_curtailed,bat_powers,LD_disaster);
        [Yb_mindisaster] = Ybus(LD_mindisaster,nbr,nbus);

        %[Vmag_min_disaster, theta_min_disaster, Pcalc_min_disaster, Qcalc_min_disaster]= NR_zero_PQVdelta(BD_mindisaster,Yb_mindisaster,nbus); %POWER FLOW     
        
        [Vmag_min_disaster, theta_min_disaster, Pcalc_min_disaster, Qcalc_min_disaster,P_GRID_MIN]= gen_cap(this,BD_mindisaster,Yb_mindisaster,nbus,this.time,this.State(this.IDX_POWER_GEN_MIN));
        LEI_max= (BD_maxdisaster(12,7) + Pcalc_max_disaster(12,1)) / (BD_maxnodisaster(12,7) +Pcalc_max_nodisaster(12,1));
        LEI_min = (Pcalc_min_disaster(12,1) + BD_mindisaster(12,7)) / (BD_minnodisaster(12,7) + Pcalc_min_nodisaster(12,1));
        VDI_max = sum(abs(Vmag_max_nodisaster) - abs(Vmag_max_disaster));
        VDI_min = sum(abs(Vmag_min_nodisaster) - abs(Vmag_min_disaster));
        resilience_metrics = [max(LEI_max,0) max(LEI_min,0) VDI_max VDI_min];
        %% Initialize next state (s_{t+1})
        next_state = zeros(this.N_OBS, 1);
        next_state(this.IDX_POWER_GEN_MAX) = 1000*this.Sbase*single(BD_maxdisaster(12,7) + Pcalc_max_disaster(12,1)); %pgen to match prev time
        next_state(this.IDX_POWER_GEN_MIN) = 1000*this.Sbase*single(BD_mindisaster(12,7) +Pcalc_min_disaster(12,1)); 
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
        charge = max(bat_powers, 0);      
        discharge = max(-bat_powers, 0);   
        next_state(this.IDX_SOC) = State(this.IDX_SOC) + (charge * 0.95 - discharge / 0.95);
        next_state(this.IDX_PGRID_MAX) = P_GRID_MAX;
        next_state(this.IDX_PGRID_MIN) = P_GRID_MIN;
        next_state(this.IDX_BROKEN) = rowsToRemove;
    end

    function [Observation,Reward,IsDone] = step(this,Action)
    %% Get action limits
    
    min_incentive = this.State(this.IDX_MARKET_MINPRICE)*0.3;
    max_incentive = this.State(this.IDX_MARKET_MINPRICE); %constraint 8
    bat_min = max(-this.Pbatmax*ones(4,1),(this.SOC_min - this.State(this.IDX_SOC))*0.95);
    bat_max=  min(this.Pbatmax*ones(4,1),(this.SOC_max - this.State(this.IDX_SOC))/0.95);
    %bat_min = -this.Pbatmax*ones(4,1);
    %bat_max=  this.Pbatmax*ones(4,1);
    max_action = [0.6.*this.State(this.IDX_PROSUMER_PKW); max_incentive; bat_max]; %constraint 4
    min_action = [zeros(5,1);min_incentive; bat_min];
    Scaled_Action = this.scale_action(Action,max_action,min_action);

    %% Update state 
    this.time = this.time + 1;
    [Observation_old,resilience_metric] = this.update_state(this.State, Scaled_Action, this.time);
    
    %% Reward + constraints
    [Reward, logStruct,Observation] = ...
    this.calculate_reward(Scaled_Action, Observation_old, this.time, this.State,resilience_metric); %R(s',a, s)
    this.State = Observation;
    this.EpisodeLogs{end+1} = logStruct;        
    %% Check if episode is done  
    IsDone = (this.time >= this.H);
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
        this.AllLogs{end+1} = this.EpisodeLogs;
        this.EpisodeLogs = {};
        this.f1 = 0;
        this.f2 = 0;
        this.f3 = 0;
        this.f4 = 0;
    end

    function [Vmag, theta, Pcalc, Qcalc,P_GRID_MAX]= gen_cap(this,BD,Yb,nbus,time,P_gen_prev)
    [Vmag, theta, Pcalc, Qcalc] = NR_zero_PQVdelta(BD,Yb,nbus);
    P_gen = 1000*10*single(BD(12,7) + Pcalc(12,1));
    amount_from_grid = penalty_checker(this,P_gen,time,P_gen_prev)/(1000*this.Sbase);
    P_GRID_MAX = Pcalc(1)*1000*this.Sbase;
    if amount_from_grid ~= 0
        %BD(1,5) = BD(1,5) + amount_from_grid;
        %BD(1,6)=  BD(1,5)*tan(acos(0.85));
        BD(1,2) = 3;
        BD(12,2) = 1;
        BD(12,5) = P_gen/(1000*this.Sbase) - amount_from_grid;
        [Vmag, theta, Pcalc, Qcalc] = NR_method(BD,Yb,nbus);
        %[Vmag, theta, Pcalc, Qcalc] = NR_zero_PQVdelta(BD,Yb,nbus);
        amount_from_grid = amount_from_grid*1000*this.Sbase;
        P_GRID_MAX = Pcalc(1)*1000*this.Sbase;
    end
    end
  %cannot be less than pgenmin -> add penalty
  %unknown power losses -> pgrid so doesnt rlly matter if i init it
function amount_from_grid = penalty_checker(this, P_gen, time, P_gen_prev)
    [~, kW_mismatch_gen] = generation_limit_constraint(this, P_gen);
    [~, kW_mismatch_ramp] = ramp_rate_constraint(this, P_gen, P_gen_prev, time);

    if kW_mismatch_gen > 0 && kW_mismatch_ramp > 0  
        % Case: Both mismatches are positive
        amount_from_grid = max(kW_mismatch_gen, kW_mismatch_ramp);

    elseif kW_mismatch_gen > 0 && kW_mismatch_ramp < 0  
        % Case: Generation mismatch is positive, but ramp mismatch is negative
        amount_from_grid = kW_mismatch_gen;  % Prioritize gen mismatch

    elseif kW_mismatch_gen < 0 && kW_mismatch_ramp < 0  
        % Case: Both mismatches are negative (may not be meaningful)
        amount_from_grid = kW_mismatch_ramp;

    elseif kW_mismatch_gen < 0 && kW_mismatch_ramp > 0  
        % Case: Generation mismatch is negative, but ramp mismatch is positive
        amount_from_grid = kW_mismatch_ramp;

    elseif abs(kW_mismatch_ramp) > 0 && kW_mismatch_gen == 0 || abs(kW_mismatch_gen) > 0 && kW_mismatch_ramp == 0  
        % Case: One is nonzero, the other is zero
        if kW_mismatch_ramp < 0 || kW_mismatch_ramp > 0
            % If ramp rate mismatch is negative, use it
            amount_from_grid = kW_mismatch_ramp;
        else
            % Otherwise, take the maximum
            amount_from_grid = max(kW_mismatch_gen, kW_mismatch_ramp);
        end

    else  
        % Default case: Both are zero
        amount_from_grid = 0; 
    end
    end

end
%% Optional Methods (set methods' attributes accordingly)
methods     
    function action = scale_action(this,nn_action, max_action, min_action)
        zeroOneAction = (nn_action + 1) / 2;
        action = min_action + (zeroOneAction) .* (max_action - min_action);
        action = min( max(action, min_action), max_action );
    end
    %(optional) Properties validation through set methods
    % function set.State(this,state)
    %     validateattributes(state,{'numeric'},{32},'','State');
    %     this.State = double(state(:));
    %     notifyEnvUpdated(this);
    % end
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
        curtailed = curtailed / 1000;
        profit = sum((alpha - incentive) .* curtailed);
    end

    function penalty = power_balance_constraint(this,P_grid, P_solar, P_wind, total_load, curtailed,bats)
        total_supply = P_grid + P_solar + P_wind;
        total_demand = total_load - sum(curtailed) + sum(bats);
        if abs(total_supply - total_demand) > 1e-5
            penalty = this.PENALTY_FACTOR * abs(total_supply - total_demand);
        else
            penalty = 0;
        end
    end

    function [penalty,unscaled] = generation_limit_constraint(this,P_gen)
        % Ensure the generation stays within its defined limits.
        PGEN_MIN = this.mingen;
        PGEN_MAX = 350;
        if P_gen < PGEN_MIN
            penalty = this.PENALTY_FACTOR * abs(PGEN_MIN - P_gen);
            unscaled = 0;
        elseif P_gen > PGEN_MAX
            penalty = 0;
            unscaled = P_gen-PGEN_MAX ;
        else
            penalty = 0;
            unscaled = 0;
        end
    end

    function [penalty,unscaled] = ramp_rate_constraint(this,P_gen, P_gen_prev, time)
        PRAMPUP = 70;
        PRAMPDOWN = 50;
        if time == 1
            penalty = 0;
            unscaled = 0;
            return;
        end
        delta = P_gen - P_gen_prev;
        if delta > PRAMPUP
            penalty = abs(this.PENALTY_FACTOR * (delta - PRAMPUP));
            unscaled = delta - PRAMPUP;
        elseif delta < -PRAMPDOWN
            penalty = abs(this.PENALTY_FACTOR * (delta + PRAMPDOWN));
            unscaled = delta + PRAMPDOWN;
        else
            penalty = 0;
            unscaled = 0;
        end
    end


    function penalty = daily_curtailment_limit(this, curtailed_sum, P_demand_sum, time)
        violations = curtailed_sum > P_demand_sum;
        
        % Compute the base penalty
        penalty = sum(abs(curtailed_sum(violations) - P_demand_sum(violations)));
    
        % % Apply scaling if there are any violations at time == 24
        % if time == 24 && any(violations)
        %     penalty = penalty * 1e3;
        % end
    end
    function [penalty, benefit_diff] = indivdiual_consumer_benefit(this, incentive, curtailed, discomforts, prev_benefit, time)
        epsilon = [1; 0.9; 0.7; 0.6; 0.4];
        curtailed = curtailed ./ 1000;
        
        % Compute benefit difference
        benefit_diff = (epsilon .* incentive .* curtailed - (1 - epsilon) .* discomforts) + prev_benefit;
        
        % Identify violations and non-violations
        violations = benefit_diff < 0;
        non_violations = ~violations;
    
        % % Compute penalty
        % if time == 24
        %     % Scale violations by 1e4 and add non-violations normally
        %     penalty = sum(1e3 * abs(benefit_diff(violations))) - sum(benefit_diff(non_violations));
        % else
        %     % Regular sum for other times   
        %     penalty = -sum(benefit_diff);
        % end
        penalty = -sum(benefit_diff);
    end   

    function [penalty, total_cost] = budget_limit_constraint(this,incentive, curtailed, prev_budget,time)
        % Check that the total cost does not exceed the specified budget.
        curtailed = curtailed ./ 1000;
        budget = 500;
        total_cost = sum(incentive .* curtailed) + prev_budget;
        if total_cost > budget

           penalty = (abs(total_cost - budget));

        else
            penalty = 0;
        end
    end

    function [penalty] = constraint_penalty(this, action, lower_limit, upper_limit)
        % Initialize penalty array
        penalty = zeros(size(action));
    
        % Apply penalty for exceeding upper limit
        penalty(action > upper_limit) = (action(action > upper_limit) - upper_limit(action > upper_limit)).^2 * this.PENALTY_FACTOR;
    
        % Apply penalty for violating lower limit
        penalty(action < lower_limit) = (lower_limit(action < lower_limit) - action(action < lower_limit)).^2 * this.PENALTY_FACTOR;
    

    end

    function [reward,logStruct, Observation_updated] = calculate_reward(this, scaled_action, next_state, time, state,resilience_metric)

        curtailed = scaled_action(1:5);
        incentive = scaled_action(6);
        bat_powers = scaled_action(7:end);
        %% constants 
        % TODO inlcude bat_powers
        P_grid_max = state(this.IDX_PGRID_MAX);
        P_grid_min = state(this.IDX_PGRID_MIN);
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
        %balance_penalty_max = this.power_balance_constraint(P_grid_max, state(this.IDX_SOLAR_MAX), state(this.IDX_WIND_MAX), state(this.IDX_TOTAL_LOAD), curtailed,bat_powers);
        %balance_penalty_min = this.power_balance_constraint(P_grid_min, state(this.IDX_SOLAR_MIN), state(this.IDX_WIND_MIN), state(this.IDX_TOTAL_LOAD), curtailed,bat_powers);
        balance_penalty_max = 0;
        balance_penalty_min =0 ;
        balance_penalty = (balance_penalty_max + balance_penalty_min) / 2;
        
        % Constraint 2
        [generation_penalty_max,~] = this.generation_limit_constraint(next_state(this.IDX_POWER_GEN_MAX));
        [generation_penalty_min,~] = this.generation_limit_constraint(next_state(this.IDX_POWER_GEN_MIN));
        generation_penalty = (generation_penalty_max + generation_penalty_min) / 2;

        %Constraint 3
        [ramp_penalty_max,~] = this.ramp_rate_constraint(next_state(this.IDX_POWER_GEN_MAX), next_state(this.IDX_PREV_GENPOWER_MAX),time);
        [ramp_penalty_min,~] = this.ramp_rate_constraint(next_state(this.IDX_POWER_GEN_MIN), next_state(this.IDX_PREV_GENPOWER_MIN),time);
        ramp_penalty = (ramp_penalty_max + ramp_penalty_min) / 2;
        if round(ramp_penalty,3) < 1
            ramp_penalty_max = 0;
            ramp_penalty_min = 0;
        end
        if round(generation_penalty,3) < 1
            generation_penalty_max = 0;
            generation_penalty_min = 0;
        end
        %Constraint 5, applied at the end of the episode with
        %intermediate penalties
        daily_curtailment_penalty = this.daily_curtailment_limit(next_state(this.IDX_CURTAILED_SUM), next_state(this.IDX_PROSUMER_SUM),time);
                                                                                    
        %Constraint 6, applied at the end of the episode with
        %intermediate penalties
        [consumer_benefit_penalty, benefit] = this.indivdiual_consumer_benefit(incentive, curtailed, discomforts, state(this.IDX_BENEFIT_SUM),time);
        

        % Constraint 9
        [budget_limit_penalty, budget] = this.budget_limit_constraint(incentive, curtailed, state(this.IDX_BUDGET_SUM),time);

        % Action constraint
        lower_limit = [zeros(5,1);0.3*state(this.IDX_MARKET_MINPRICE);ones(4,1)*this.SOC_min];
        upper_limit = [0.6.*state(this.IDX_PROSUMER_PKW);state(this.IDX_MARKET_MINPRICE);ones(4,1)*this.SOC_max];
        [action_penalty] = this.constraint_penalty([curtailed;incentive;state(this.IDX_SOC)], lower_limit,upper_limit);
        
        %% resilience metric
        F5_max = -(resilience_metric(1) + resilience_metric(3));
        F5_min = -(resilience_metric(2) + resilience_metric(4));
        F5 = (F5_max + F5_min) / 2;
        %% interval optimisation
        consumer_benefit_limit = max(0,time*consumer_benefit_penalty); %clip at -50    to max. MGO_PROFIT without setting icn
        penalties_max = time * (balance_penalty_max  + daily_curtailment_penalty...
            + budget_limit_penalty + generation_penalty_max + ramp_penalty_max )+ consumer_benefit_limit + sum(action_penalty) ;
        penalties_min = time* (balance_penalty_min  + daily_curtailment_penalty...
            + budget_limit_penalty + generation_penalty_min + ramp_penalty_min)  + consumer_benefit_limit + sum(action_penalty);
        penalties = (penalties_max + penalties_min) / 2;
        
        %Compute total reward
        generation_cost = generation_cost / this.ref_f2;
        power_transfer_cost = power_transfer_cost / this.ref_f1;
        mgo_profit = mgo_profit / this.ref_f3;
        reward = -this.w1 *(generation_cost) - this.w2*(power_transfer_cost) + this.w3 * (mgo_profit)...
        - this.w4 * penalties + this.w5 * F5;
        % if time == 24
        %     if penalties < 100 & mgo_profit > 250
        %         reward = reward + 1000*(mgo_profit - 250);
        %     elseif penalties > 1e3 & mgo_profit < 200
        %         reward = reward - 1000*(200 - mgo_profit);
        %     end
        % end
        %reward = reward / 10^6; %scale between [-1,1]
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
            this.f5 = this.f5 + F5;
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
            'consumer_benefit_penalty', consumer_benefit_limit, ...
            'Benefit', benefit, ...
            'budget_limit_penalty', budget_limit_penalty, ...
            'Budget', budget, ...   
            'reward', reward, ...
            'f1', this.f1, ...
            'f2', this.f2, ...
            'f3', this.f3, ...
            'f4', this.f4, ...
            'f5', this.f5, ...
            'VDI_avg', (resilience_metric(3) + resilience_metric(4))/2, ...
            'LEI_avg', (resilience_metric(1) + resilience_metric(2))/2, ...
            'action', scaled_action, ...
            'action_penalty', action_penalty ...
        );
        else
            logStruct = 0;
        end
    end
end

end
