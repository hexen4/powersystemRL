classdef Copy_of_environment_case3 < rl.env.MATLABEnvironment   
   %%% most metrics have min or max appended. This comes from using
   %%% minimum / maximum values for RE-based energy. HILF-event applied to
   %%% both.
properties
    %% Weights for objectives
    f1min;
    f1max;
    f2min;
    f2max;
    f3min;
    f3max;
    f5min;
    f5max;
    w1;
    w2;
    w3;
    w4;
    w5;
    %% constants / data
    H; %hours in day
    market_prices; %$/MWh
    load_resi;
    load_comm;
    load_indu;
    State; 
    wt_KW_max; %max. predicted wind power
    pv_KW_max;
    wt_KW_min;
    pv_KW_min;
    customer_ids_residential;
    customer_ids_commercial;
    customer_ids_industrial;
    n_cust;
    Sbase;
    Vbase;
    Zbase;
    PENALTY_FACTOR;
    ref_f1; %from Case I
    ref_f2;
    ref_f3;
    ref_voltage; %reference voltage during best case scenerio (only reconfigured, no HILF)
    %% array initilisation 
    init_obs;
    IDX_POWER_GEN_MAX; %power import from grid is smaller during max case
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
    IDX_BROKEN; %broken branch list
    LEI_MAX; %energy loss (compared to non hilf event)
    LEI_MIN;
    VDI_min; %voltage deviation
    VDI_max;

    %% variables init.
    time;
    N_OBS; %number of observations in state
    EpisodeLogs;
    AllLogs;
    lambda_ %constant from 2021 PH paper
    f5;
    f4;
    f3;
    f2;
    f1;
    minprice;
    discomforts;
    %% battery and HILF constants
    SOC_min;
    Pbatmax; 
    SOC_max;
    start_SOC;
    event_time; %when HILF happens
    training;
end

properties(Access = protected)
    % Termination Flag
    IsDone = false;     
end

methods              
    function this = Copy_of_environment_case3(trainingMode)
        %% compatability with RL 
        warning('off', 'MATLAB:table:ModifiedAndSavedVarnames');
        ObservationInfo = rlNumericSpec([217 1], ...
            'LowerLimit', -inf, 'UpperLimit', inf);
        ObservationInfo.Name = 'Microgrid State';
        ActionInfo = rlNumericSpec([37 1], ...
            'LowerLimit', 0 , 'UpperLimit', 1); %important, action output between 0 and 1
        ActionInfo.Name = 'Microgrid Action';
        this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo); % Call Base Class Constructor
        if nargin > 0
            this.training = trainingMode;
        else
            this.training = 1;  % default value
        end
        this.PENALTY_FACTOR = 1;  
        w1 = 5;
        w2 = 0.5;
        w3 = 20;
        w4 = 1;
        this.w1 = w1/(w1+w2+w3+w4);
        this.n_cust = 32;
        this.w2 = w2/(w1+w2+w3+w4);
        this.w3 = w3/(w1+w2+w3+w4);
        this.w4 = w4/(w1+w2+w3+w4);
        this.H = 24;
        this.EpisodeLogs = {};   
        this.AllLogs = {};       
        this.time = 1;
        this.f5 = 0;
        this.f4 = 0;
        this.f3 = 0;
        this.f2= 0;
        this.f1 = 0;
        this.f1min =0;
        this.VDI_max = 0;
        this.VDI_min = 0;
        this.f1max=0;
        this.f2min=0;
        this.f2max=0;
        this.f3min=0;
        this.f3max=0;
        this.f5min=0;
        this.f5max=0;
        this.lambda_ = 0.4;
        this.Vbase = 11; %kV
        this.SOC_min = 100; %from paper
        this.SOC_max = 600; %from paper
        this.Pbatmax = 90; %from paper
        this.start_SOC = this.SOC_min+(this.SOC_max-this.SOC_min)*0.5; 
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
        this.IDX_PROSUMER_PKW             = 11:42;   % 5 consumers
        this.IDX_PROSUMER_SUM             = 43:74;   % t
        this.IDX_CURTAILED_SUM            = 75:106;   % t
        this.IDX_BENEFIT_SUM              = 107:138;   % t
        this.IDX_BUDGET_SUM              = 139; % t
        this.IDX_MARKET_MINPRICE = 140;
        this.IDX_DISCOMFORTS = 141:172;
        this.IDX_TIME = 173;
        this.IDX_SOC = 174:177;
        this.IDX_PGRID_MAX = 178;
        this.IDX_PGRID_MIN = 179;
        this.IDX_BROKEN = 180:217;
        this.event_time = [6:9 12:15 19:22];
        %this.event_time = [];
        this.N_OBS = this.IDX_BROKEN(end);
        
        %% read tables and variable initialisation
        if this.training == 0
            this.market_prices = 0.9*readtable("data/Copy_of_solar_wind_data.csv").price;  
            this.load_resi = 0.9*readtable("data/Copy_of_solar_wind_data.csv").residential;  
            this.load_comm = 0.9*readtable("data/Copy_of_solar_wind_data.csv").commercial;  
            this.load_indu = 0.9*readtable("data/Copy_of_solar_wind_data.csv").industrial;  
            this.wt_KW_max = 0.9*1000*readtable("data/wt_profile.csv").P_wind_max;
            this.wt_KW_min = 0.9*1000*readtable("data/wt_profile.csv").P_wind_min; %everything in kW
            this.pv_KW_max = 0.9*1000*readtable("data/pv_profile.csv").P_solar_max;
            this.pv_KW_min = 0.9*1000*readtable("data/pv_profile.csv").P_solar_min;  
        else
            this.market_prices = readtable("data/Copy_of_solar_wind_data.csv").price;  
 
            this.load_resi = readtable("data/Copy_of_solar_wind_data.csv").residential;  
            this.load_comm = readtable("data/Copy_of_solar_wind_data.csv").commercial;  
            this.load_indu = readtable("data/Copy_of_solar_wind_data.csv").industrial;  
            this.wt_KW_max = 1000*readtable("data/wt_profile.csv").P_wind_max;
            this.wt_KW_min = 1000*readtable("data/wt_profile.csv").P_wind_min; %everything in kW
            this.pv_KW_max = 1000*readtable("data/pv_profile.csv").P_solar_max;
            this.pv_KW_min = 1000*readtable("data/pv_profile.csv").P_solar_min;  
        end
        this.State = zeros(this.N_OBS,1);
        this.init_obs = zeros(this.N_OBS,1);
        this.Sbase = 10; %MVA
        this.Zbase = 121/10;
        this.ref_f1 = 3600; %% TODO CHANGE
        this.ref_f2 =  520; %% TODO CHANGE
        this.LEI_MAX = 0;
        this.LEI_MIN = 0;
        this.ref_f3  = 500; %% TODO CHANGE
        %this.ref_voltage = load("savedconstants_OLD\Vmag_reconfig.mat").vmag; %% TODO CHANGE
        this.customer_ids_residential =[2,3,4,6,11,12,13,15,18,21,22,25,30,31,33];
        this.customer_ids_commercial =[5,10,14,19,20,24,27,29,32];
        this.customer_ids_industrial =[7,8,9,16,17,23,26,28];
        this.discomforts =[repmat(0.33, 1, numel(this.customer_ids_residential)), ...
        repmat(0.66, 1, numel(this.customer_ids_commercial)), ...
        repmat(1.00, 1, numel(this.customer_ids_industrial))];
        %% cache state t = 1 with a = 0

        %NR with BIBC reconfiguration to find minimum power loss
        [LD_new_max,~] = reconfiguration_func(this.load_resi(1), ...
                this.load_comm(1),this.load_indu(1), this.pv_KW_max(1), ...
                this.wt_KW_max(1),zeros(this.n_cust,1),zeros(4,1),0,this.training);
        [LD_new_min,~] = reconfiguration_func(this.load_resi(1), ...
                this.load_comm(1),this.load_indu(1), this.pv_KW_min(1), ...
                this.wt_KW_min(1),zeros(this.n_cust,1),zeros(4,1),0,this.training);

        LD_new_max(:,4:5)=LD_new_max(:,4:5)*this.Zbase;
        LD_new_min(:,4:5)=LD_new_min(:,4:5)*this.Zbase;
        [init_BD_max,init_LD_max,CPKW_b4_action,sumload_b4_action]= ieee33(this.load_resi(1), ...
                this.load_comm(1),this.load_indu(1),this.pv_KW_max(1),this.wt_KW_max(1),zeros(this.n_cust,1),zeros(4,1),LD_new_max);
        
        nbr=size(init_LD_max,1);
        nbus=size(init_BD_max,1);
        [init_Yb_max] = Ybus(init_LD_max,nbr,nbus);
        %calcualte generator power (losses) PQVD
        [init_Vmag_max, init_theta_max, init_Pcalc_max, init_Qcalc_max]= NR_zero_PQVdelta(init_BD_max,init_Yb_max,nbus); 

        [init_BD_min,init_LD_min,~,~]= ieee33(this.load_resi(1), ...
                this.load_comm(1),this.load_indu(1),this.pv_KW_min(1),this.wt_KW_min(1),zeros(this.n_cust,1),zeros(4,1),LD_new_min);
        [init_Yb_min] = Ybus(init_LD_min,nbr,nbus);
        [init_Vmag_min, init_theta_min, init_Pcalc_min, init_Qcalc_min]= NR_zero_PQVdelta(init_BD_min,init_Yb_min,nbus); 
        %% create state
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
        this.init_obs(this.IDX_CURTAILED_SUM) = zeros(this.n_cust,1);
        this.init_obs(this.IDX_BENEFIT_SUM) = zeros(this.n_cust,1);
        this.init_obs(this.IDX_DISCOMFORTS) = zeros(this.n_cust,1); 
        this.init_obs(this.IDX_BUDGET_SUM) = 0;
        this.init_obs(this.IDX_SOC) = ones(4,1) * this.start_SOC; %kwh
        %others are 0 either due to 0 curtailment or no prev gen power
        this.init_obs(this.IDX_PGRID_MAX) = this.init_obs(this.IDX_TOTAL_LOAD) - this.init_obs(this.IDX_WIND_MAX) - this.init_obs(this.IDX_SOLAR_MAX);
        this.init_obs(this.IDX_PGRID_MIN) = this.init_obs(this.IDX_TOTAL_LOAD) - this.init_obs(this.IDX_WIND_MIN) - this.init_obs(this.IDX_SOLAR_MIN);
        this.State = this.init_obs;          
        
    end

    function [next_state,resilience_metrics,Vmagdata,x] = update_state(this, State, Action, time)
        %%% whole point of NR is to calculate line losses and set pgen
        curtailed = Action(1:32);
        bat_powers = Action(34:end);
        %% perform reconfiguration without distaster
        [LD_nodisaster_max,~] = reconfiguration_func(this.load_resi(time-1), ...
                this.load_comm(time-1),this.load_indu(time-1), this.pv_KW_max(time-1), ...
                this.wt_KW_max(time-1),zeros(this.n_cust,1),zeros(4,1),0,this.training);
        [LD_nodisaster_min,~] = reconfiguration_func(this.load_resi(time-1), ...
                this.load_comm(time-1),this.load_indu(time-1), this.pv_KW_min(time-1), ...
                this.wt_KW_min(time-1),zeros(this.n_cust,1),zeros(4,1),0,this.training);
        
        %% disaster modelling
        if ismember(time, this.event_time) %if else hilf event -> flag != 0 => disaster
        [LD_disaster_min,rowsToRemove_min] = reconfiguration_func(this.load_resi(time-1), ...
                this.load_comm(time-1),this.load_indu(time-1), this.pv_KW_min(time-1), ...
                this.wt_KW_min(time-1),curtailed,bat_powers,time,this.training);
        [LD_disaster_max,rowsToRemove_max,x] = reconfiguration_func(this.load_resi(time-1), ...
                this.load_comm(time-1),this.load_indu(time-1), this.pv_KW_max(time-1), ...
                this.wt_KW_max(time-1),curtailed,bat_powers,time,this.training);
        else
        [LD_disaster_min,rowsToRemove_min] = reconfiguration_func(this.load_resi(time-1), ...
                this.load_comm(time-1),this.load_indu(time-1), this.pv_KW_min(time-1), ...
                this.wt_KW_min(time-1),curtailed,bat_powers,0,this.training);
        [LD_disaster_max,rowsToRemove_max,x] = reconfiguration_func(this.load_resi(time-1), ...
                this.load_comm(time-1),this.load_indu(time-1), this.pv_KW_max(time-1), ...
                this.wt_KW_max(time-1),curtailed,bat_powers,0,this.training);
        end
        LD_disaster_max(:,4:5)=LD_disaster_max(:,4:5)*this.Zbase;
        LD_nodisaster_max(:,4:5)=LD_nodisaster_max(:,4:5)*this.Zbase;
        LD_nodisaster_min(:,4:5)=LD_nodisaster_min(:,4:5)*this.Zbase;
        LD_disaster_min(:,4:5)=LD_disaster_min(:,4:5)*this.Zbase;

        %% max values power flow 
        [BD_maxnodisaster,LD_maxnodisaster,CPKW_b4_action,sumload_b4_action]= ieee33(this.load_resi(time-1), ...
         this.load_comm(time-1),this.load_indu(time-1),this.pv_KW_max(time-1),this.wt_KW_max(time-1), ...
         zeros(this.n_cust,1),zeros(4,1),LD_nodisaster_max);

        nbr=size(LD_maxnodisaster,1);
        nbus=size(BD_maxnodisaster,1);
        [Yb_max] = Ybus(LD_maxnodisaster,nbr,nbus);
        [Vmag_max_nodisaster, theta_max, Pcalc_max_nodisaster, Qcalc_max]= NR_zero_PQVdelta(BD_maxnodisaster,Yb_max,nbus); %POWER FLOW
        
        [BD_maxdisaster,LD_maxdisaster,~,~]= ieee33(this.load_resi(time-1), ...
         this.load_comm(time-1),this.load_indu(time-1), ...
         this.pv_KW_max(time-1),this.wt_KW_max(time-1),curtailed,bat_powers,LD_disaster_max);

        [Yb_maxdisaster] = Ybus(LD_maxdisaster,nbr,nbus);
        [Vmag_max_disaster, theta_max_disaster, Pcalc_max_disaster, Qcalc_max_disaster,P_GRID_MAX]= gen_cap(this,BD_maxdisaster, ...
            Yb_maxdisaster,nbus,this.time,this.State(this.IDX_POWER_GEN_MAX)); %power flow with gen_cap consideration. 
        % extra imported from grid
        %% min values power flow 
        [BD_minnodisaster,LD_minnodisaster,~,~]= ieee33(this.load_resi(time-1), ...
         this.load_comm(time-1),this.load_indu(time-1),this.pv_KW_min(time-1),this.wt_KW_min(time-1), ...
         zeros(this.n_cust,1),zeros(4,1),LD_nodisaster_min);

        [Yb_minnodisaster] = Ybus(LD_minnodisaster,nbr,nbus);
        [Vmag_min_nodisaster, theta_min, Pcalc_min_nodisaster, Qcalc_min]= NR_zero_PQVdelta(BD_minnodisaster,Yb_minnodisaster,nbus);      
        [BD_mindisaster,LD_mindisaster,~,~]= ieee33(this.load_resi(time-1), ...
         this.load_comm(time-1),this.load_indu(time-1), this.pv_KW_min(time-1),...
        this.wt_KW_min(time-1),curtailed,bat_powers,LD_disaster_min);
        [Yb_mindisaster] = Ybus(LD_mindisaster,nbr,nbus);
        [Vmag_min_disaster, theta_min_disaster, Pcalc_min_disaster, Qcalc_min_disaster,P_GRID_MIN]= gen_cap(this, ...
            BD_mindisaster,Yb_mindisaster,nbus,this.time,this.State(this.IDX_POWER_GEN_MIN));
        if ismember(time, this.event_time)
            LEI_max= (BD_maxdisaster(12,7) + Pcalc_max_disaster(12,1)) / (BD_maxnodisaster(12,7) +Pcalc_max_nodisaster(12,1));
            LEI_min = (Pcalc_min_disaster(12,1) + BD_mindisaster(12,7)) / (BD_minnodisaster(12,7) + Pcalc_min_nodisaster(12,1));
            LEI_max_unscaled = 1000*this.Sbase*(BD_maxdisaster(12,7) + Pcalc_max_disaster(12,1)) - (BD_maxnodisaster(12,7) +Pcalc_max_nodisaster(12,1));
            LEI_min_unscaled = 1000*this.Sbase*(Pcalc_min_disaster(12,1) + BD_mindisaster(12,7)) - (BD_minnodisaster(12,7) + Pcalc_min_nodisaster(12,1));
        else
            LEI_max = 1;
            LEI_min = 1;
            LEI_max_unscaled = 0;
            LEI_min_unscaled = 0;
        end
        VDI_max = (sum(ones(33,1)-abs(Vmag_max_disaster))) / (sum(ones(33,1)-abs(Vmag_max_nodisaster)));
        VDI_min =  (sum(ones(33,1)-abs(Vmag_min_disaster))) / (sum(ones(33,1)-abs(Vmag_min_nodisaster)));
        resilience_metrics = [max(LEI_max,0) max(LEI_min,0) VDI_max VDI_min LEI_max_unscaled LEI_min_unscaled];
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
        next_state(this.IDX_CURTAILED_SUM) = curtailed + this.State(this.IDX_CURTAILED_SUM); %for constraint
        next_state(this.IDX_TIME) = time;
        charge = max(bat_powers, 0);      
        discharge = max(-bat_powers, 0);   
        next_state(this.IDX_SOC) = State(this.IDX_SOC) + (charge * 0.95 - discharge / 0.95);
        next_state(this.IDX_PGRID_MAX) = P_GRID_MAX;
        next_state(this.IDX_PGRID_MIN) = P_GRID_MIN;
        next_state(this.IDX_BROKEN) = 0;
        rows_broken = rowsToRemove_max + 180;
        next_state(rows_broken) = 1;
        [Vmagdata] = [Vmag_max_disaster ;Vmag_min_disaster];
    end

    function [Observation,Reward,IsDone] = step(this,Action)
    %% Get action limits
    
    min_incentive = this.State(this.IDX_MARKET_MINPRICE)*0.3;
    max_incentive = this.State(this.IDX_MARKET_MINPRICE); %constraint 8
    bat_min = max(-this.Pbatmax*ones(4,1),(this.SOC_min - this.State(this.IDX_SOC))*0.95);
    bat_max=  min(this.Pbatmax*ones(4,1),(this.SOC_max - this.State(this.IDX_SOC))/0.95);
    max_action = [0.6.*this.State(this.IDX_PROSUMER_PKW); max_incentive; bat_max]; %constraint 4
    min_action = [zeros(this.n_cust,1);min_incentive; bat_min];
    Scaled_Action = this.scale_action(Action,max_action,min_action);

    %% Update state 
    this.time = this.time + 1;
    [Observation_old,resilience_metric,Vmagdata,x] = this.update_state(this.State, Scaled_Action, this.time);
    
    %% Reward + constraints
    [Reward, logStruct,Observation] = ...
    this.calculate_reward(Scaled_Action, Observation_old, this.time, this.State,resilience_metric,Vmagdata,x); %R(s',a, s)
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
        this.AllLogs{end+1} = this.EpisodeLogs;
        this.EpisodeLogs = {};
        this.f1 = 0;
        this.f2 = 0;
        this.f3 = 0;
        this.f4 = 0;
        this.f5 = 0;
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
        PGEN_MIN = 10;
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
    function [penalty, benefit_diff,violations] = indivdiual_consumer_benefit(this, incentive, curtailed, discomforts, prev_benefit, time)
        epsilon = this.discomforts';
        curtailed = curtailed ./ 1000;
        
        % Compute benefit difference
        benefit_diff = (epsilon .* incentive .* curtailed - (1 - epsilon) .* discomforts) + prev_benefit;
        
        % Identify violations and non-violations
        violations = benefit_diff < 0;
        non_violations = ~violations;
        violations = benefit_diff(violations);
        
        % Double the penalty for violations
        %scaled_benefits = benefit_diff;
        %scaled_benefits(violations) = 2 * scaled_benefits(violations);
        %penalty = -sum(scaled_benefits);
        % % Compute penalty
        % if time == 24
        %     % Scale violations by 1e4 and add non-violations normally
        %     penalty = sum(1e3 * abs(benefit_diff(violations))) - sum(benefit_diff(non_violations));
        % else
        %     % Regular sum for other times   
        %     penalty = -sum(benefit_diff);
        % end
        penalty = -sum(benefit_diff(non_violations));
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

    % function [penalty] = constraint_penalty(this, action, lower_limit, upper_limit)
    %     % Initialize penalty array
    %     penalty = zeros(size(action));
    % 
    %     % Apply penalty for exceeding upper limit
    %     penalty(action > upper_limit) = (action(action > upper_limit) - upper_limit(action > upper_limit)).^2 * this.PENALTY_FACTOR;
    % 
    %     % Apply penalty for violating lower limit
    %     penalty(action < lower_limit) = (lower_limit(action < lower_limit) - action(action < lower_limit)).^2 * this.PENALTY_FACTOR;
    % 
    % 
    % end

    function [reward,logStruct, Observation_updated] = calculate_reward(this, scaled_action, next_state, time, state,resilience_metric,Vmagdata,x)
        %% actions
        curtailed = scaled_action(1:32);
        incentive = scaled_action(33);
        bat_powers = scaled_action(34:end);
        %% constants 
        P_grid_max = state(this.IDX_PGRID_MAX);
        P_grid_min = state(this.IDX_PGRID_MIN);
        discomforts = this.calculate_discomforts(curtailed, state(this.IDX_PROSUMER_PKW));
        %% interval optimisation, penalties and costs
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
        balance_penalty_max = 0; %always 0
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
        [consumer_benefit_penalty, benefit,flag] = this.indivdiual_consumer_benefit(incentive, curtailed, discomforts, state(this.IDX_BENEFIT_SUM),time);
        

        % Constraint 9
        [budget_limit_penalty, budget] = this.budget_limit_constraint(incentive, curtailed, state(this.IDX_BUDGET_SUM),time);

        % % Action constraint (always adhered to)
        % lower_limit = [zeros(5,1);0.3*state(this.IDX_MARKET_MINPRICE);ones(4,1)*this.SOC_min];
        % upper_limit = [0.6.*state(this.IDX_PROSUMER_PKW);state(this.IDX_MARKET_MINPRICE);ones(4,1)*this.SOC_max];
        % [action_penalty] = this.constraint_penalty([curtailed;incentive;state(this.IDX_SOC)], lower_limit,upper_limit);
        
        % resilience metric
        F5_max = -(resilience_metric(1) + resilience_metric(3));
        F5_min = -(resilience_metric(2) + resilience_metric(4));
        F5 = (F5_max + F5_min) / 2;
 
        consumer_benefit_limit = max(0,time/2*consumer_benefit_penalty); %TODO have a look at this if bad
        % if any(flag)
        %     consumer_benefit_limit = consumer_benefit_limit - time/2*sum(flag);
        % end
        penalties_max = time/2 * (balance_penalty_max  + daily_curtailment_penalty...
            + budget_limit_penalty + generation_penalty_max + ramp_penalty_max)+ consumer_benefit_limit;
        penalties_min = time/2* (balance_penalty_min  + daily_curtailment_penalty...
            + budget_limit_penalty + generation_penalty_min + ramp_penalty_min)  + consumer_benefit_limit; %actionb penalty here
        penalties = (penalties_max + penalties_min) / 2;
        
        %% Compute total reward
        generation_cost = generation_cost / this.ref_f2;
        power_transfer_cost = power_transfer_cost / this.ref_f1;
        mgo_profit = mgo_profit / this.ref_f3;
        reward = -this.w1 *(generation_cost) - this.w2*(power_transfer_cost) + this.w3 * (mgo_profit)...
        - penalties + this.w4 * F5;
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
            this.f1min = this.f1min +power_transfer_cost_min;
            this.f1max = this.f1max +power_transfer_cost_max;
            this.f2min = this.f2min + generation_cost_min;
            this.f2max = this.f2max + generation_cost_max;
            this.f3min = mgo_profit * this.ref_f3;
            this.f3max = mgo_profit * this.ref_f3;
            this.f5min = this.f5min + F5_min;
            this.f5max = this.f5max + F5_max;
            this.f1 = this.f1 + power_transfer_cost;
            this.f2 = this.f2 + generation_cost;
            this.f3 = this.f3 + mgo_profit;
            this.f4 = this.f4 + penalties;
            this.f5 = this.f5 + F5;
            this.LEI_MAX = resilience_metric(5);
            this.LEI_MIN = resilience_metric(6);
            this.VDI_min = this.VDI_min+resilience_metric(3);
            this.VDI_max = this.VDI_max+resilience_metric(4);
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
            'only_f5', F5,...
            'VDI_avg', (resilience_metric(3) + resilience_metric(4))/2, ...
            'LEI_avg', (resilience_metric(1) + resilience_metric(2))/2, ...
            'action', scaled_action, ...
            'vmag', Vmagdata, ...
            'tie_lines', x,...
            'f1min', this.f1min, ...
            'f1max',this.f2max, ...
            'f2min',this.f2min, ...
            'f2max',this.f2max, ...
            'f3min',this.f3min, ...
             'f3max',this.f3max, ...
            'f5min',this.f5min, ...
            'f5max',this.f5max, ...
            'LEI_MAX_unscaled', this.LEI_MAX, ...
            'LEI_MIN_unscaled', this.LEI_MIN, ...
            'vdi_min', this.VDI_min, ...
            'vdi_max', this.VDI_max ...
        );
        else
            logStruct = 0;
        end
    end
end

end
