function [] = singleagent_evaluator(seed,agent,dir,algo)
% singleagent_evaluator - evalutes single agent. draws pretty figures
T = 24; 
full_dir = fullfile(regexprep(dir, 's\d+_', sprintf('s%d_', seed)), sprintf('Agent%d.mat', agent));
saved_agent = load(full_dir).saved_agent;
env = Copy_of_environment(0);
saved_agent.UseExplorationPolicy = 0;
observations = zeros(env.N_OBS, T+1);
rewards = zeros(1, T);
Action_scaled = zeros(33,T);
Action = zeros(33,T);
done_flags = false(1, T);
observations(:,1) = env.State; 
tic
for t = 1:T
    
    currentObs = observations(:, t);
    action = cell2mat(getAction(saved_agent, currentObs));
    %% calculating scaled actions
    min_incentive = currentObs(env.IDX_MARKET_MINPRICE)*0.3;
    max_incentive = currentObs(env.IDX_MARKET_MINPRICE); %constraint 8
    max_action = [0.6*currentObs(env.IDX_PROSUMER_PKW); max_incentive]; %constraint 4
    min_action = [zeros(32,1);min_incentive];
    action_scaled = env.scale_action(action,max_action,min_action);
    Action_scaled(:,t) = action_scaled;
    Action(:,t) = action;
    [obs, reward, isDone] = env.step(action);
    observations(:, t+1) = obs;
    rewards(t) = reward;
    done_flags(t) = isDone;
end
t = toc;

%% store results
consumer_penalty = [];
daily = [];
mgo_profit = [];
for w= 1:24
    P_grid_max(w) = env.EpisodeLogs{1, w}.P_grid_max;
    P_grid_min(w) = env.EpisodeLogs{1, w}.P_grid_min;
    generator_power(w) = env.EpisodeLogs{1, w}.generation_max;
    consumer_penalty(w) = env.EpisodeLogs{1, w}.consumer_benefit_penalty;
    daily(w) = env.EpisodeLogs{1, w}.daily_curtailment_penalty;
    f1(w) = env.EpisodeLogs{1, w}.power_transfer_cost_culm;
    f2(w) =  env.EpisodeLogs{1, w}.generator_cost_culm;
    f3(w) =  env.EpisodeLogs{1, w}.mgo_profit_culm;
    ramp_penalty(w) = env.EpisodeLogs{1, w}.ramp_penalty;
    generator(w) = env.EpisodeLogs{1, w}.generation_penalty;
end

%% calculations
curtailed = Action_scaled(1:32,:);
incentives = Action_scaled(33,:);
consumer_load = observations(43:74,:);
env.customer_ids_residential(end) = env.customer_ids_residential(end)-1;
consumer_sum = sum(observations(43:74,w));
f1min = env.EpisodeLogs{1, T}.f1min;
f1max = env.EpisodeLogs{1, T}.f1max;
f2min = env.EpisodeLogs{1, T}.f2min;
f2max = env.EpisodeLogs{1, T}.f2max;
f1_interval = abs(f1max-f1(T));
f2_interval = abs(f2max-f2(T));
mean_total_load = mean([sum(consumer_load(env.customer_ids_residential,:).*env.load_resi',1); ...
    sum(consumer_load(env.customer_ids_commercial,:).*env.load_comm',1); ...
    sum(consumer_load(env.customer_ids_industrial,:).*env.load_indu',1)], 1);
[max_value, max_load_index] = max(mean_total_load);
disp("Algorithm" + " "+ algo)
disp("Inference time per hour" + " " +t/24 + "s")
disp("Reward:" + " " + sum(rewards))
disp("f1:" + " " + env.EpisodeLogs{1, T}.power_transfer_cost_culm + "±" + f1_interval)
disp("f2:" + " " + env.EpisodeLogs{1, T}.generator_cost_culm+ "±" + f2_interval)
disp("f3:" + " " + env.EpisodeLogs{1, T}.mgo_profit_culm)
disp("Penalties:" + " " + env.EpisodeLogs{1, T}.sum_penalties)
disp("Total daily curtailed" + " " + sum(sum(curtailed)) + "kWh");
disp("Total daily consumer load" + " " + consumer_sum + "kWh");
disp("Curtailed/Demand daily" + " " + sum(sum(curtailed))/consumer_sum)

%% PH 2024 fig. 6
incentive_hourly = incentives .* curtailed / 1000;
[a,b] = max(sum(incentive_hourly,1));
max_curt = a / incentives(b);
load_reduction = 100*(sum(curtailed(:,b))/sum(observations(43:74,b)));
disp("Curtailed amount at max incentive time"+ " " + max_curt*1000 + "kWh")
disp("Max incentive time" + " " + b + "hours")
disp("Peak load time" + " " +max_load_index + "hour" )
disp("Load reduction at peak incentive time" + " " + load_reduction + "%");
yyaxis left;
bar(sum(incentive_hourly,1));
ylabel('Incentive ($)');
set(gca,'YColor','k');
yyaxis right;
plot(incentives,'-.','LineWidth',1.5,'Color','r');
ylabel('Incentive rate ($/MWh)');
set(gca,'YColor','r');
xlabel("Time (h)")
legend("Curtailed","Incentive Rate")
%Legends and Formatting
legend("Total incentive", "Incentive Rate",'Location','best');
set(gca, 'FontName', 'Times', 'FontSize', 24, 'FontWeight', 'bold');
ax = gca;
set(gcf, 'Position', [100, 100, 1000, 800]); %pos from left, pos from bottom, width, height
%Set background to white for better appearance:
set(gcf,'Color','w');
end 