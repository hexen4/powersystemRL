%   clc; clear; close all;
%load('C:\Users\jlhb83\OneDrive - Durham University\Desktop\savedAgents\Agent1102.mat')
% warning('off','all')
% close all
% folderPath = 'savedAgents\'; % change to your folder
% agentFiles = dir(fullfile(folderPath, '*.mat'));
% numAgents = numel(agentFiles);
% agentResults = table('Size', [numAgents, 6], ...
%     'VariableTypes', {'string','single','single','single','single','single'}, ...
%     'VariableNames', {'AgentName', 'F1', 'F2', 'F3', 'F4','Reward'});

% for i = 1:numel(agentFiles)
%     filePath = fullfile(folderPath, agentFiles(i).name);
%     agent = load(filePath).saved_agent;
%     agent.UseExplorationPolicy = 0;
env = new_environment();
T = 24;
observations = zeros(env.N_OBS, T+1);
rewards = zeros(1, T);
Action_scaled = zeros(6,T);
Action = zeros(6,T);
done_flags = false(1, T);
zero_action = [-.1*ones(5,1);-1];
trained = 0 ;
if trained == 1
    agent = agent1_Trained;
    agent.UseExplorationPolicy = 0;
end
observations(:,1) = env.State;
tic;
for t = 1:T
    currentObs = observations(:, t);
    if trained == 1
        action = cell2mat(getAction(agent, currentObs));
    else
        action = zero_action;
    end
    %calculating scaled actions
    min_incentive = currentObs(env.IDX_MARKET_MINPRICE)*0.3;
    max_incentive = currentObs(env.IDX_MARKET_MINPRICE); %constraint 8
    max_action = [0.6*420.*ones(5,1); max_incentive]; %constraint 4
    max_action = [0.6*currentObs(11:15); max_incentive]; %constraint 4
    min_action = [zeros(5,1);min_incentive];
    action_scaled = env.scale_action(action,max_action,min_action);
    Action_scaled(:,t) = action_scaled;
    Action(:,t) = action;
    %applying non-scaled actions (


    [obs, reward, isDone] = env.step(action);
    observations(:, t+1) = obs;
    rewards(t) = reward;
    done_flags(t) = isDone;

    % if isDone
    %     break;
    % end
end
elapsed_time = toc; % Stop timer and get elapsed time in seconds
fprintf('Total elapsed time: %.4f seconds\n', elapsed_time);
%     if all(env.EpisodeLogs{1, T}.Benefit>0) && env.EpisodeLogs{1, T}.daily_curtailment_penalty == 0 
%         f1 = env.EpisodeLogs{T}.f1;
%         f2 = env.EpisodeLogs{T}.f2;
%         f3 = env.EpisodeLogs{T}.f3;
%         f4 = env.EpisodeLogs{T}.f4;
% 
%         % Store the results in the table
%         agentResults.AgentName(i) = string(agentFiles(i).name);
%         agentResults.F1(i) = f1;
%         agentResults.F2(i) = f2;
%         agentResults.F3(i) = f3;
%         agentResults.F4(i) = f4;
%         agentResults.Reward(i) = sum(rewards);
%     else
%         % If criteria are not met, store NaN for the metrics
%         agentResults.AgentName(i) = string(agentFiles(i).name);
%         agentResults.F1(i) = NaN;
%         agentResults.F2(i) = NaN;
%         agentResults.F3(i) = NaN;
%         agentResults.F4(i) = NaN;
%         agentResults.Reward(i) = NaN;
%     end
% end
% end
curtailed = Action_scaled(1:5,:);
incentives = Action_scaled(6,:);
vmag = zeros(66,24);
%% plotting penalties
for i= 1:T
    P_grid_max(i) = env.EpisodeLogs{1, i}.P_grid_max;
    P_grid_min(i) = env.EpisodeLogs{1, i}.P_grid_min;
    consumer_penalty(i) = env.EpisodeLogs{1, i}.consumer_benefit_penalty;
    daily(i) = env.EpisodeLogs{1, i}.daily_curtailment_penalty;
    f1(i) =env.EpisodeLogs{1, i}.f1;
    f2(i) =env.EpisodeLogs{1, i}.f2;
    f3(i) =env.EpisodeLogs{1, i}.f3;
    f4(i) =env.EpisodeLogs{1, i}.f4;
    %f5(i) =env.EpisodeLogs{1, i}.f5;
    mgo_profit_timestep(i) = env.EpisodeLogs{1, i}.mgo_profit;
    ramp_penalty(i) = env.EpisodeLogs{1, i}.ramp_penalty;
    generator(i) = env.EpisodeLogs{1, i}.generation_penalty;
    f1scaled(i) =-env.w1*env.EpisodeLogs{1, i}.f1;
    f2scaled(i) =-env.w2*env.EpisodeLogs{1, i}.f2;
    f3scaled(i) =env.w3*env.EpisodeLogs{1, i}.f3;
    f4scaled(i) =-env.w4*env.EpisodeLogs{1, i}.f4;
    vmag(:,i) = env.EpisodeLogs{1, i}.vmag;
    %action_penalties(i) = env.EpisodeLogs{1, i}.action_penalty;
    %penalties_sum(i) = env.EpisodeLogs{1, i}.sum_penalties;
end
disp(sum(rewards))
disp("f1:" + " " + env.EpisodeLogs{1, T}.f1)
disp("f2:" + " " + env.EpisodeLogs{1, T}.f2)
disp("f3:" + " " + env.EpisodeLogs{1, T}.f3)
disp("f4:" + " " + env.EpisodeLogs{1, T}.f4)
gen_power_max = sum(observations(1,:))/1e3;
gen_power_min = sum(observations(2,:))/1e3;
gen_power = (gen_power_max + gen_power_min) / 2;
grid_power_max = sum(P_grid_max)/1e3;
grid_power_min = sum(P_grid_min)/1e3;
grid_power = (grid_power_max + grid_power_min) / 2;
C1_total = sum(curtailed(1,:))/1000; %MWh/day
C2_total = sum(curtailed(2,:))/1000; %MWh/day
C3_total = sum(curtailed(3,:))/1000; %MWh/day
C4_total = sum(curtailed(4,:))/1000; %MWh/day
C5_total = sum(curtailed(5,:))/1000; %MWh/day
% if all(env.EpisodeLogs{1, T}.Benefit>0) && env.EpisodeLogs{1, T}.daily_curtailment_penalty == 0 && env.EpisodeLogs{1, T}.mgo_profit_culm > 340
%     disp(filePath + " " + env.EpisodeLogs{1, T}.mgo_profit_culm)
% end
%% actions
% figure()
% plot(env.market_prices(1:end-1))
% hold on 
% plot(incentives)
% legend("incentives", "market price")
% figure()
% plot(curtailed')
% legend("C1","C2","C3","C4","C5")
% 
%% plotting obj. function
% figure()
% plot(consumer_penalty)
% hold on
% plot(daily)
% hold on
% plot(mgo_profit_timestep)
% legend("consumer","daily","mgo_timestep")
%% verify action constraints true 
% scaled_obs = 0.6.*observations(11:15,1:end-1);
% constraint = scaled_obs - Action_scaled(1:5,:) < 0;




