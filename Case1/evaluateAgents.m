function agentResults = evaluateAgents(reconfiguration, folderPath, seed, HL_size, L2, LR_actor, LR_critic, DF, algo)
    % evaluateAgents - Evaluates all agents saved in a folder and returns a results table.
    %
    % This function runs through all saved agents, evaluates them in the
    % environment, logs their results, and deletes any agent whose final
    % episode violates the completion condition.
    %
    % Output:
    %   agentResults : table containing AgentName, F1, F2, F3, Reward

    T = 24; % number of timesteps per episode
    
    env = Copy_of_environment(0);
    agentFiles = dir(fullfile(folderPath, '*.mat'));
    numAgents = numel(agentFiles);
    agentResults = table('Size', [numAgents, 6], ...
        'VariableTypes', {'string','single','single','single','single','single'}, ...
        'VariableNames', {'AgentName', 'F1', 'F2', 'F3', 'penalty_sum','Reward'});

   for i = 1:numAgents
    env.reset();
    if reconfiguration == 1
        env.reconfiguration = 1;
    else
        env.reconfiguration = 0;
    end

    filePath = fullfile(folderPath, agentFiles(i).name);
    observations = zeros(env.N_OBS, T+1);
    saved_agent = load(filePath).saved_agent;
    saved_agent.UseExplorationPolicy = 0;
    rewards = zeros(1, T);
    Action_scaled = zeros(33, T);
    Action = zeros(33, T);
    observations(:,1) = env.State;
    for t = 1:T
        currentObs = observations(:, t);
        action = cell2mat(getAction(saved_agent, currentObs));

        % Scaled actions
        min_incentive = currentObs(env.IDX_MARKET_MINPRICE)*0.3;
        max_incentive = currentObs(env.IDX_MARKET_MINPRICE);
        max_action = [0.6*currentObs(env.IDX_PROSUMER_PKW); max_incentive];
        min_action = [zeros(32,1); min_incentive];
        action_scaled = env.scale_action(action, max_action, min_action);
        Action_scaled(:,t) = action_scaled;
        Action(:,t) = action;

        [obs, reward, isDone] = env.step(action);
        observations(:, t+1) = obs;
        rewards(t) = reward;

    end 

    f1 = env.EpisodeLogs{1, T}.power_transfer_cost_culm;
    f2 = env.EpisodeLogs{1, T}.generator_cost_culm;
    f3 = env.EpisodeLogs{1, T}.mgo_profit_culm;
    finalBenefit = env.EpisodeLogs{1, T}.Benefit;
    finalDailyCurt = env.EpisodeLogs{1, T}.daily_curtailment_penalty;



    % Check completion condition
    if all(finalBenefit > 0) && finalDailyCurt == 0
        agentResults.AgentName(i) = string(agentFiles(i).name);
        agentResults.F1(i) = f1;
        agentResults.F2(i) = f2;
        agentResults.F3(i) = f3;
        agentResults.penalty_sum(i) = 0;
        agentResults.Reward(i) = sum(rewards);
    else
        %if failed reawrd = -1e6;
        agentResults.AgentName(i) = string(agentFiles(i).name);
        agentResults.F1(i) = f1;
        agentResults.F2(i) = f2;
        agentResults.F3(i) = f3;
        agentResults.penalty_sum(i) = sum(finalBenefit) + finalDailyCurt;
        agentResults.Reward(i) = -1e6;
    end
   

    end

if nargin >= 3
% Create filename matching saveDir pattern
tableName = sprintf('table_s%d_r%d_h%d_L2%d_LRa%.4f_LRc%.4f_DF%.2f_%s.mat', ...
seed, reconfiguration, HL_size, L2, LR_actor, LR_critic, DF, algo);

% Save in the same folder as agents
tablePath = fullfile(tableName);
save(tablePath, 'agentResults');
fprintf('Results table saved as: %s\n', tableName);
end
end



   
