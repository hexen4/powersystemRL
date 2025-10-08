function agentResults = evaluateAgents()
    % evaluateAgents - Evaluates all agents saved in a folder and returns a results table.
    %
    % This function runs through all saved agents, evaluates them in the
    % environment, logs their results, and deletes any agent whose final
    % episode violates the completion condition.
    %
    % Output:
    %   agentResults : table containing AgentName, F1, F2, F3, Reward

    folderPath = 'savedAgents\';
    T = 24; % number of timesteps per episode

    agentFiles = dir(fullfile(folderPath, '*.mat'));
    numAgents = numel(agentFiles);
    agentResults = table('Size', [numAgents, 5], ...
        'VariableTypes', {'string','single','single','single','single'}, ...
        'VariableNames', {'AgentName', 'F1', 'F2', 'F3', 'Reward'});

    for i = 1:numAgents
        env = Copy_of_environment();
        filePath = fullfile(folderPath, agentFiles(i).name);
        observations = zeros(env.N_OBS, T+1);
        saved_agent = load(filePath).saved_agent;
        rewards = zeros(1, T);
        Action_scaled = zeros(33, T);
        Action = zeros(33, T);
        zero_action = ones(33,1); 
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

        % Extract objective function values from logs (final step)
        f1 = env.EpisodeLogs{1, T}.power_transfer_cost_culm;
        f2 = env.EpisodeLogs{1, T}.generator_cost_culm;        
        f3 = env.EpisodeLogs{1, T}.mgo_profit_culm;

        % Check completion condition
        if all(env.EpisodeLogs{1, T}.Benefit > 0) && env.EpisodeLogs{1, T}.daily_curtailment_penalty == 0
            agentResults.AgentName(i) = string(agentFiles(i).name);
            agentResults.F1(i) = f1;
            agentResults.F2(i) = f2;
            agentResults.F3(i) = f3;
            agentResults.Reward(i) = sum(rewards);
        else
            % If criteria are not met, mark as NaN and delete the agent file
            agentResults.AgentName(i) = string(agentFiles(i).name);
            agentResults{i, 2:end} = NaN;

            % Delete the corresponding .mat file
            fprintf('Deleting agent: %s (failed completion condition)\n', agentFiles(i).name);
            delete(filePath);
        end
    end
end

    end
end
