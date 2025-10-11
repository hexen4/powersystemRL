function [trainingInfo] = training_CaseI(training, seed, reconfiguration,HL_size,algo,resume_from)
% TRAINING_CASEI - Train a DDPG agent with RNN for a given environment setup
%
% Inputs:
%   training        - logical flag or parameter for environment
%   seed            - random seed for reproducibility
%   reconfiguration - environment reconfiguration flag
%
% Output:
%   trainingInfo    - training statistics from the RL training session

    %% Environment Setup
    env = Copy_of_environment;
    env.training = training;
    env.reconfiguration = reconfiguration;

    rng(seed);
    obsInfo = getObservationInfo(env);
    actInfo = getActionInfo(env);
    %% Training Options
    trainOpts = rlTrainingOptions( ...
        'MaxEpisodes',                3, ...
        'MaxStepsPerEpisode',         500, ...
        'StopTrainingCriteria',       "AverageReward", ...
        'StopTrainingValue',          10000, ...
        'ScoreAveragingWindowLength', 5, ...
        'SaveAgentCriteria',          "AverageReward", ...
        'SaveAgentValue',             100, ...
        'SaveAgentDirectory',         "savedAgents", ...
        'Verbose',                    false, ...
        'Plots',                      "training-progress");

    if nargin >= 6 && ~isempty(resume_from)
        % Resume from saved agent
        loaded_data = load(resume_from);
        agent = loaded_data.saved_agent;
        fprintf('Resuming training from: %s\n', resume_from);
    else
        if algo == "DDPG"
            %% Observation and Action Info

        
            %% ===== ACTOR NETWORK (Recurrent) =====
            scaleLayer = scalingLayer('Name', 'scale', 'Scale', 0.5, 'Bias', 0.5);
        
            actorNet = [
                featureInputLayer(obsInfo.Dimension(1), 'Name', 'input')
                fullyConnectedLayer(HL_size, 'Name', 'fc1')
                reluLayer('Name', 'relu1')
                layerNormalizationLayer('Name', 'layernorm1')
                fullyConnectedLayer(HL_size, 'Name', 'fc2')
                reluLayer('Name', 'relu2')
                dropoutLayer(0.07, 'Name', 'dropout')
                fullyConnectedLayer(prod(actInfo.Dimension), 'Name', 'fc_out')
                tanhLayer('Name', 'tanh')
                scaleLayer
            ];
            actor = rlContinuousDeterministicActor(actorNet, obsInfo, actInfo,UseDevice="gpu");
          %% ===== CRITIC NETWORK =====
            statePath = [
                featureInputLayer(obsInfo.Dimension(1), 'Name', 'state')
                fullyConnectedLayer(HL_size, 'Name', 'fc1_state')
                %reluLayer('Name', 'relu_state')
            ];
            
            actionPath = [
                featureInputLayer(prod(actInfo.Dimension), 'Name', 'action')
                fullyConnectedLayer(HL_size, 'Name', 'fc1_action')
                %reluLayer('Name', 'relu_action')
            ];
            
            commonPath = [
                concatenationLayer(1, 2, 'Name', 'concat')
                fullyConnectedLayer(HL_size, 'Name', 'fc2')
                reluLayer('Name', 'relu2')
                fullyConnectedLayer(HL_size/2, 'Name', 'fc3')
                reluLayer('Name', 'relu3')
                dropoutLayer(0.07, 'Name', 'dropout')
                fullyConnectedLayer(1, 'Name', 'value')
            ];
            
            criticNet = layerGraph(statePath);
            criticNet = addLayers(criticNet, actionPath);
            criticNet = addLayers(criticNet, commonPath);
            criticNet = connectLayers(criticNet, 'fc1_state', 'concat/in1');
            criticNet = connectLayers(criticNet, 'fc1_action', 'concat/in2');
            critic = rlQValueFunction(criticNet, obsInfo, actInfo,UseDevice="gpu");
        
            %% ===== AGENT OPTIONS =====
            agentOpts = rlDDPGAgentOptions( ...
                'SampleTime',            1, ...
                'DiscountFactor',        0.99, ...
                'TargetSmoothFactor',    1e-3, ...
                'ExperienceBufferLength',5e5, ...
                'MiniBatchSize',         128, ...
                 'NumStepsToLookAhead',   24 ... %1 for RNN. number of future rewards used to estimate value (how many gammas)
             );
                %'SequenceLength',        1, ...
               
        end
        % Optimizer Settings
        agentOpts.CriticOptimizerOptions.LearnRate         = 1e-3;
        agentOpts.CriticOptimizerOptions.GradientThreshold = 500;
        agentOpts.ActorOptimizerOptions.LearnRate          = 1e-3;
        agentOpts.ActorOptimizerOptions.GradientThreshold  = 500;
        % Exploration Noise
        agentOpts.NoiseOptions.StandardDeviation          = 0.45;
        agentOpts.NoiseOptions.StandardDeviationDecayRate = 1e-4;
        agentOpts.NoiseOptions.Mean                       = 0;
        agentOpts.NoiseOptions.InitialAction = 0.2;
        agentOpts.NoiseOptions.MeanAttractionConstant = 0.1;
    
        %% ===== CREATE AND TRAIN AGENT =====
        agent = rlDDPGAgent(actor, critic, agentOpts);
        agent.ExperienceBuffer = rlPrioritizedReplayMemory(obsInfo,actInfo,5e5);
    end
    agent.UseExplorationPolicy = true;
    %% Training with Timing
    totalStart = tic;
    trainingInfo = train(agent, env, trainOpts);
    totalTime = toc(totalStart);
    
    %% Timing Information
    numEpisodes = numel(trainingInfo.EpisodeReward);
    timingInfo = struct(...
        'TotalTrainingTime', totalTime, ...
        'NumberOfEpisodes', numEpisodes, ...
        'AverageEpisodeTime', totalTime / numEpisodes, ...
        'Timestamp', datetime('now'));

    fprintf('=== Training Timing ===\n');
    fprintf('Total time: %.2f seconds (%.2f minutes)\n', totalTime, totalTime/60);
    fprintf('Episodes: %d\n', numEpisodes);
    fprintf('Time per episode: %.2f seconds\n', totalTime/numEpisodes);
    
    %% Save Results (CORRECTED)
    results = struct();
    results.trainingInfo = trainingInfo;
    results.timingInfo = timingInfo;
    results.algorithm = algo;
    results.trainingParams = struct(...
        'seed', seed, ...
        'reconfiguration', reconfiguration, ...
        'HL_size', HL_size);
    
    filename = sprintf('training_results_%s_%s.mat', algo, datestr(now, 'yyyy-mm-dd_HH-MM-SS'));
    save(filename, 'results', 'agent');
    
    fprintf('Results saved to: %s\n', filename);
end
