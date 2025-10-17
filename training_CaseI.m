function [trainingInfo] = training_CaseI(episode,seed,reconfiguration,HL_size,algo,LR_actor,LR_critic,DF,L2, ...
        soft,batch_size,temperature,experience_length,resume_from)
% TRAINING_CASEI - Train an  algo WITHOUT RNN for a given environment setup
%
% Inputs:
%   training        - logical flag or parameter for environment
% %   seed            - random seed for reproducibility
%   reconfiguration - environment reconfiguration flag
%   HL_size - size of hidden layers for static 3 HL strucutre for actor /
%   critic
% algo - DDPG, TD3 or SAC. 
% resume_from(opt) - 
% Output:
%   trainingInfo    - training statistics from the RL training session
    %% Environment Setup
    env = Copy_of_environment;
    env.reconfiguration = reconfiguration;
    saveDir = sprintf('savedAgents_s%d_r%d_h%d_L2%d_LRa%.4f_LRc%.4f_DF%.2f_%s', ...
        seed, reconfiguration, HL_size, L2,LR_actor, LR_critic, DF, algo);
    % Remove any special characters that might cause issues
    rng(seed);
    obsInfo = getObservationInfo(env);
    actInfo = getActionInfo(env);
    %% Training Options
    trainOpts = rlTrainingOptions( ...
        'MaxEpisodes',                episode, ...
        'MaxStepsPerEpisode',         500, ...
        'StopTrainingCriteria',       "AverageReward", ...
        'StopTrainingValue',          10000, ...
        'ScoreAveragingWindowLength', 5, ...
        'SaveAgentCriteria',          "EpisodeReward", ...
        'SaveAgentValue',             -10, ...
        'SaveAgentDirectory',         saveDir, ...
        'Verbose',                    false, ...
        'Plots',                      "training-progress", ...
        'UseParallel',                false);          % disable parallel training
    if nargin >= 14 && ~isempty(resume_from) 
        % Resume from saved agent
        loaded_data = load(resume_from);
        agent = loaded_data.saved_agent;
        fprintf('Resuming training from: %s\n', resume_from);
    else
        if algo == "TD3"
            %% ===== TD3 ACTOR NETWORK =====
            % Input path
            inputLayer = featureInputLayer(obsInfo.Dimension(1), 'Name', 'input_1');
            % Hidden layers with Layer Normalization 
            fc1 = fullyConnectedLayer(2*HL_size, 'Name', 'fc_1');
            relu1 = reluLayer('Name', 'relu1');
            fcBody = fullyConnectedLayer(HL_size, 'Name', 'fc_body');    
            reluBody = reluLayer('Name', 'relu_body');
            % Output path
            outputLayer = fullyConnectedLayer(prod(actInfo.Dimension), 'Name', 'output');
            %softsignLayer = functionLayer(@(x) x./(1+abs(x)), 'Name', 'softsign');  
            tanhLayer = functionLayer(@(x) tanh(x), 'Name', 'tanh');
            scaleLayer = scalingLayer('Name', 'scale', 'Scale', 0.5, 'Bias', 0.5); 

            %layerNorm1 = layerNormalizationLayer('Name', 'layernorm1');
            %layerNorm2 = layerNormalizationLayer('Name', 'layernorm2');
            % Build the network graph
            actorNet = layerGraph(inputLayer);
            actorNet = addLayers(actorNet, fc1);
            actorNet = addLayers(actorNet, relu1); 
            actorNet = addLayers(actorNet, fcBody);
            actorNet = addLayers(actorNet, reluBody);
            actorNet = addLayers(actorNet, outputLayer);
            actorNet = addLayers(actorNet, tanhLayer);  
            actorNet = addLayers(actorNet, scaleLayer);
            
            %actorNet = addLayers(actorNet, layerNorm1);
            %actorNet = addLayers(actorNet, layerNorm2);
            
            % Connect all layers
            actorNet = connectLayers(actorNet, 'input_1', 'fc_1');
            actorNet = connectLayers(actorNet, 'fc_1', 'relu1');
            actorNet = connectLayers(actorNet, 'relu1', 'fc_body');   
            actorNet = connectLayers(actorNet, 'fc_body', 'relu_body');
            actorNet = connectLayers(actorNet, 'relu_body', 'output');  
            %actorNet = connectLayers(actorNet, 'layernorm2', 'output');
            actorNet = connectLayers(actorNet, 'output', 'tanh'); 
            actorNet = connectLayers(actorNet, 'tanh', 'scale'); 
            %actorNet = connectLayers(actorNet, 'softsign', 'scale'); 
            % Create TD3 actor (deterministic policy)
            actor = rlContinuousDeterministicActor(actorNet, obsInfo, actInfo, 'UseDevice', 'gpu');

            %% ===== TD3 CRITIC NETWORKS =====
            % Critic 1
            statePath1 = [
                featureInputLayer(obsInfo.Dimension(1), 'Name', 'input_1')
                %layerNormalizationLayer('Name', 'ln_state')
                fullyConnectedLayer(HL_size, 'Name', 'fc_1')
                %reluLayer('Name', 'relu_state')
            ];
            
            actionPath1 = [
                featureInputLayer(prod(actInfo.Dimension), 'Name', 'input_2')
                fullyConnectedLayer(HL_size, 'Name', 'fc_2')
                %layerNormalizationLayer('Name', 'ln_action')'
                %reluLayer('Name', 'relu_state2')
            ];
            
            commonPath1 = [
                concatenationLayer(1, 2, 'Name', 'concat')
                reluLayer('Name', 'relu2')
                fullyConnectedLayer(2*HL_size, 'Name', 'fc2')
                %layerNormalizationLayer('Name', 'ln2')  
                %fullyConnectedLayer(HL_size, 'Name', 'fc3')
                %layerNormalizationLayer('Name', 'ln3')  
                reluLayer('Name', 'relu3')
                fullyConnectedLayer(1, 'Name', 'output')
            ];
            
            criticNet1 = layerGraph(statePath1);
            criticNet1 = addLayers(criticNet1, actionPath1);
            criticNet1 = addLayers(criticNet1, commonPath1);
            criticNet1 = connectLayers(criticNet1, 'fc_1', 'concat/in1');
            criticNet1 = connectLayers(criticNet1, 'fc_2', 'concat/in2');
            critic1 = rlQValueFunction(criticNet1, obsInfo, actInfo, 'UseDevice', 'gpu');
            %% ===== TD3 AGENT OPTIONS =====
            agentOpts = rlTD3AgentOptions( ...
                'SampleTime', 1, ...
                'DiscountFactor', DF, ...
                'TargetSmoothFactor', soft, ...      % Target policy smoothing
                'TargetUpdateFrequency', 1, ...      % Delayed target updates
                'PolicyUpdateFrequency', 2, ...      % Delayed policy updates
                'ExperienceBufferLength', experience_length, ...
                'MiniBatchSize', batch_size, ...
                'NumStepsToLookAhead', 23);
            
            % Exploration Noise:
            agentOpts.ExplorationModel.StandardDeviationDecayRate = 0.00035;
            agentOpts.ExplorationModel.StandardDeviation     = 0.2;
            agentOpts.ExplorationModel.StandardDeviationMin = 0;
            %agentOpts.NoiseOptions.
            
            % Target policy smoothing noise
            agentOpts.TargetPolicySmoothModel.StandardDeviation = 0.3;
            agentOpts.TargetPolicySmoothModel.StandardDeviationMin = 0.1;
            
            % Optimizer settings
            agentOpts.ActorOptimizerOptions.LearnRate = LR_actor;
            agentOpts.CriticOptimizerOptions(1, 1).LearnRate = LR_critic;
            agentOpts.CriticOptimizerOptions(1, 2).LearnRate = LR_critic;
            agentOpts.CriticOptimizerOptions(1, 1).GradientThreshold = 1000;
            agentOpts.CriticOptimizerOptions(1, 2).GradientThreshold =1000;
            agentOpts.ActorOptimizerOptions.GradientThreshold = 1000;
            
            agent.AgentOptions.PolicyUpdateFrequency = 2; %twin DELAYED ...
            l2Factor = L2;
            agentOpts.CriticOptimizerOptions(1, 1).L2RegularizationFactor = l2Factor;  % L2 for critic
            agentOpts.CriticOptimizerOptions(1, 2).L2RegularizationFactor = l2Factor;  % L2 for critic
            agentOpts.ActorOptimizerOptions.L2RegularizationFactor = l2Factor;  % L2 for critic
            %% ===== CREATE TD3 AGENT =====
            agent = rlTD3Agent(actor, [critic1], agentOpts); %TODO check this
        elseif algo == "SAC"
            %% ===== SAC ACTOR NETWORK =====
            % Input path
            inputLayer = featureInputLayer(obsInfo.Dimension(1), 'Name', 'inputLayer');
            % Hidden layers with Layer Normalization 
            fc1 = fullyConnectedLayer(2*HL_size, 'Name', 'fc_1');
            reluBody = reluLayer('Name', 'relu_body');
            fcBody = fullyConnectedLayer(HL_size, 'Name', 'fc_body');    
            bodyOutput = reluLayer('Name', 'body_output');
            tanhMean = functionLayer(@(x) tanh(x), 'Name', 'tanh_mean');
            scaleMean = scalingLayer('Name', 'scale_mean', 'Scale', 0.5, 'Bias', 0.5);
            % Standard deviation path  
            fcStd = fullyConnectedLayer(prod(actInfo.Dimension), 'Name', 'fc_std');
            % mean path  
            fcMean = fullyConnectedLayer(prod(actInfo.Dimension), 'Name', 'fc_mean');
            clamp = functionLayer(@(x) min(max(x,-10),2), 'Name', 'clamp');
            std = functionLayer(@(x) exp(x), 'Name', 'std');
            %softsignLayer = functionLayer(@(x) x./(1+abs(x)), 'Name', 'softsign');  
            % Softplus ensures std > 0: std = log(1 + exp(x))
            %softplusStd = functionLayer(@(x) log(1 + exp(x)), 'Name', 'std');
            
            % Build the network graph
            actorNet = layerGraph(inputLayer);
            actorNet = addLayers(actorNet, fc1);
            actorNet = addLayers(actorNet, reluBody);
            actorNet = addLayers(actorNet, fcBody);
            actorNet = addLayers(actorNet, bodyOutput);
            actorNet = addLayers(actorNet, fcMean);
            actorNet = addLayers(actorNet, tanhMean);
            actorNet = addLayers(actorNet, scaleMean);
            actorNet = addLayers(actorNet, fcStd);
            %actorNet = addLayers(actorNet, softplusStd);
            actorNet = addLayers(actorNet, clamp);
            actorNet = addLayers(actorNet, std);
            % Connect the main body
            actorNet = connectLayers(actorNet, 'inputLayer', 'fc_1');
            actorNet = connectLayers(actorNet, 'fc_1', 'relu_body');
            actorNet = connectLayers(actorNet, 'relu_body', 'fc_body');
            actorNet = connectLayers(actorNet, 'fc_body', 'body_output');
            
            % Connect mean path
            actorNet = connectLayers(actorNet, 'body_output', 'fc_mean');
            actorNet = connectLayers(actorNet, 'fc_mean', 'tanh_mean');
            actorNet = connectLayers(actorNet, 'tanh_mean', 'scale_mean');
            
            % Connect standard deviation path
            actorNet = connectLayers(actorNet, 'body_output', 'fc_std');
            actorNet = connectLayers(actorNet, 'fc_std', 'clamp');
            actorNet = connectLayers(actorNet, 'clamp', 'std');
            % Create SAC actor
            actor = rlContinuousGaussianActor(actorNet, obsInfo, actInfo, ...
                'ActionMeanOutputNames', 'scale_mean', ...    % Mean output port
                'ActionStandardDeviationOutputNames', 'std', ... % Std output port
                'UseDevice', 'gpu');
            %% ===== SAC CRITIC NETWORK =====
            % Critic 1
            statePath1 = [
                featureInputLayer(obsInfo.Dimension(1), 'Name', 'input_1')
                %layerNormalizationLayer('Name', 'ln_state')
                fullyConnectedLayer(HL_size, 'Name', 'fc_1')
                %reluLayer('Name', 'relu_state')
            ];
            
            actionPath1 = [
                featureInputLayer(prod(actInfo.Dimension), 'Name', 'input_2')
                fullyConnectedLayer(HL_size, 'Name', 'fc_2')
                %layerNormalizationLayer('Name', 'ln_action')'
                %reluLayer('Name', 'relu_state2')
            ];
            
            commonPath1 = [
                concatenationLayer(1, 2, 'Name', 'concat')
                reluLayer('Name', 'relu2')
                fullyConnectedLayer(2*HL_size, 'Name', 'fc2')
                %layerNormalizationLayer('Name', 'ln2')  
                %fullyConnectedLayer(HL_size, 'Name', 'fc3')
                %layerNormalizationLayer('Name', 'ln3')  
                reluLayer('Name', 'relu3')
                fullyConnectedLayer(1, 'Name', 'output')
            ];
            
            criticNet1 = layerGraph(statePath1);
            criticNet1 = addLayers(criticNet1, actionPath1);
            criticNet1 = addLayers(criticNet1, commonPath1);
            criticNet1 = connectLayers(criticNet1, 'fc_1', 'concat/in1');
            criticNet1 = connectLayers(criticNet1, 'fc_2', 'concat/in2');
            critic1 = rlQValueFunction(criticNet1, obsInfo, actInfo, 'UseDevice', 'gpu');
            agentOpts = rlSACAgentOptions( ...
                'SampleTime', 1, ...
                'DiscountFactor', DF, ...
                'TargetSmoothFactor', soft, ...
                'ExperienceBufferLength', experience_length, ...
                'MiniBatchSize', batch_size, ...
                'NumStepsToLookAhead', 23);
            
            % SAC-specific optimizations
            agentOpts.TargetUpdateFrequency = 1;      % Update targets every step
            agentOpts.NumWarmStartSteps = 2000;
            agentOpts.EntropyWeightOptions.TargetEntropy = -prod(actInfo.Dimension); % Automatic entropy tuning
            agentOpts.EntropyWeightOptions.GradientThreshold = 5;
            agentOpts.EntropyWeightOptions.EntropyWeight = 1;
            % Create SAC agent
            % Optimizer settings
            agentOpts.ActorOptimizerOptions.LearnRate = LR_actor;
            agentOpts.CriticOptimizerOptions(1, 1).LearnRate =LR_critic;
            agentOpts.CriticOptimizerOptions(1, 2).LearnRate = LR_critic;
            agentOpts.EntropyWeightOptions.LearnRate = 1e-3;
            agentOpts.CriticOptimizerOptions(1, 1).GradientThreshold = 1000;
            agentOpts.CriticOptimizerOptions(1, 2).GradientThreshold =1000;
            agentOpts.ActorOptimizerOptions.GradientThreshold = 1000;
            l2Factor = L2;
            agentOpts.CriticOptimizerOptions(1, 1).L2RegularizationFactor = l2Factor;  % L2 for critic
            agentOpts.CriticOptimizerOptions(1, 2).L2RegularizationFactor = l2Factor;  % L2 for critic
            agentOpts.ActorOptimizerOptions.L2RegularizationFactor = l2Factor;  % L2 for critic
            agent = rlSACAgent(actor,[critic1], agentOpts);
        elseif algo == "DDPG"
            %% Observation and Action Info

        
            %% ===== ACTOR NETWORK (Recurrent) =====
            scaleLayer = scalingLayer('Name', 'scale', 'Scale', 0.5, 'Bias', 0.5);
            %layer = tanhLayer('Name','tanh');
            actorNet = [

                featureInputLayer(obsInfo.Dimension(1), 'Name', 'input')
                fullyConnectedLayer(2*HL_size, 'Name', 'fc1')
                reluLayer('Name', 'relu1')
                %layerNormalizationLayer('Name', 'layernorm1')
                fullyConnectedLayer(HL_size, 'Name', 'fc2')
                reluLayer('Name', 'relu2')
                %dropoutLayer(0.05, 'Name', 'dropout')
                fullyConnectedLayer(prod(actInfo.Dimension), 'Name', 'fc_out')
                functionLayer(@(x) tanh(x), 'Name', 'tanh')
                %softsignLayer
                scaleLayer
            ];
            actor = rlContinuousDeterministicActor(actorNet, obsInfo, actInfo,UseDevice="gpu");
            %% ===== CRITIC NETWORK =====
            statePath = [
                featureInputLayer(obsInfo.Dimension(1), 'Name', 'state')
                fullyConnectedLayer(HL_size, 'Name', 'fc1_state')
                %layerNormalizationLayer('Name', 'ln_state') 
                %reluLayer('Name', 'relu_state')
                %layerNormalizationLayer('Name', 'layernorm2')
                
            ];
            
            actionPath = [
                featureInputLayer(prod(actInfo.Dimension), 'Name', 'action')
                fullyConnectedLayer(HL_size, 'Name', 'fc1_action')
               % layerNormalizationLayer('Name', 'ln_action')  
                %reluLayer('Name', 'relu_action')
                
            ];
            
            commonPath = [
                concatenationLayer(1, 2, 'Name', 'concat')
                reluLayer('Name', 'relu2')
                fullyConnectedLayer(2*HL_size, 'Name', 'fc2')
                %layerNormalizationLayer('Name', 'ln2')     
                %fullyConnectedLayer(HL_size, 'Name', 'fc3')
                %layerNormalizationLayer('Name', 'ln3')
                reluLayer('Name', 'relu3')
                fullyConnectedLayer(1, 'Name', 'value')
            ];
            
            criticNet = layerGraph(statePath);
            criticNet = addLayers(criticNet, actionPath);
            criticNet = addLayers(criticNet, commonPath);
            criticNet = connectLayers(criticNet, 'fc1_action', 'concat/in1');
            criticNet = connectLayers(criticNet, 'fc1_state', 'concat/in2');
            critic = rlQValueFunction(criticNet, obsInfo, actInfo,UseDevice="gpu");
        
            %% ===== AGENT OPTIONS =====
            agentOpts = rlDDPGAgentOptions( ...
                'SampleTime',            1, ...
                'DiscountFactor',        DF, ... %or 0.99; 0.7
                'TargetSmoothFactor',    soft, ...
                'ExperienceBufferLength',experience_length, ...   
                'MiniBatchSize',         batch_size, ...
                 'NumStepsToLookAhead',   23 ... %1 for RNN. number of future rewards used to estimate value (how many gammas)
             );
                %'SequenceLength',        1, ... %for RNN 
            %% ===== CREATE AND TRAIN AGENT =====
            
                    % Exploration Noise:
            agentOpts.NoiseOptions.StandardDeviation          = 0.4;   % High initial exploration
            agentOpts.NoiseOptions.StandardDeviationDecayRate = 0.00015;  % Decays to ~0.1 by episode 1500
            agentOpts.NoiseOptions.Mean                       = 0;
            agentOpts.NoiseOptions.InitialAction = 0.2;
            agentOpts.NoiseOptions.MeanAttractionConstant = 0.1;     % Balanced stability
                    % Optimizer Settings
            agentOpts.CriticOptimizerOptions.LearnRate         = LR_critic;
            agentOpts.CriticOptimizerOptions.GradientThreshold = 1000;
            agentOpts.ActorOptimizerOptions.LearnRate          = LR_actor;
            agentOpts.ActorOptimizerOptions.GradientThreshold  = 1000;
            
            l2Factor =L2;
            agentOpts.ActorOptimizerOptions.L2RegularizationFactor = l2Factor;   % L2 for actor
            agentOpts.CriticOptimizerOptions.L2RegularizationFactor = l2Factor;  % L2 for critic
            agent = rlDDPGAgent(actor, critic, agentOpts);
        end

    end
    agent.UseExplorationPolicy = true;
    
    %% experience buffer
    % agent.ExperienceBuffer = rlPrioritizedReplayMemory(obsInfo,actInfo,experience_length);
    % agent.ExperienceBuffer.PriorityExponent = 0.6;
    % agent.ExperienceBuffer.InitialImportanceSamplingExponent = 0.5;
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
        'HL_size', HL_size, ...
        'learning_rate_actor', LR_actor, ...
        'learning_rate_critic', LR_critic, ...
        'batch_size', batch_size, ...
        'soft', soft, ...
        'algo', algo, ...
        'buffer_length', experience_length, ...
        'discount_factor', DF, ...
        'L2_regularization', L2, ...
        'max_episodes', episode);
    
   resultsFile = sprintf('savedAgents_s%d_r%d_h%d_L2%d_LRa%.4f_LRc%.4f_DF%.2f_%s.mat', ...
       seed, reconfiguration, HL_size, L2,LR_actor, LR_critic, DF, algo);
   save(resultsFile, 'results', 'agent');
        
    fprintf('Results saved to: %s\n', resultsFile);
end
