reconfiguration = 1;
%% evaluate a specific agent
% filepath = 'savedAgents_t1_s42_r1_h512_SAC\Agent4749.mat'; 
% singleagent_evaluator(filepath) %displays f1,f2,f3, reward



%% training
% saves struct - make sure not to interrupt (7-10 hours). otherwise, input file path of last saved agent into this function

% traininginfo_reconfig_seed42 = training_CaseI(1,42,1,512,"SAC"); %training =1,seed=42,reconfiguration = 1, HL_size = 512
% traininginfo_noreconfig_seed42 = training_CaseI(1,42,0,512,"DDPG");
% traininginfo_reconfig_seed1 = training_CaseI(1,1,1,512,"DDPG");

%% evaluate saved agents
%outputs table containing AgentName, F1, F2, F3, Reward. deletes constraint violating agents

%filepath = 'Case1\savedAgents_1DPPG\';
%table_h512_t1_s42_r1_h512 = evaluateAgents(reconfiguration,filepath); 

