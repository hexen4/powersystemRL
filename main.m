computer = 1; %PH change from 1 -> 3
HILF = 1; %PH change from 1 -> 0

%% define constants
seeds = [1]; %include more when each agent works
reconfiguration = 1;
episodes = 15e3;
if computer == 1 %SAC settings from https://ieeexplore.ieee.org/document/10345718
    algo = "SAC";
    LR_actor = 1e-3;
    LR_critic = 1e-3;
    HL_size = 256;
    batch_size = 128;
    soft = 1e-3;
    DF = 0.9;
    temperature = 0.1;
    experience_length = 1e6;
    L2 = 1e-4;
elseif computer ==2 %TD3 settings from https://www.mdpi.com/1996-1073/14/3/531
    algo = "TD3";
    LR_actor = 1e-4;
    LR_critic = 1e-3;
    HL_size = 256;
    batch_size = 64;
    soft = 0.005;
    DF = 0.99;
    experience_length = 1e6;
    L2 = 1e-4;
    temperature = 0;
elseif computer ==3 %DDPG settings from https://ieeexplore.ieee.org/document/10345718
    algo = "DDPG";
    LR_actor = 1e-3;
    LR_critic = 1e-3;
    HL_size = 256;
    batch_size = 128;
    soft = 1e-3;
    DF = 0.9;
    experience_length = 1e6;
    L2 = 1e-4;
    temperature = 0;
end  

for seed = 1:length(seeds)
    if HILF == 0
        traininginfo = training_CaseI(episodes,seed,reconfiguration,HL_size,algo,LR_actor,LR_critic,DF,L2, ...
        soft,batch_size,temperature,experience_length); %train (var name overwritten)
        saveDir = sprintf('savedAgents_s%d_r%d_h%d_L2%d_LRa%.4f_LRc%.4f_DF%.2f_%s', ...
            seed, reconfiguration, HL_size, L2, LR_actor, LR_critic, DF, algo);
        filepath = [saveDir '\'];
        table_evaluated = evaluateAgents(reconfiguration, filepath, seed, HL_size, L2, LR_actor, LR_critic, DF, algo); %evalulate on test; scaled by 0.9
    else
        % traininginfo = training_CaseII(episodes,seed,reconfiguration,HL_size,algo,LR_actor,LR_critic,DF,L2, ...
        %      soft,batch_size,temperature,experience_length); %train (var name overwritten)
        saveDir = sprintf('savedAgents2_s%d_r%d_h%d_L2%d_LRa%.4f_LRc%.4f_DF%.2f_%s', ...
            seed, reconfiguration, HL_size, L2, LR_actor, LR_critic, DF, algo);
        filepath = [saveDir '\'];
        table_evaluated = evaluateAgents2(reconfiguration, filepath, seed, HL_size, L2, LR_actor, LR_critic, DF, algo); %evalulate on test; scaled by 0.9
    end

end
 