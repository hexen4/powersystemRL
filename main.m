close all
%% figures
warning("off")
dirPath = "C:\Users\rando\Desktop\powersystemRL\"; %PH CHANGE
learning_curve = 1; %PH CHANGE
single_agent = 0; %PH CHANGE
seeds = [1,2,3,4,5];
%% training constants
computer = 3; 
HILF = 1; 
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
SAC_dir = sprintf('savedAgents_s%d_r%d_h%d_L2%d_LRa%.4f_LRc%.4f_DF%.2f_%s', ...
                1, 1, 256, 1e-4, 1e-3, 1e-3, 0.9, "SAC");
TD3_dir = sprintf('savedAgents_s%d_r%d_h%d_L2%d_LRa%.4f_LRc%.4f_DF%.2f_%s', ...
                1, 1, 256, 1e-4, 1e-4, 1e-3, 0.99, "TD3");
DDPG_dir = sprintf('savedAgents_s%d_r%d_h%d_L2%d_LRa%.4f_LRc%.4f_DF%.2f_%s', ...
                1, 1, 256, 1e-4, 1e-3, 1e-3, 0.9, "DDPG");
saveddirec = {SAC_dir, DDPG_dir,TD3_dir};

%% figure 3b. + results
if single_agent == 1
    addpath(genpath(dirPath + 'Case1'));
    rmpath(genpath(dirPath + 'Case2'));
    DDPG_best_seed = 1; %PH CHANGE EVERYTHING from 61 to 66. need to go to relevant computer
    %i.e. SAC -> computer 1.
    SAC_best_seed= 0;
    TD3_best_seed = 0;
    DDPG_best_agent = 4;
    SAC_best_agent= 0;
    TD3_best_agent = 0;
    %comment out unneccessary func calls below. i.e. computer 1 -> only SAC
    singleagent_evaluator(DDPG_best_seed,DDPG_best_agent,[DDPG_dir '\'],"DDPG") %incentive, energy curt, total energy curt, f1,f2,f3 values
    singleagent_evaluator(TD3_best_seed,TD3_best_agent,[TD3_dir '\'],"TD3")
    singleagent_evaluator(SAC_best_seed,SAC_best_agent,[SAC_dir '\'],"SAC")
end
%% figure plotting
if learning_curve == 1
    addpath(genpath(dirPath + 'Case1'));
    rmpath(genpath(dirPath + 'Case2'));
    plot_learningcurve(seeds,saveddirec) %learning curve, table, training time
    
else
    for seed = 1:length(seeds)
        if HILF == 0
            addpath(genpath(dirPath + 'Case1'));
            rmpath(genpath(dirPath + 'Case2'));
            traininginfo = training_CaseI(episodes,seed,reconfiguration,HL_size,algo,LR_actor,LR_critic,DF,L2, ...
            soft,batch_size,temperature,experience_length); %train (var name overwritten)
            saveDir = sprintf('savedAgents_s%d_r%d_h%d_L2%d_LRa%.4f_LRc%.4f_DF%.2f_%s', ...
                seed, reconfiguration, HL_size, L2, LR_actor, LR_critic, DF, algo);
            filepath = [saveDir '\'];
            table_evaluated = evaluateAgents(reconfiguration, filepath, seed, HL_size, L2, LR_actor, LR_critic, DF, algo); %evalulate on test; scaled by 0.9
            
        else
            addpath(genpath(dirPath + 'Case2'));
            rmpath(genpath(dirPath + 'Case1'));
            traininginfo = training_CaseII(episodes,seed,reconfiguration,HL_size,algo,LR_actor,LR_critic,DF,L2, ...
            soft,batch_size,temperature,experience_length); %train (var name overwritten)
            saveDir = sprintf('savedAgents2_s%d_r%d_h%d_L2%d_LRa%.4f_LRc%.4f_DF%.2f_%s', ...
                seed, reconfiguration, HL_size, L2, LR_actor, LR_critic, DF, algo);
            filepath = [saveDir '\'];
            table_evaluated = evaluateAgents2(reconfiguration, filepath, seed, HL_size, L2, LR_actor, LR_critic, DF, algo); %evalulate on test; scaled by 0.9
        end
    
    end
end