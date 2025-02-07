%clc; clear; close all;

env = environment();
T = 23; 
observations = zeros(env.N_OBS, T+1);
rewards = zeros(1, T);
done_flags = false(1, T);
zero_action = [ones(5,1);1]; 
trained = 1;
observations(:,1) = env.State;  
for t = 1:T
    currentObs = observations(:, t);
    if trained == 1
        action = cell2mat(getAction(agent1_Trained_1, currentObs));
    else
        action = zero_action;
    end 
    [obs, reward, isDone] = env.step(action);
    observations(:, t+1) = obs;
    rewards(t) = reward;
    done_flags(t) = isDone;

    if isDone
        break;
    end
end

% Display results
disp('Observations:');
disp(observations);
disp('Rewards:');
disp(rewards);
disp('Episode Completion:');
disp(done_flags);