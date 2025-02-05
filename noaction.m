clc; clear; close all;

env = environment();
T = 23; 
observations = zeros(env.N_OBS, T);
rewards = zeros(1, T);
done_flags = false(1, T);
zero_action = zeros(6,1); 

for t = 1:T
    [obs, reward, isDone] = env.step(zero_action);
    
    observations(:, t) = obs;
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