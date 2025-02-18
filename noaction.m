%clc; clear; close all;
env = environment();
T = 23; 
observations = zeros(env.N_OBS, T+1);
rewards = zeros(1, T);
Action_scaled = zeros(6,T);
Action = zeros(6,T);
done_flags = false(1, T);
zero_action = [0.5*100*ones(5,1);0]; 
trained = 1;
observations(:,1) = env.State;  
for t = 1:T
    currentObs = observations(:, t);
    if trained == 1
        action = cell2mat(getAction(saved_agent, currentObs));
    else
        action = zero_action;
    end 
    %% calculating scaled actions
    min_incentive = currentObs(env.IDX_MARKET_MINPRICE)*0.4;
    max_incentive = currentObs(env.IDX_MARKET_MINPRICE); %constraint 8
    max_action = [0.6.*currentObs(env.IDX_PROSUMER_PKW); max_incentive]; %constraint 4
    min_action = [zeros(5,1);min_incentive];
    action_scaled = env.scale_action(action,max_action,min_action);
    Action_scaled(:,t) = action_scaled;
    Action(:,t) = action;
    %% applying non-scaled actions (
    [obs, reward, isDone] = env.step(action);
    observations(:, t+1) = obs;
    rewards(t) = reward;
    done_flags(t) = isDone;

    if isDone
        break;
    end
end

% Display results
disp(sum(rewards))
%% plotting
% incentives = Action(6,:);
% scaled_incentive = env.market_prices(1:end-1).*0.4.*(incentives/100)';
% plot(env.market_prices(1:end-1))
% hold on 
% plot(scaled_incentive)
% curtailed = observations(21:25,:);
% plot(curtailed')
% legend("1","2","3","4","5")