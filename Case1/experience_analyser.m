%% analysing buffer 

episodes = length(agent1_Trained_Trained.ExperienceBuffer);
all_experiences = agent1_Trained_Trained.ExperienceBuffer.allExperiences;
last_episode = all_experiences(end-22:end);
actions = zeros(33,episodes);
Action_scaled_buffer = zeros(33,episodes);
obs = zeros(173,episodes);
next_obs = zeros(173,episodes);
rewards_experience = zeros(1, episodes);
rewards_calc = zeros(1, episodes);
env = Copy_of_environment();
init_obs = env.State;
for i = 1:episodes
    actions(:,i) = last_episode(i).Action{1,1};
    obs(:,i) = last_episode(i).Observation{1,1};
    next_obs(:,i) = last_episode(i).NextObservation{1,1};
    rewards_experience(i) = all_experiences(i).Reward;
    action_scaled = env.scale_action(last_episode(i).Action{1,1},[0.6*420.*ones(5,1); 50],[zeros(5,1);50*0.3]);
    Action_scaled_buffer(:,i) = action_scaled;
    [~, reward, isDone] = env.step(actions(:,i));
    observations(:,i+1) = obs2;
    rewards_calc(i) = reward;
end
T = T+1;
for i= 1:T
    P_grid_max(i) = env.EpisodeLogs{1, i}.P_grid_max;
    P_grid_min(i) = env.EpisodeLogs{1, i}.P_grid_min;
    consumer_penalty(i) = env.EpisodeLogs{1, i}.consumer_benefit_penalty;
    daily(i) = env.EpisodeLogs{1, i}.daily_curtailment_penalty;
    mgo_profit_culm(i) =env.EpisodeLogs{1, i}.mgo_profit_culm;
    mgo_profit_timestep(i) = env.EpisodeLogs{1, i}.mgo_profit;
    %action_penalties(i) = env.EpisodeLogs{1, i}.action_penalty;
    penalties_sum(i) = env.EpisodeLogs{1, i}.sum_penalties;
end

disp("f1:" + " " + env.EpisodeLogs{1, T}.power_transfer_cost_culm)
disp("f2:" + " " + env.EpisodeLogs{1, T}.generator_cost_culm)
disp("f3:" + " " + env.EpisodeLogs{1, T}.mgo_profit_culm)
disp("f4:" + " " + env.EpisodeLogs{1, T}.sum_penalties)
reward = -env.w1*env.EpisodeLogs{1, T}.power_transfer_cost_culm ...
-env.w2*env.EpisodeLogs{1, T}.generator_cost_culm + ...
env.w3*env.EpisodeLogs{1, T}.mgo_profit_culm - ...
env.w4*env.EpisodeLogs{1, T}.sum_penalties;
disp("reward_calculated" + " " + reward)
disp("reward_stepping" + " " + sum(rewards_calc))
disp("reward_agent" + " " + sum(rewards_experience))
gen_power_max = sum(obs(1,:))/1e3;
gen_power_min = sum(obs(2,:))/1e3;
gen_power = (gen_power_max + gen_power_min) / 2;
grid_power_max = sum(P_grid_max)/1e3;
grid_power_min = sum(P_grid_min)/1e3;
grid_power = (grid_power_max + grid_power_min) / 2;

plot(consumer_penalty)
hold on
plot(daily)
hold on
%plot(action_penalties)
hold on
plot(mgo_profit_timestep)
legend("consumer","daily","mgo_timestep")
