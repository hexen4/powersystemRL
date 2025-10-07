% clearvars -except agent1_Trained
close all
agent = agent1_Trained;
episode_length = 23;
episodes = 20000;
total = episodes * episode_length;
all_experiences = agent.ExperienceBuffer.allExperiences;
for i = 1:total
    step_reward(i) = all_experiences(i).Reward;
end 

episode_rewardPPO = zeros(1, episodes);

for episode = 1:episodes
    start_idx = (episode - 1) * episode_length + 1;
    end_idx = episode * episode_length;

    episode_rewardPPO(episode) = sum(step_reward(start_idx:end_idx));
end

% %% Case I learning curve

load('C:\Users\rando\OneDrive - Durham University\Desktop\Case1\results\DDPG_episodereward.mat')
load('C:\Users\rando\OneDrive - Durham University\Desktop\Case1\results\SAC_episodereward.mat')
load('C:\Users\rando\OneDrive - Durham University\Desktop\Case1\results\TD3_episodereward.mat')
yline(-227.2284,LineWidth=2)
hold on
plot(episode_rewardSAC)
hold on
plot(DDPG)
hold on
plot(episode_reward)
legend("Baseline","SAC","DDPG","TD3")
ylabel("Reward($/day)")
xlabel("Episode")
set(gca, 'FontName', 'Times', 'FontSize', 24, 'FontWeight', 'bold');
ax = gca;
set(gcf, 'Position', [100, 100, 1000, 800]); 
legend('Location', 'southeast');
% Set background to white for better appearance:
set(gcf,'Color','w');

print(gcf, 'all.png', '-dpng', '-r800');