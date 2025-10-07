clearvars -except agent1_Trained saved_agent savedAgentResult
close all

episode_length = 23;
episodes = 20000;
total = episodes * episode_length;
all_experiences = agent1_Trained.ExperienceBuffer.allExperiences;
for i = 1:total
    step_reward(i) = all_experiences(i).Reward;
end 

episode_reward_DDPG = zeros(1, episodes);

for episode = 1:episodes
    start_idx = (episode - 1) * episode_length + 1;
    end_idx = episode * episode_length;

    episode_reward_DDPG(episode) = sum(step_reward(start_idx:end_idx));
end
%display(max(episode_reward_DDPG));
episode_reward_DDPG = max(-400,episode_reward_DDPG);
yline(-2.1413,LineWidth=1)
hold on
plot(episode_reward_DDPG,LineWidth=2)
legend("Baseline","DDPG")
ylabel("Reward(p.u.)")
xlabel("Episode")
set(gca, 'FontName', 'Times', 'FontSize', 24, 'FontWeight', 'bold');
ax = gca;
% set(gcf, 'Position', [100, 100, 1000, 800]); 
set(gcf, 'Position', [200, 200, 1500, 800]); 
legend('Location', 'southeast');
% Set background to white for better appearance:
set(gcf,'Color','w');
box on;  % show box around inset
episodes = 1:20000;
% axesInset = axes('Position',[0.55 0.3 0.3 0.3]); 
axesInset = axes('Position',[0.45 0.35 0.4 0.4]); 
% Plot the same data on the inset
yline(-2.1413,LineWidth=2); hold on;
plot(episodes, episode_reward_DDPG, 'b');
xlabel('Episode (zoom)');
ylabel('Reward (p.u.)');

% Set the zoomed range
xlim([19000 20000]);
ylim([-5 0]);  % adjust based on your data range
set(gca, 'FontName', 'Times', 'FontSize', 24, 'FontWeight', 'bold');
ax = gca;
set(gcf, 'Position', [100, 100, 1000, 800]); 
print(gcf, 'CASE3.png', '-dpng', '-r800');




