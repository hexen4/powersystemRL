function [] = plot_learningcurve(seeds, savedirect)
%% load results table
    ALGOS = ["SAC","DDPG","TD3"];
    seed_results = {};
    training_time = [];
    max_episodes = 15000;

    algo_mean = zeros(15000,3);
    algo_std = zeros(15000,3);
    for w = 1:length(ALGOS)
        savedirec_used = savedirect{w};
        for i = 1:length(seeds)
            seed = seeds(i);
            savedirectory = regexprep(savedirec_used, 's\d+_', sprintf('s%d_', seed));
            % Load the data
            data = load(savedirectory);
            training_time = [training_time data.results.timingInfo.TotalTrainingTime];
            all_rewards(1:max_episodes, i) = data.results.trainingInfo.EpisodeReward;
            
        end
 
        final_training_results{w} = all_rewards; %123 -> SAC, DPPG, TD3
        max_value = -inf;
        max_row = 0;
        max_col = 0;
            
        for col = 1:length(seeds)
            for row = 1:15000
                current_value = final_training_results{w}(row, col);
                if current_value > max_value
                    max_value = current_value;
                    max_row = row;
                    max_col = col;
                end
            end
        end

        if w == 1
            fprintf('Maximum SAC value: %f at Agent %d, seed %d\n', max_value, max_row, max_col);
        elseif w == 2
            fprintf('Maximum DDPG value: %f at Agent %d, seed %d\n', max_value, max_row, max_col);
        elseif w == 3
            fprintf('Maximum TD3 value: %f at Agent %d, seed %d\n', max_value, max_row, max_col);
        end
        
        algo_mean(:,w) = mean(final_training_results{w},2);
        algo_std(:,w) = std(final_training_results{w},[],2);

    end



    SAC_mean = algo_mean(:,1);
    DDPG_mean = algo_mean(:,2);
    TD3_mean = algo_mean(:,3);

    SAC_mean_clipped = max(SAC_mean, -2000);
    DDPG_mean_clipped = max(DDPG_mean, -2000);
    TD3_mean_clipped = max(TD3_mean, -2000);

    %% fig. 3a
    figure()
    yline(-365.5146,LineWidth=2)
    hold on
    plot(SAC_mean_clipped,LineWidth=1)
    hold on
    plot(DDPG_mean_clipped,LineWidth=1)
    hold on
    plot(TD3_mean_clipped,LineWidth=1)
    legend("Baseline","SAC","DDPG","TD3")
    ylabel("Reward($/day)")
    xlabel("Episode")
    set(gca, 'FontName', 'Times', 'FontSize', 24, 'FontWeight', 'bold');
    ax = gca;
    set(gcf, 'Position', [100, 100, 1000, 800]); 
    legend('Location', 'southeast');
    set(gcf,'Color','w');
    ylim([-2500 0])
    xlim([0 15000])
    %print(gcf, 'all.png', '-dpng', '-r800');

    display("Mean SAC training time "+ " " + mean(training_time(1,5)))
    display("Mean DDPG training time "+ " " + mean(training_time(6,10)))
    display("Mean TD3 training time "+ " " + mean(training_time(11,15)))
end