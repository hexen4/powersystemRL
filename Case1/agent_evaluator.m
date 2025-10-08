T = 24; 
folderPath = 'savedAgents_1DPPG/'; % change to your folder
agentFiles = dir(fullfile(folderPath, '*.mat'));
agentResults_sorted =sortrows(agentResults2,"Reward","descend");
agentFiles = agentResults_sorted.AgentName(1:100);
numAgents = numel(agentFiles);
agentResults = table('Size', [numAgents, 5], ...
    'VariableTypes', {'string','single','single','single','single'}, ...
    'VariableNames', {'AgentName', 'F1', 'F2', 'F3','Reward'});
trained = 1;
zero_action = zeros(33,1);
%agent = load('saved_sessions\9bbestagent(full).mat').saved_agent;
%agent = agent1_Trained
for i = 1:numel(agentFiles)
    env = Copy_of_environment();
    %filePath = fullfile(folderPath, agentFiles(i).name);
    filePath = fullfile(folderPath, agentFiles(i));
    observations = zeros(env.N_OBS, T+1);
    saved_agent = load(filePath).saved_agent;
    rewards = zeros(1, T);
    Action_scaled = zeros(33,T);
    Action = zeros(33,T);
    done_flags = false(1, T);
    zero_action = ones(33,1); 
    observations(:,1) = env.State;  
    for t = 1:T
        currentObs = observations(:, t);
        if trained == 1
            action = cell2mat(getAction(saved_agent, currentObs));
        else
            action = zero_action;
        end 
        %% calculating scaled actions
        min_incentive = currentObs(env.IDX_MARKET_MINPRICE)*0.3;
        max_incentive = currentObs(env.IDX_MARKET_MINPRICE); %constraint 8
        %max_action = [0.6*420.*ones(5,1); max_incentive]; %constraint 4
        max_action = [0.6*currentObs(env.IDX_PROSUMER_PKW); max_incentive]; %constraint 4
        min_action = [zeros(32,1);min_incentive];
        action_scaled = env.scale_action(action,max_action,min_action);
        Action_scaled(:,t) = action_scaled;
        Action(:,t) = action;
        %% applying non-scaled actions (
    
    
        [obs, reward, isDone] = env.step(action);
        observations(:, t+1) = obs;
        rewards(t) = reward;
        done_flags(t) = isDone;
    
        % if isDone
        %     break;
        % end
    end
    %% store results
    consumer_penalty = [];
    daily = [];
    mgo_profit = [];
    curtailed = Action_scaled(1:32,:);
    incentives = Action_scaled(33,:);
    for w= 1:24
        P_grid_max(w) = env.EpisodeLogs{1, w}.P_grid_max;
        P_grid_min(w) = env.EpisodeLogs{1, w}.P_grid_min;
        consumer_penalty(w) = env.EpisodeLogs{1, w}.consumer_benefit_penalty;
        daily(w) = env.EpisodeLogs{1, w}.daily_curtailment_penalty;
        %VDI(w) =env.EpisodeLogs{1, w}.VDI_avg;
        %LEI(w) = env.EpisodeLogs{1, w}.LEI_avg;
        f1(w) = env.w1*env.EpisodeLogs{1, w}.power_transfer_cost_culm;
        f2(w) =  env.w2*env.EpisodeLogs{1, w}.generator_cost_culm;
        f3(w) =  env.w3*env.EpisodeLogs{1, w}.mgo_profit_culm;
        %f4(w) =  env.w4*env.EpisodeLogs{1, w}.f4;
        ramp_penalty(w) = env.EpisodeLogs{1, w}.ramp_penalty;
        generator(w) = env.EpisodeLogs{1, w}.generation_penalty;
        %f5(w) =  env.w5*env.EpisodeLogs{1, w}.f5;
        %bus_voltages{w} = env.EpisodeLogs{1,w}.vmag;
        %tie_line(1:length(env.EpisodeLogs{1,w}.tie_lines),w) = env.EpisodeLogs{1,w}.tie_lines;
        %LEI_unscaled(w) = (env.EpisodeLogs{1,w}.LEI_MAX_unscaled+env.EpisodeLogs{1,w}.LEI_MIN_unscaled)/2;
        %[vmagmin(w), loc(w)] = min(env.EpisodeLogs{1,w}.vmag(1:33));
    end
    if all(env.EpisodeLogs{1, T}.Benefit>0) && env.EpisodeLogs{1, T}.daily_curtailment_penalty == 0 
        f1 = env.EpisodeLogs{1, w}.power_transfer_cost_culm;
        f2 = env.EpisodeLogs{1, w}.generator_cost_culm;        
        f3 = env.EpisodeLogs{1, w}.mgo_profit_culm;
        %f4 = env.EpisodeLogs{T}.f4;
        %f5 = env.EpisodeLogs{T}.f5;

        % Store the results in the table
        agentResults.AgentName(i) = string(agentFiles(i));
        % agentResults.AgentName(i) = string(agentFiles(i).name);
        agentResults.F1(i) = f1;
        agentResults.F2(i) = f2;
        agentResults.F3(i) = f3;
      %  agentResults.F4(i) = f4;
       % agentResults.F5(i) = f5;
       % agentResults.Vallbig(i) = check;
        agentResults.Reward(i) = sum(rewards);
    else
        % If criteria are not met, store NaN for the metrics
        % agentResults.AgentName(i) = string(agentFiles(i).name);
        agentResults.AgentName(i) = string(agentFiles(i));
        agentResults.F1(i) = NaN;
        agentResults.F2(i) = NaN;
        agentResults.F3(i) = NaN;
       % agentResults.F4(i) = NaN;
        %agentResults.F5(i) = NaN;
      %  agentResults.Vallbig(i) = NaN;
       % agentResults.Reward(i) = NaN;
    end
end

% elapsed_time = toc; % Stop timer and get elapsed time in seconds
% fprintf('Total elapsed time: %.4f seconds\n', elapsed_time);
% Vmag = zeros(66,24);
% curtailed = Action_scaled(1:32,:);
% incentives = Action_scaled(33,:);
% %% plotting penalties
% for i= 1:T
%     P_grid_max(i) = env.EpisodeLogs{1, i}.P_grid_max;
%     P_grid_min(i) = env.EpisodeLogs{1, i}.P_grid_min;
%     consumer_penalty(i) = env.EpisodeLogs{1, i}.consumer_benefit_penalty;
%     daily(i) = env.EpisodeLogs{1, i}.daily_curtailment_penalty;
%     mgo_profit_culm(i) =env.EpisodeLogs{1, i}.mgo_profit_culm;
%     mgo_profit_timestep(i) = env.EpisodeLogs{1, i}.mgo_profit;
%     ramp_penalty(i) = env.EpisodeLogs{1, i}.ramp_penalty;
%     generator(i) = env.EpisodeLogs{1, i}.generation_penalty;
%     f1scaled(i) =-env.w1*env.EpisodeLogs{1, i}.power_transfer_cost_culm;
%     f2scaled(i) =-env.w2*env.EpisodeLogs{1, i}.generator_cost_culm;
%     f3scaled(i) =env.w3*env.EpisodeLogs{1, i}.mgo_profit_culm;
%     f4scaled(i) =env.EpisodeLogs{1, i}.sum_penalties;
%     %Vmag(:,i) = env.EpisodeLogs{1, i}.vmagdata;
%     %action_penalties(i) = env.EpisodeLogs{1, i}.action_penalty;
%     %penalties_sum(i) = env.EpisodeLogs{1, i}.sum_penalties;
% end
% disp(sum(rewards))
% disp("f1:" + " " + env.EpisodeLogs{1, T}.power_transfer_cost_culm)
% disp("f2:" + " " + env.EpisodeLogs{1, T}.generator_cost_culm)
% disp("f3:" + " " + env.EpisodeLogs{1, T}.mgo_profit_culm)
% disp("f4:" + " " + env.EpisodeLogs{1, T}.sum_penalties)
% 
% gen_power_max = sum(observations(1,:))/1e3;
% gen_power_min = sum(observations(2,:))/1e3;
% gen_power = (gen_power_max + gen_power_min) / 2;
% grid_power_max = sum(P_grid_max)/1e3;
% grid_power_min = sum(P_grid_min)/1e3;
% grid_power = (grid_power_max + grid_power_min) / 2;
% C1_curtail= sum(curtailed(1,:))/1000; %MWh/day
% C2_curtail = sum(curtailed(2,:))/1000; %MWh/day
% C3_curtail= sum(curtailed(3,:))/1000; %MWh/day
% C4_curtail = sum(curtailed(4,:))/1000; %MWh/day
% C5_curtail = sum(curtailed(5,:))/1000; %MWh/day
% C1_total = sum(observations(11,:))/1000;
% C2_total = sum(observations(12,:))/1000;
% C3_total = sum(observations(13,:))/1000;
% C4_total = sum(observations(14,:))/1000;
% C5_total = sum(observations(15,:))/1000;
% total_curt = C1_curtail + C2_curtail+ C3_curtail+C4_curtail + C5_curtail;
% disp("Total curtailed" + total_curt)

%% actions
% figure()
% plot(env.market_prices(1:end-1), LineWidth=2)
% hold on 
% plot(incentives,LineWidth=2)
% legend("Market Price", "Incentive Rate")
% ylabel("Price ($/MWh)")
% xlabel("Time (Hour)")

% figure()
% plot(curtailed')
% legend("C1","C2","C3","C4","C5")
% 
%% plotting obj. function
% figure()
% plot(consumer_penalty)
% hold on
% plot(daily)
% hold on
% plot(mgo_profit_timestep)
% legend("consumer","daily","mgo_timestep")


%% verify action constraints true 
% scaled_obs = 0.6.*observations(11:15,1:end-1);
% constraint = scaled_obs - Action_scaled(1:5,:) < 0;

% %%PH fig.5 
% grid_mean = (P_grid_min + P_grid_max) / 2;
% WT_mean = (observations(6,:) + observations(7,:)) / 2;
% PV_mean = (observations(4,:) + observations(5,:)) / 2;
% gen_power_mean = (observations(1,:) + observations(2,:)) / 2;
% grid_mean = (P_grid_max);
% WT_mean = (observations(6,:));
% PV_mean = (observations(4,:));
% gen_power_mean = (observations(1,:));
% load("savedconstants/obs_noaction.mat")
% load_demand = observations(8,1:24)-sum(curtailed,1);
% data = [grid_mean;WT_mean(1:24);PV_mean(1:24);gen_power_mean(1:24)]';
% bar(data,'stacked')
% hold on
% plot(load_demand, LineWidth=2)
% hold on
% plot(observations_noDR(8,1:24),LineWidth=2)
% 
% legend("Power exchange between grid and MG", ...
%     "Power output of wind-based DG","Power output of solar-based DG", ...
%     "Power output of conventional DG","Post DR load demand","Actual load demand")
% ylabel("Power (KW)")
% xlabel("Time (Hour)")
% set(gca, 'FontName', 'Times', 'FontSize', 24, 'FontWeight', 'bold');
% ax = gca;
% set(gcf, 'Position', [100, 100, 1000, 800]); 
% legend('Location', 'southeast');
% % Set background to white for better appearance:
% set(gcf,'Color','w');
% % 
% % Save figure as PNG at high resolution (e.g., 300 dpi):
% print(gcf, 'fig5b.png', '-dpng', '-r800');

% % PH fig. 7
% 
% load("obs_noaction.mat")
% CPKW = observations(11:15,2:25);
% data = [CPKW;-curtailed]';
% bar(data,'stacked')
% legend("C1","C2","C3","C4","C5")
% ylabel("Power (KW)")
% xlabel("Time (Hour)")
% a = get(gca,'XTickLabel');
% set(gca,'XTickLabel',a,'FontName','Times','fontsize',26, 'FontWeight','bold')


%% PH 2024 fig. 6
% incentive_hourly = incentives .* curtailed / 1000;
% yyaxis left;
% bara = bar(incentive_hourly','stacked');
% bara(1).FaceColor = 'c';
% bara(2).FaceColor = 'g';
% bara(3).FaceColor = 'b';
% bara(4).FaceColor = 'm';
% bara(5).FaceColor = 'y';
% ylabel('Incentive ($)');
% set(gca,'YColor','k');

% Right Y-axis (Line plot)
% yyaxis right;
% plot(incentives,'-.','LineWidth',1.5,'Color','r');
% ylabel('Incentive rate ($/MWh)');
% set(gca,'YColor','r');
% xlabel("Time (h)")
% legend("C1","C2","C3","C4","C5","Incentive Rate")
% % Legends and Formatting
% %legend([bar_handle; plot(NaN,NaN,'-or')], {'Category 1','Category 2','Category 3','Incentive'},...
% %    'Location','best');
% 
% set(gca, 'FontName', 'Times', 'FontSize', 24, 'FontWeight', 'bold');
% ax = gca;
% set(gcf, 'Position', [100, 100, 1000, 800]); 
% legend('Location', 'northeast');
% % Set background to white for better appearance:
% set(gcf,'Color','w');
% 
% % Save figure as PNG at high resolution (e.g., 300 dpi):
% print(gcf, 'incentivefig.png', '-dpng', '-r800');