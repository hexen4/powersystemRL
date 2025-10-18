close all; 
%clear all;
% VDI_without = load('savedconstants/VDI_withoutagent.mat').bus_voltages;
saved_agent = load('savedAgents2_s1_r1_h256_L21.000000e-04_LRa0.0010_LRc0.0010_DF0.90_SAC\Agent7384.mat').saved_agent;
env = Copy_of_environment_case3(0);
T = 24; 
observations = zeros(env.N_OBS, T+1);
rewards = zeros(1, T);
Action_scaled = zeros(37,T);
Action = zeros(37,T);
done_flags = false(1, T);
zero_action = [zeros(32,1);0;0.5*ones(4,1)]; 
trained = 1;
observations(:,1) = env.State;  
for t = 1:T
    currentObs = observations(:, t);
    if trained == 1
        saved_agent.UseExplorationPolicy = 0;
        action = cell2mat(getAction(saved_agent, currentObs));
    else
        action = zero_action;
    end 
%% calculating scaled actions
    min_incentive = currentObs(env.IDX_MARKET_MINPRICE)*0.3;
    max_incentive = currentObs(env.IDX_MARKET_MINPRICE); %constraint 8
    bat_min = max(-env.Pbatmax*ones(4,1),0.95*(env.SOC_min - currentObs(env.IDX_SOC)));
    bat_max=  min(env.Pbatmax*ones(4,1),(env.SOC_max - currentObs(env.IDX_SOC))/0.95);
    max_action = [0.6.*env.State(env.IDX_PROSUMER_PKW); max_incentive; bat_max]; %constraint 4
    min_action = [zeros(32,1);min_incentive; bat_min];
    
    action_scaled = env.scale_action(action,max_action,min_action);
    Action_scaled(:,t) = action_scaled;
    Action(:,t) = action;
    % applying non-scaled actions (
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
batt =  Action_scaled(34:37,:);
SOC = observations(174:177,:);
tie_line = zeros(5,24);
check = 0;
for w= 1:24
    P_grid_max(w) = env.EpisodeLogs{1, w}.P_grid_max;
    P_grid_min(w) = env.EpisodeLogs{1, w}.P_grid_min;
    consumer_penalty(w) = env.EpisodeLogs{1, w}.consumer_benefit_penalty;
    daily(w) = env.EpisodeLogs{1, w}.daily_curtailment_penalty;
    VDI(w) =env.EpisodeLogs{1, w}.VDI_avg;
    LEI(w) = env.EpisodeLogs{1, w}.LEI_avg;
    f1(w) = env.EpisodeLogs{1, w}.f1;
    f2(w) =  env.EpisodeLogs{1, w}.f2;
    f3(w) =  env.EpisodeLogs{1, w}.f3;
    f4(w) =  env.EpisodeLogs{1, w}.f4;
    ramp_penalty(w) = env.EpisodeLogs{1, w}.ramp_penalty;
    generator(w) = env.EpisodeLogs{1, w}.generation_penalty;
    f5(w) =  env.w4*env.EpisodeLogs{1, w}.f5;
    only_f5(w) =  env.w4*env.EpisodeLogs{1, w}.only_f5;
    bus_voltages{w} = env.EpisodeLogs{1,w}.vmag;
    tie_line(1:length(env.EpisodeLogs{1,w}.tie_lines),w) = env.EpisodeLogs{1,w}.tie_lines;
    LEI_unscaled(w) = (env.EpisodeLogs{1,w}.LEI_MAX_unscaled+env.EpisodeLogs{1,w}.LEI_MIN_unscaled)/2;
    [vmagmin(w), loc(w)] = min(env.EpisodeLogs{1,w}.vmag(1:33));
    market_price(w) = env.market_prices(w);
end



LEI_unscaled_without =[0	0	0	0	0	0	0	0	0	0	90.4188167387160	77.9695650197411	89.2358123167236	53.0247256589972	0	0	0	109.428180873084	104.008673435107	101.927719781951	117.644642767904	0	0	0];
LEI_diff = LEI_unscaled - LEI_unscaled_without; %kW
disp('reward' + " " + sum(rewards))
disp('LEI_DIFF' + " " + sum(LEI_diff))
% AVG_LEI_unscaled = (env.EpisodeLogs{1, 24}.LEI_MAX_unscaled+env.EpisodeLogs{1, 24}.LEI_MIN_unscaled)/2;
% disp("AVG_LEI" + " " + AVG_LEI_unscaled/1000)

disp("f1:" + " " + env.EpisodeLogs{1, 24}.f1)
disp("f2:" + " " + env.EpisodeLogs{1, 24}.f2)
disp("f3:" + " " +env.EpisodeLogs{1, 24}.f3)
disp("f4:" + " " +env.EpisodeLogs{1, 24}.f4)
disp("f5:" + " " + env.EpisodeLogs{1, 24}.f5)
disp("VDI:"+ " " + sum(VDI))
disp("LEI:"+ " " + sum(LEI))
disp("w1f1" + " " + env.w1*env.EpisodeLogs{1, 24}.f1)
disp("w2f2" + " " + env.w2*env.EpisodeLogs{1, 24}.f2)
disp("w3f3" + " " + env.w3*env.EpisodeLogs{1, 24}.f3)
disp("w4f4" + " " + env.w4*env.EpisodeLogs{1, 24}.f5)

%% Resilience Plot
% yline(1)
% hold on
% plot(LEI)
% legend("NO HILF","HILF")
% title("Resilience metric throughout day")
% xlabel("Time (hrs)")
% ylabel("Loss of energy (p.u.) (normalised with no HILF + reconfig.)")

%% Battery plot
% Create figure
figure1 = figure;

% Create axes
axes1 = axes('Parent',figure1);
hold(axes1,'on');

% Activate the left side of the axes
yyaxis(axes1,'left');
% Create multiple lines using matrix input to bar
bar1 = bar(batt','BarLayout','stacked','Parent',axes1);
set(bar1(4),'DisplayName','P^t_1','FaceColor',[0 1 1]);
set(bar1(3),'DisplayName','P^t_2','FaceColor',[0 1 0]);
set(bar1(2),'DisplayName','P^t_3','FaceColor',[0 0 1]);
set(bar1(1),'DisplayName','P^t_4','FaceColor',[1 0 1]);

% Create xline
xline(11,'DisplayName','Market Price','Parent',axes1,'LineWidth',2);

% Create xline
xline(14,'Parent',axes1,'LineWidth',2);

% Create xline
xline(18,'Parent',axes1,'LineWidth',2);

% Create xline
xline(21,'Parent',axes1,'LineWidth',2);

% Create ylabel
ylabel('P_{b,ch/dis} (kW)');

% Activate the right side of the axes
yyaxis(axes1,'right');
% Create plot
plot(market_price,'LineWidth',2,'Color',[0 0 0]);

% Create ylabel
ylabel('Market Price ($/MWh)');

% Create xlabel
xlabel('Time (Hour)','FontWeight','bold');

box(axes1,'on');
hold(axes1,'off');
% Set the remaining axes properties
set(axes1,'FontName','Times','FontSize',24,'FontWeight','bold');
% Create legend
legend(axes1,'show');

% Create textbox
annotation(figure1,'textbox',...
    [0.462279293739952 0.522999998807908 0.116372391653304 0.0625000011920932],...
    'String',{'HILF Event'},...
    'FontWeight','bold',...
    'FontSize',24,...
    'FitBoxToText','off',...
    'EdgeColor',[1 1 1]);

% Create textbox
annotation(figure1,'textbox',...
    [0.674157303370749 0.523499998807909 0.116372391653303 0.0625000011920932],...
    'String',{'HILF Event'},...
    'FontWeight','bold',...
    'FontSize',24,...
    'FitBoxToText','off',...
    'EdgeColor',[1 1 1]);

%% actions
% figure()
% % plot(env.market_prices(1:end-1))
% % hold on 
% plot(incentives)
% legend("incentives")
% 
% figure()
% plot(curtailed')
% legend("C1","C2","C3","C4","C5")
% 
% figure()
% plot(batt')
% legend("BAT1","BAT2","BAT3","BAT4")

% % %% penalties
% figure()
% plot(f1)
% hold on
% plot(f2)
% hold on
% plot(f3)
% hold on
% plot(f4)
% hold on
% plot(f5)
% legend("f1","f2","f3","f4","f5")
%% gen power plot
% figure()
% observations_DR = observations;
% observations_noaction = load("obs_noaction.mat").observations2;
% obs_mean = (observations(1,:)+observations(2,:))/2;
% obs2_mean = (observations_noaction(1,:)+observations_noaction(2,:))/2;
% obs3_mean = (observations_DR(1,:)+observations_DR(2,:))/2;
% plot(obs_mean(1,:));
% hold on
% plot(obs2_mean(1,:))
% hold on
% plot(obs3_mean(1,:))
% legend("Power loss without DR program in reconfigured network","Power loss without DR program in original network", "Power loss with DR program in original network" )
% ylabel("Power (kW)")
% xlabel("Time (Hour)")

%% objective functions
% plot(f1)
% hold on
% plot(f2)
% hold on
% plot(f3)
% hold on
% plot(f4)
% hold on
% plot(f5)
% legend("f1","f2","f3","f4","f5")

%% resilience
% figure()
% plot(VDI)
% hold on
% VDI_metricwo = load('C:\Users\jlhb83\OneDrive - Durham University\Desktop\case2\savedconstants\VDI_metric_without.mat').VDI;
% plot(VDI_metricwo)
% legend("VDI","VDI baseline")

% VDI
% VDI_with = bus_voltages;
% buses = linspace(1,33,33);
% time = linspace(1,24,24);
% VDI_without = load('savedconstants/VDI_withoutagent.mat').bus_voltages;
% V_with = cell2mat(VDI_with); 
% V_max = V_with(1:33,:);
% V_without = cell2mat(VDI_without);  
% V_max_without = V_without(1:33,:);
% % Create a meshgrid for plotting. 
% % Meshgrid returns matrices for x and y axes: here x (time) and y (buses).
% [TimeGrid, BusGrid] = meshgrid(time, buses);
% figure();
% mesh(TimeGrid, BusGrid, V_max,'FaceColor', 'r', 'EdgeColor','k');
% hold on
% mesh(TimeGrid, BusGrid, V_max_without,'FaceColor', 'w', 'EdgeColor','k');
% legend("DDPG", "Baseline")
% xlabel('Time (h)');
% ylabel('Bus Number');
% zlabel('Voltage Magnitude (p.u.)');
% xlim([0 24])
% ylim([0 33])
% zlim([0.90 1])
% set(gca, 'FontName', 'Times', 'FontSize', 24, 'FontWeight', 'bold');
% ax = gca;
% set(gcf, 'Position', [100, 100, 1000, 800]); 
% print(gcf, 'voltage_deviation.png', '-dpng', '-r800');
% % %colorbar;  % adds a color scale
% % 
% % 
%% LEI
% figure()
% LEI_with = LEI;
% LEI_without = load("savedconstants/LEI_without.mat").LEI;
% yyaxis left;
% bara = bar(batt','stacked');
% bara(1).FaceColor = 'c';
% bara(2).FaceColor = 'g';
% bara(3).FaceColor = 'b';
% bara(4).FaceColor = 'm';
% ylabel("Pbat (kW)")
% yyaxis right;
% plot(LEI_with, "LineWidth",2, "Color","r")
% xlabel('Time (hours)', 'FontWeight', 'bold', 'FontSize', 20);
% xline(11)
% xline(14)
% xline(18)
% xline(21)
% hold on
% plot(LEI_without,"LineWidth",2,"Color","r")
% ylim([0.8 1.7])
% ylabel("Average LEI (p.u.)")
% legend("P^1_t","P^2_t","P^3_t","P^4_t","LEI DDPG","LEI Baseline", Location="best")
% set(gca, 'FontName', 'Times', 'FontSize', 20, 'FontWeight', 'bold');
% ax = gca;
% set(gcf, 'Position', [100, 100, 1000, 800]); 
% % print(gcf, 'SOC.png', '-dpng', '-r800');
% 
% %% battery
% figure()
% yyaxis left;
% bara = bar(batt','stacked');
% bara(1).FaceColor = 'c';
% bara(2).FaceColor = 'g';
% bara(3).FaceColor = 'b';
% bara(4).FaceColor = 'm';
% ylabel('P_{b,ch/dis} (kW)','Interpreter','tex')
% xlabel('Time (Hour)', 'FontWeight', 'bold', 'FontSize', 20);
% xline(11)
% xline(14)
% xline(18)
% xline(21)
% hold on
% yyaxis right;
% plot(env.market_prices(1:24), "LineWidth",2, "Color","k")
% ylim([30 200])
% ylabel("Market Price ($/MWh)")
% legend("P^t_1","P^t_2","P^t_3","P^t_4","Market Price")
% set(gca, 'FontName', 'Times', 'FontSize', 24, 'FontWeight', 'bold');
% ax = gca;
% set(gcf, 'Position', [100, 100, 1000, 800]); 
% print(gcf, 'SOCvsprice.png', '-dpng', '-r800');

%% PARETO FRONT
% Load data
% Load data
% agentresults = load("savedconstants\caseII_table.mat").agentResults;
% f1 = agentresults.F1;
% f2 = agentresults.F2;
% f3 = agentresults.F3;
% f4 = agentresults.F4;
% f5 = agentresults.F5;
% total_cost = f1 + f2;
% 
% figure()
% 
% % -- Plot Pareto Front with a distinct style --
% scatter3(total_cost, f3, f5, ...
%     70, ...                       % Marker size (larger for clarity)
%     's', ...                      % Marker shape: squares
%     'LineWidth',1.2, ...          % Thicker outline
%     'MarkerEdgeColor','k', ...    % Black edge
%     'MarkerFaceColor','w');       % White fill
% 
% xlabel("F_1 (p.u.)")
% ylabel("F_2 (p.u.)")
% zlabel("F_3 (p.u.)")
% zlim([-48.6 -47.8])
% ylim([0.4 0.7])
% xlim([2 2.05])
% set(gca, 'FontName', 'Times', 'FontSize', 24, 'FontWeight', 'bold');
% set(gcf, 'Position', [100, 100, 1000, 800]);
% 
% % Identify and plot the selected solution
% tol = 1e-3;
% targetF5 = -47.8066;
% selectedIdx = find(abs(f5 - targetF5) < tol);
% 
% hold on
% % -- Plot Selected Solution (filled red circle) --
% scatter3(total_cost(selectedIdx), f3(selectedIdx), f5(selectedIdx), ...
%     120, ...                     % Marker size
%     'o', ...                     % Marker shape: circle
%     'MarkerEdgeColor','k', ...   % Black edge
%     'MarkerFaceColor','r');      % Red fill
% 
% legend("Pareto Front", "Selected Agent", "Location","best")
% print(gcf, 'pareto_frontier.png', '-dpng', '-r800');

%%PH fig.5 
% grid_mean = (P_grid_min + P_grid_max) / 2;
% WT_mean = (observations(6,:) + observations(7,:)) / 2;
% PV_mean = (observations(4,:) + observations(5,:)) / 2;
% gen_power_mean = (observations(1,:) + observations(2,:)) / 2;
% grid_mean = (observations(44,:));
% WT_mean = (observations(6,:));
% PV_mean = (observations(4,:));
% gen_power_mean = (observations(1,:));
% observations_noDR = load("savedconstants/obs_noaction.mat").observations2;
% load_demand = observations(8,1:24)-sum(curtailed,1)+sum(batt,1);
% data = [grid_mean(1:24);WT_mean(1:24);PV_mean(1:24);gen_power_mean(1:24)]';
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
% Set background to white for better appearance:
% set(gcf,'Color','w');

% Save figure as PNG at high resolution (e.g., 300 dpi):
%print(gcf, 'fig5b.png', '-dpng', '-r800');

% PH fig. 7
% figure()
% CPKW = observations(11:15,2:25);
% data = [CPKW;-curtailed]';
% bar(data,'stacked')
% legend("C1","C2","C3","C4","C5")
% ylabel("Power (KW)")
% xlabel("Time (Hour)")
% a = get(gca,'XTickLabel');
% set(gca,'XTickLabel',a,'FontName','Times','fontsize',24, 'FontWeight','bold')
