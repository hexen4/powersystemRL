function [BD,LD,TL,rowsToRemove]=disaster_model(resi,comm,indu,pv,wt, curtailed,bat_powers,disaster,training)
customer_ids_residential =[2,3,4,6,11,12,13,15,18,21,22,25,30,31,33];
customer_ids_commercial =[5,10,14,19,20,24,27,29,32];
customer_ids_industrial =[7,8,9,16,17,23,26,28];
all_ids = [customer_ids_residential,customer_ids_commercial,customer_ids_industrial];
Vbase=11; %%---Base Voltage in kV---%%
Sbase=10; %%---Base Power in MVA---%%
Zbase=Vbase*Vbase/Sbase; %%---Base impedance in ohms---%%

%% ----Line Data (or Branch data) of the system----- %%

%%--- Line   From    To     R          X     Line Charging  Tap    Angle----%% 
%%---Number  Bus    Bus (in ohms)  (in ohms)  (in ohms)    Ratio   Shift----%%
LD = [ 1      1      2   0.0922      0.0470       0         1.0     0
       2      2      3   0.4930      0.2511       0         1.0     0
       3      3      4   0.3660      0.1864       0         1.0     0
       4      4      5   0.3811      0.1941       0         1.0     0
       5      5      6   0.8190      0.7070       0         1.0     0
       6      6      7   0.1872      0.6188       0         1.0     0
       7      7      8   0.7114      0.2351       0         1.0     0
       8      8      9   1.0300      0.7400       0         1.0     0
       9      9      10  1.0440      0.7400       0         1.0     0
       10     10     11  0.1966      0.0650       0         1.0     0
       11     11     12  0.3744      0.1238       0         1.0     0
       12     12     13  1.4680      1.1550       0         1.0     0
       13     13     14  0.5416      0.7129       0         1.0     0
       14     14     15  0.5910      0.5260       0         1.0     0
       15     15     16  0.7463      0.5450       0         1.0     0
       16     16     17  1.2890      1.7210       0         1.0     0
       17     17     18  0.7320      0.5740       0         1.0     0
       18     2      19  0.1640      0.1565       0         1.0     0
       19     19     20  1.5042      1.3554       0         1.0     0
       20     20     21  0.4095      0.4784       0         1.0     0
       21     21     22  0.7089      0.9373       0         1.0     0
       22     3      23  0.4512      0.3083       0         1.0     0
       23     23     24  0.8980      0.7091       0         1.0     0
       24     24     25  0.8960      0.7011       0         1.0     0
       25     6      26  0.2030      0.1034       0         1.0     0
       26     26     27  0.2842      0.1447       0         1.0     0
       27     27     28  1.0590      0.9337       0         1.0     0
       28     28     29  0.8042      0.7006       0         1.0     0
       29     29     30  0.5075      0.2585       0         1.0     0
       30     30     31  0.9744      0.9630       0         1.0     0
       31     31     32  0.3105      0.3619       0         1.0     0
       32     32     33  0.3410      0.5302       0         1.0     0];

%% ----Tie Line Data of the system---- %%

%%--- Line   From    To     R          X    
%%---Number  Bus    Bus (in ohms)  (in ohms)
TL = [ 33      8      21  2.0000      2.0000 0 1.0 0
       34      9      15  2.0000      2.0000 0 1.0 0
       35      12     22  2.0000      2.0000 0 1.0 0
       36      18     33  0.5000      0.5000 0 1.0 0
       37      25     29  0.5000      0.5000 0 1.0 0];  

%% ----Bus Data of the system---- %%

%%--- Type 1 - PQ Bus ; Type 2 - PV Bus; Type 3 - Swing Bus

%%----Bus    Bus     V     theta     Pg     Qg       Pl      Ql     Qgmax    Qgmin----%% 
%%---Number  Type (in pu) (in rad) (in kW)(in kVAR)(in kW)(in kVAR)(in pu)  (in pu)----%%
BD = [ 1      3     1.0      0       0      0        0       0       0        0        2
       2      1     1.0      0       0      0        100     60      0        0        1
       3      1     1.0      0       0      0        90      40      0        0        2
       4      1     1.0      0       0      0        120     80      0        0        3
       5      1     1.0      0       wt      0        60      30      0        0        1
       6      1     1.0      0       0      0        60      20      0        0        2
       7      1     1.0      0       0      0        200     100     0        0        3
       8      1     1.0      0       0      0        200     100     0        0        3
       9      1     1.0      0       0      0        60      20      0        0        2
       10     1     1.0      0       0      0        60      20      0        0        3
       11     1     1.0      0       0      0        45      30      0        0        3
       12     1     1.0      0       0      0        60      35      0        0        1
       13     1     1.0      0       0      0        60      35      0        0        3
       14     1     1.0      0       pv      0        120     80      0        0        1
       15     1     1.0      0       0      0        60      10      0        0        3
       16     1     1.0      0       0      0        60      20      0        0        2
       17     1     1.0      0       0      0        60      20      0        0        3
       18     1     1.0      0       0      0        90      40      0        0        2
       19     1     1.0      0       0      0        90      40      0        0        1
       20     1     1.0      0       0      0        90      40      0        0        3
       21     1     1.0      0       0      0        90      40      0        0        2
       22     1     1.0      0       0      0        90      40      0        0        1
       23     1     1.0      0       0      0        90      50      0        0        3
       24     1     1.0      0       0      0        420     200     0        0        3
       25     1     1.0      0       0      0        420     200     0        0        3
       26     1     1.0      0       0      0        60      25      0        0        3
       27     1     1.0      0       0      0        60      25      0        0        2
       28     1     1.0      0       0      0        60      20      0        0        3
       29     1     1.0      0       0      0        120     70      0        0        3
       30     1     1.0      0       0      0        200     600     0        0        3
       31     1     1.0      0       0      0        150     70      0        0        1
       32     1     1.0      0       0      0        210     100     0        0        1
       33     1     1.0      0       0      0        60      40      0        0        3];
   
   


BD(customer_ids_residential,7:8) = BD(customer_ids_residential,7:8)*resi; %apply load percent
BD(customer_ids_commercial,7:8) = BD(customer_ids_commercial,7:8)*comm; %apply load percent
BD(customer_ids_industrial,7:8) = BD(customer_ids_industrial,7:8)*indu; %apply load percent

batteryRows = [21, 26, 10, 23];
for i = 1:numel(batteryRows)
    row = batteryRows(i);
    batValue = bat_powers(i);

    if batValue < 0
        % Negative => battery is discharging => treat as generation
        BD(row, 5) = abs(batValue);     % set Pg column to positive
    elseif batValue > 0
        % Positive => battery is charging => treat as load
        BD(row, 7) = BD(row, 7) + batValue;  % add to Pl column
    end
end
BD(all_ids,7) = BD(all_ids,7) - curtailed; %apply curtailment
% BD(1,5) = sum(BD(:,7))-wt-pv ; %calculate grid transfer
% BD(1,6)=  BD(1,5)*tan(acos(0.85));
%% disater removing line
numToRemove = 3;
rowsToRemove = zeros(numToRemove,1);
if disaster ~= 0
    if training == 0
        rng(disaster,'twister');  % same 'disaster' -> same random combination
    end
    
    possibleLines = [24,9,6,21,32]; %taken from EC paper %no 13, 
    %possibleLines = [22,25,8,14,18,3,8,31];
    rowsToRemove = randsample(possibleLines, numToRemove, false);
    LD(rowsToRemove, :) = [];
    LD(:,1) = (1:size(LD, 1))';
    TL(:,1) = (size(LD,1)+1:size(LD,1)+size(TL,1));
end 
[LD,TL] = tieline_finder(BD,LD,TL); %this ensures relatively good connection 

BD(:,5:8)=BD(:,5:8)/(1000*Sbase); %%---Conversion of P and Q in pu quantity---%%
LD(:,4:5)=LD(:,4:5)/Zbase; %%----Conversion of R and X in pu quantity---%%
TL(:,4:5)=TL(:,4:5)/Zbase; %%----Conversion of R and X in pu quantity---%%