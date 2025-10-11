
function [LD,x]=reconfiguration_func(resi,comm,indu,pv,wt,curtailed,training)
%%% outputs line data (with tie switches) and sectionalising switches
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
    
    CPKW_before_curtailment = BD(all_ids,7); %output for state before scaling
    totalload_before_action = sum(BD(:,7));
    BD(all_ids,7) = BD(all_ids,7) - curtailed; %apply curtailment
    BD(1,5) = sum(BD(:,7))-wt-pv; %calculate grid transfer
    BD(1,6)=  BD(1,5)*tan(acos(0.85));
    
    BD(:,5:8)=BD(:,5:8)/(1000*Sbase); %%---Conversion of P and Q in pu quantity---%%
    LD(:,4:5)=LD(:,4:5)/Zbase; %%----Conversion of R and X in pu quantity---%%
    TL(:,4:5)=TL(:,4:5)/Zbase; %%----Conversion of R and X in pu quantity---%%


    
    %% ---Other informations--- %%
    
    nbr=size(LD,1);
    ntl=size(TL,1);
    nbus=size(BD,1);
    
    itermax=20;
    iter=1;
    Loss=zeros(itermax,1);
    
    ref=find(BD(:,2)==3,1);
    
    [Yb,~] = Y_busCaseI(LD,nbr,nbus);
    [Vmag, ~, Pcalc, ~]= NR_method(BD,Yb,nbus);
    
    Vmin=min(Vmag);
    Vmax=max(Vmag);
    
    Loss(iter)=sum(Pcalc);
    iter=iter+1;
    
    x=sort(TL(:,1));
    %fprintf('Tie switch in initial Configuration: %s \n',num2str(x'))
    %fprintf('The total active power loss in the initial configuration of %d bus system is %f kW. \n\n', nbus, Loss(1)*Sbase*1000)
    
    %tstart=tic;
    
    %% ---List of switches--- %%
    
    [LD]=Rearrange(LD,ref);
    [bibc, ~]=BIBC(LD,nbus,ref);
    
    n_1=3;
    n_2=2;
    
    sw=zeros(nbr+ntl,1);
    sw(TL(:,1))=sw(TL(:,1))+ones(ntl,1);
    
    %%---Exclusion of Type-3 switch---%%
    for i=1:ntl
        sw(LD(:,1))=sw(LD(:,1))+abs(bibc(:,TL(i,2))-bibc(:,TL(i,3)));
    end
    tempsw=sw;
    
    %% ---Sequential Switch Opening using minimum current method--- %%
    
    LD=[LD;TL];
    
    t=0;
    
    [Yb, A] = Y_busCaseI(LD,size(LD,1),nbus);
    [Vmag, theta, Pcalc, ~]= NR_method(BD,Yb,nbus);
    
    Vbus=Vmag.*(cos(theta)+1i.*sin(theta));
    Vbr=Vbus(LD(:,2))-Vbus(LD(:,3));
    Z=LD(:,4)+1i.*LD(:,5);
    Ibr=Vbr./Z;
    
    [~,index1]=sort((abs(Ibr)),'ascend');
    index1=LD(index1,1);
    
    t1=1;
    
    while t<ntl
        if sw(index1(t1))>0
            index=find(LD(:,1)==index1(t1));
            tempA=A;
            tempA(index,:)=[];
    
            if rank(tempA)==nbus-1
                t=t+1;
                TL(t,:)=LD(index,:);
                LD(index,:)=[];
    
                A=tempA;
                t1=t1+1;
            else
                t1=t1+1;
            end
        else
            t1=t1+1;
        end
    
    end
    
    %% ---Branch exchange method using BIBC matrix--- %%
    
    nbr=size(LD,1);
    [LD]=Rearrange(LD,ref);
    [bibc, bcbv]=BIBC(LD,nbus,ref);
    
    %%---Exclusion of Type-1 switch---%%
    temp=sum(bibc,1);
    for i=1:nbus
        if temp(i)<n_1
            for j=1:nbr
                if bibc(j,i)==1
                    sw(LD(j,1))=0;
                end
            end
        end
    end
    
    %%---Exclusion of Type-2 switch---%%
    for i=1:nbr
        x=find(LD(:,2)==LD(i,3));
        if size(x,1)>n_2
            sw(LD(i,1))=0;
        end
    end
    
    Vbus=BD(:,3).*(cos(BD(:,4))+1i.*sin(BD(:,4)));
    
    Sinj=(BD(:,5)-BD(:,7))+1i*(BD(:,6)-BD(:,8));
    
    [Vbus,Iinj]= dlf(ref,nbus,Sinj,Vbus,bibc,bcbv);
    Ibr=-bibc*Iinj;
    Vmag=abs(Vbus);
    
    Loss(iter)=sum((abs(Ibr).^2).*LD(:,4));
    
    iter=iter+1;
    maxdPloss=inf;
    iter1=0;
    
    while iter1<1
    
        %%---Selection of switch pair for the exchange operation---%%
    
        maxdPloss=-inf;
    
        for k1=1:ntl
    
            m1=TL(k1,2);
            n1=TL(k1,3);
            Rk1=TL(k1,4);
    
            loop=abs(bibc(:,m1)-bibc(:,n1));
    
            for k2=1:nbr
                if loop(k2)==1 && sw(LD(k2,1))>0
    
                    Ibrnew=Ibr+loop.*(1-2.*(bibc(:,m1).*bibc(k2,m1)+bibc(:,n1).*bibc(k2,n1))).*Ibr(k2);
    
                    dPloss=-sum((abs(Ibrnew).^2).*LD(:,4))+sum((abs(Ibr).^2).*LD(:,4))-abs(Ibr(k2)^2)*Rk1;
    
                    if maxdPloss<dPloss
                        maxdPloss=dPloss;
                        index2=k2;
                        index1=k1;
                    end
                end
    
            end
        end
    
        %%---Updating the configuration---%%
    
        tempLD=[LD; TL(index1,:)];
        temp=tempLD(index2,:);
        tempLD(index2,:)=[];
    
        [tempLD]=Rearrange(tempLD,ref);
        [tempbibc, tempbcbv]=BIBC(tempLD,nbus,ref);
    
        [tempVbus,Iinj]= dlf(ref,nbus,Sinj,Vbus,tempbibc,tempbcbv);
    
        tempIbr=-tempbibc*Iinj;
        temploss=sum((abs(tempIbr).^2).*tempLD(:,4));
        tempVmag=abs(Vbus);
    
        if temploss<Loss(iter-1) && min(tempVmag)>=Vmin && max(tempVmag)<=Vmax
    
            LD=tempLD;
    
            TL(index1,:)=[];
            TL(ntl,:)=temp;
    
            Vbus=tempVbus;
            bibc=tempbibc;
            Ibr=tempIbr;
    
            sw=tempsw;
    
            %%---Exclusion of Type-1 switch---%%
            temp=sum(bibc,1);
            for i=1:nbus
                if temp(i)<n_1
                    for j=1:nbr
                        if bibc(j,i)==1
                            sw(LD(j,1))=0;
                        end
                    end
                end
            end
    
            %%---Exclusion of Type-2 switch---%%
            for i=1:nbr
                x=find(LD(:,2)==LD(i,3));
                if size(x,1)>n_2
                    sw(LD(i,1))=0;
                end
            end
    
            Loss(iter)=temploss;
            iter=iter+1;
    
        else
            Loss(iter)=Loss(iter-1);
            iter=iter+1;
    
            temp=TL(index1,:);
            TL(index1,:)=[];
            TL(ntl,:)=temp;
            iter1=iter1+1;
    
        end
    
    end
    
    Loss(iter:end)=[];
    Loss=Loss.*Sbase*1000;
    
    % tstop=toc(tstart);
    
    %% ---RESULTS--- %%
    
     x=sort(TL(:,1));
    % if size(x,1) == 2
    % fprintf('Tie switch in final Configuration: %s \n',num2str(x'))
    % 
    % fprintf('The total active power loss in the final configuration of %d bus system is %f kW. \n\n', nbus, Loss(end))
    %fprintf('The total computational time is %f sec. \n\n',tstop)

end
    
    
    
    
    
    

