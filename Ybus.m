function [Yb] = Ybus(LD,nbr,nbus) 

Ysh=zeros(nbus,1);

for i=1:nbus
    for j=1:nbr
        if LD(j,2)==i || LD(j,3)==i
            Ysh(i)=Ysh(i)+1i*LD(j,6)/2;
        end
    end
end
ish=find(Ysh); %%---Bus at which shunt branches are connected
nsh=size(find(Ysh),1); %%---Number of shunt branches
Ypr=zeros(nbr+nsh,nbr+nsh); %%---Initialization of Primitive Ybus
A=zeros(nbr+nsh,nbus); %%---Initialization of Incidence Matrix

%% ---Formnation Of Primitive Admittance Matrix and Incidence Matrix--- %%

for k=1:nbr
    
    if LD(k,7)==1.0 && LD(k,8)==0
        Ypr(k,k)=1/(LD(k,4)+1i*LD(k,5));
        
        i=LD(k,2);
        j=LD(k,3);
        A(k,i)=1;
        A(k,j)=-1;
        
    end

end

if nsh>0
    for i=1:nsh
        Ypr(nbr+i,nbr+i)=Ysh(ish(i));
        A(nbr+i,ish(i))=1;
    end    
end

%% ----Calculation of Admittance Matrix---- %%

Yb=A'*Ypr*A;

%% ----Modification of calculated Ybus matrix by including the given tap changing transformer data---- %%

for k=1:nbr
    if LD(k,7)~=1.0
        y=1/(LD(k,4)+1i*LD(k,5)); %%---Leakage Admittance of the Transformer
        t=LD(k,7)+1i*LD(k,8); %%---Off Nominal Tap Ratio
        i=LD(k,2);
        j=LD(k,3);
        Yb(i,i)=Yb(i,i)+((abs(t))^2)*y;
        Yb(i,j)=Yb(i,j)-(conj(t))*y;
        Yb(j,i)=Yb(j,i)-t*y;
        Yb(j,j)=Yb(j,j)+y;
    end
end