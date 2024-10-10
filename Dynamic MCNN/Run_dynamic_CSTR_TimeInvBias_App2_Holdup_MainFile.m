% NLS-NLD type of Series Static Dynamic Network (Simultaneous MCNN)
% Mass Constrained Neural Network for Dynamic Data
% Applied on isothermal Van de Vusse Reactor system (changing V)
% Mass Constraints applied on Dynamic Data assuming holdup info available

clc
clear

ni = 1; nh = ni; no = 4;

% Loading the data 
data_dyn_changeV = xlsread('Dynamic CSTR Data.xlsx','TimeInvBias+Noise+Holdup');

data = data_dyn_changeV;

V_ins = data_dyn_changeV(:,2);       % instantaneous volume

% Constant volumetric flow rate and inlet concentrations
F_in = data(1,3); 
CAf = data(1,4);
CBf = data(1,5);
CCf = data(1,6);
CDf = data(1,7);

data_dyn = data(:, [2,8:end]);

n = floor(size(data_dyn,1));

% Reaction Model: Cyclopentadiene (A, C5H6) reacts in water (liquid) medium
% to form Cyclopentenol (B, C5H8O), which again gives Cyclopentanediol (C, 
% C5H10O2). Simultaneously, 2 moles of A combine to form Dicyclopentadiene
% (D, C10H12). C and D are undesirable products. Density changes in the
% reactor are assumed to be negligible.

% Number of C atoms in one mole of each reaction species

ncA = 5; ncB = 5; ncC = 5; ncD = 10;

% nC = [ncA;ncB;ncC;ncD];

% Number of H atoms in one mole of each reaction species

nhA = 6; nhB = 8; nhC = 10; nhD = 12; nhW = 2;

% Number of O atoms in one mole of each reaction species

noA = 0; noB = 1; noC = 2; noD = 0; noW = 1;

% nH = [nhA;nhB;nhC;nhD];

% Density of each reaction species (in Kg/m3)

rhoA = 786; rhoB = 949; rhoC = 1094; rhoD = 980; rhoW = 997;

% Molecular Weight of each reaction species 

MWA = 66; MWB = 90; MWC = 102; MWD = 132; MWW = 18;


tt = size(data_dyn,1);              % Total time steps
tn = floor(0.6*tt);                 % Number of training time steps

nt = size(data_dyn,2);

norm_mat = zeros(tt,nt);
delta = zeros(1,nt);
for i = 1:nt
    delta(1,i) = (max(data_dyn(:,i)) - min(data_dyn(:,i)));
    norm_mat(:,i) = (data_dyn(:,i)-min(data_dyn(:,i)))/(delta(1,i));
end

dsr = (norm_mat(:,ni+1:ni+no))';
Imat = (norm_mat(:,1:ni))';

tr_steps = 1:tn;
tr_steps = sort(tr_steps);
tr_steps = tr_steps';

dsr_t = dsr(:,tr_steps); Imat_t = Imat(:,tr_steps); 
V_ins_t = V_ins(tr_steps,:); 

% Calculation of Inlet Atoms (C Atoms)
% Note that for steady-state using constant volume, all atom balance
% constraints for this system get simplified to the same form as C balance,
% i.e., CA + CB + CC + 2CD = CAf + CBf + CCf + CDf

% Calculation of number of moles of A,B,C,D in Inlet Stream (kmol/min)
mol_A_in = F_in.*CAf.*ones(tt,1); mol_B_in = F_in.*CBf.*ones(tt,1);
mol_C_in = F_in.*CCf.*ones(tt,1); mol_D_in = F_in.*CDf.*ones(tt,1);

at_C_in = mol_A_in*ncA + mol_B_in*ncB + mol_C_in*ncC + mol_D_in*ncD;
at_C_in_t = at_C_in(tr_steps,:);

%% Training of Unconstrained NLS-NLD Model

[ynn_t,nn_stat,nn_dyn] = SolveNLSNLD(Imat_t,dsr_t,ni,nh,no,tn);

dsr_t_p = zeros(tn,no);
ynn_wo_t_p = zeros(tn,no);

for i = 1:no
    dsr_t_p(:,i) = dsr_t(i,:)'.*delta(1,ni+i) + min(data_dyn(:,ni+i));
    ynn_wo_t_p(:,i) = ynn_t(:,i).*delta(1,ni+i) + min(data_dyn(:,ni+i));
end

% dV/dt = F_in - F_out
F_out_t = zeros(tn,1);
F_out_t(1,1) = F_in;
for i = 2:tn
    F_out_t(i,1) = F_in - (V_ins(i,1)-V_ins(i-1,1));
end

CA_NN_out_wo = ynn_wo_t_p(:,1); CB_NN_out_wo = ynn_wo_t_p(:,2);
CC_NN_out_wo = ynn_wo_t_p(:,3); CD_NN_out_wo = ynn_wo_t_p(:,4);

% Calculation of number of moles of A,B,C,D in Outlet Stream (kmol/min)
mol_A_out_wo = F_out_t.*CA_NN_out_wo; mol_B_out_wo = F_out_t.*CB_NN_out_wo;
mol_C_out_wo = F_out_t.*CC_NN_out_wo; mol_D_out_wo = F_out_t.*CD_NN_out_wo;

at_C_out_t_wo = mol_A_out_wo*ncA + mol_B_out_wo*ncB + mol_C_out_wo*ncC + mol_D_out_wo*ncD;

diff_wo = 100.*abs((at_C_in_t-at_C_out_t_wo)./at_C_in_t);

whs = (nn_stat.IW{1})'; wos = (nn_stat.LW{2,1})'; bhs = nn_stat.b{1}; bos = nn_stat.b{2};
whd = (nn_dyn.IW{1})'; wfd = (nn_dyn.LW{1,2})'; wod = (nn_dyn.LW{2,1})'; bhd = nn_dyn.b{1}; bod = nn_dyn.b{2};

% ---------------------------------------------------------------------- %

% Mass Constrained Network using NLS-NLD type Series Hybrid Structure

ynn_t = zeros(tn,no);
int_mat_1 = zeros(nh,tn);
check1 = Inf;
x_in = Imat_t;

net_s = newff([0 1].*ones(ni,1),[nh, nh],{'logsig','logsig'},'trainlm');
net_s.IW{1} = whs'; net_s.LW{2,1} = wos';
net_s.b{1} = bhs; net_s.b{2} = bos;

p = x_in;
    
wd0 = [reshape(whd,[1 nh*nh]),reshape(wod,[1 nh*no]),reshape(wfd,[1 no*nh]),bhd',bod',reshape(ynn_wo_t_p,[1 tn*no])];

lb = [-1e5.*ones(1,size(wd0,2)-(no*tn))';zeros(no*tn,1)];
ub = 1e5.*ones(1,size(wd0,2))';

obj = @(x)SolveDynMCNN(x,Imat_t,dsr_t,ni,nh,no,tn,delta,data_dyn);
nlcon = @(x)SolveDynMCNNCons(x,Imat_t,dsr_t,tn,ni,nh,no,at_C_in_t,F_out_t,V_ins_t,delta,data_dyn);
nlrhs = zeros((2*no*tn)+(tn-1),1);
nle = [ones(2*no*tn,1);zeros(tn-1,1)];

opts = optiset('solver','ipopt','display','iter','maxiter',1e3,'maxtime',20000); 
Opt = opti('fun',obj,'ineq',[],[],'nlmix',nlcon,nlrhs,nle,'bounds',lb,ub,'options',opts);

[w_sol,fval,exitflag,info] = solve(Opt,wd0);

whd = reshape(w_sol(1:ni*nh),[ni,nh]);
wod = reshape(w_sol(ni*nh+1:ni*nh+nh*no),[nh,no]);
wfd = reshape(w_sol(ni*nh+nh*no+1:ni*nh+nh*no+no*nh),[no,nh]);
bhd = (w_sol(ni*nh+nh*no+no*nh+1:ni*nh+nh*no+no*nh+nh));
bod = (w_sol(ni*nh+nh*no+no*nh+nh+1:ni*nh+nh*no+no*nh+nh+no));
y_r = reshape(w_sol(ni*nh+nh*no+no*nh+nh+no+1:ni*nh+nh*no+no*nh+nh+no+tn*no),[tn,no]);

ynn_t(1,:) = dsr_t(:,1);

for i = 2:tn
    ynn_t(i,:) = purelin(wod'*tansig(whd'*Imat_t(:,i) + wfd'*(ynn_t(i-1,:))' + bhd) + bod);
end
             
for i = 1:nh
    int_mat_1(i,:) = x_in(i,:);
end

% Training of Static Network

net_s.trainParam.epochs = 5000;
inp = Imat_t;

[net_s,tr] = train(net_s,inp,int_mat_1);

ynn_final = y_r;

CA_NN_out = ynn_final(:,1); CB_NN_out = ynn_final(:,2);
CC_NN_out = ynn_final(:,3); CD_NN_out = ynn_final(:,4);

% Calculation of number of moles of A,B,C,D in Outlet Stream (kmol/min)
mol_A_out = F_out_t.*CA_NN_out; mol_B_out = F_out_t.*CB_NN_out;
mol_C_out = F_out_t.*CC_NN_out; mol_D_out = F_out_t.*CD_NN_out;

at_C_out_t = mol_A_out*ncA + mol_B_out*ncB + mol_C_out*ncC + mol_D_out*ncD;

diff = zeros(tn-1,1);
for i = 2:tn
    diff(i-1,1) = (V_ins_t(i,1)*(CA_NN_out(i,1)*ncA + CB_NN_out(i,1)*ncB + CC_NN_out(i,1)*ncC + CD_NN_out(i,1)*ncD)) - ...
        (V_ins_t(i-1,1)*(CA_NN_out(i-1,1)*ncA + CB_NN_out(i-1,1)*ncB + CC_NN_out(i-1,1)*ncC + CD_NN_out(i-1,1)*ncD)) - ...
        (at_C_in_t(i,1)-at_C_out_t(i,1));
end

diff_mcnn = 100.*abs(diff./at_C_in_t(2:end,:));




















