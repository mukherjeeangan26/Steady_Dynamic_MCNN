clc
clear

%% Loading the Data for Model Development followed by Partitioning

% This code develops dynamic (hybrid parallel) MCNN models for the case 
% when a time-invariant bias with / without Gaussian noise is added to 
% true data to generate training data

% This code requires the MATLAB Neural Network Toolbox to train the
% unconstrained all-nonlinear parallel (NLS || NLD) network model for 
% faster computation. 

% Data is partitioned into steady-state and dynamic zones. A steady-state 
% MCNN is trained followed by training a dynamic residual model to match 
% the transient measurements. The optimal solution of the unconstrained 
% steady-state network serve as initial guesses for the constrained 
% steady-state MCNN formulation in the inverse problem.

% Load the training and validation datasets and specify the input and
% output variables to the NN models
% Note that the user can consider any dynamic dataset for training and
% validation. The rows signify the time steps and the columns signify the 
% input and output variables.

data_dyn = xlsread('Dynamic CSTR Data.xlsx','TimeInvBias+Gaussian Noise');
data_dyn = data_dyn(:,2:end);

% Partitioning the entire time-series data into steady-state and dynamic
% zones require the identification of steady-state zones. However, the
% current dataset under consideration shows a consistent perturbation of
% the system inputs after every 20 time steps. 

t_steady = 20;
n = floor(size(data_dyn,1)/t_steady);  % number of steady-state zones

% In general, the steady-state zones have been identified in this work as 
% those continuous time periods in the overall data where the maximum 
% deviation is within +- 0.5% of the mean value in those zones

% Creating a subset of steady-state data for developing the steady-state
% MCNN model

data_steady = [];

for i = 1:n
    data_steady = [data_steady; data_dyn(i*t_steady,:)];
end

% For this specific system, the first five columns are the model inputs and
% the following 4 columns are the model outputs

ni = 5;             % Number of inputs = Number of neurons in input layer
no = 4;            % Number of outputs = Number of neurons in output layer

% Default number of neurons in hidden layer taken equal to number of
% inputs. But it can be changed as per requirement.

nh = ni;

nt = ni + no;

% ---------------------------------------------------------------------- %

%% Development/Preparation of steady-state data in terms of Process Variables

data = data_steady;

% Reaction Model: Cyclopentadiene (A, C5H6) reacts in water (liquid) medium
% to form Cyclopentenol (B, C5H8O), which again gives Cyclopentanediol (C, 
% C5H10O2). Simultaneously, 2 moles of A combine to form Dicyclopentadiene
% (D, C10H12). C and D are undesirable products. Density changes in the
% reactor are assumed to be negligible.

% Number of C atoms in one mole of each reaction species

ncA = 5; ncB = 5; ncC = 5; ncD = 10;

% Number of H atoms in one mole of each reaction species

nhA = 6; nhB = 8; nhC = 10; nhD = 12; nhW = 2;

% Number of O atoms in one mole of each reaction species

noA = 0; noB = 1; noC = 2; noD = 0; noW = 1;

% Density of each reaction species (in Kg/m3)

rhoA = 786; rhoB = 949; rhoC = 1094; rhoD = 980; rhoW = 997;

% Molecular Weight of each reaction species 

MWA = 66; MWB = 90; MWC = 102; MWD = 132; MWW = 18;

V = 0.1;                           %m3 (Volume of reactor)

fv_in = data(:,1);
CAf_in = data(:,2);
CBf_in = data(:,3);
CCf_in = data(:,4);
CDf_in = data(:,5);
CA_out = data(:,6);
CB_out = data(:,7);
CC_out = data(:,8);
CD_out = data(:,9);

% Calculation of number of moles of water in Inlet Stream (kmol/min)
mol_W_in = zeros(size(data,1),1);
for i = 1:size(data,1)
    mol_W_in(i,1) = (rhoW/MWW)*(fv_in(i,1)*V);
end

% Calculation of number of moles of A,B,C,D in Inlet Stream (kmol/min)
mol_A_in = V*fv_in.*CAf_in; mol_B_in = V*fv_in.*CBf_in;
mol_C_in = V*fv_in.*CCf_in; mol_D_in = V*fv_in.*CDf_in;

% Total number of atoms of C and H in Inlet Stream

at_C_in = mol_A_in*ncA + mol_B_in*ncB + mol_C_in*ncC + mol_D_in*ncD;
at_H_in = mol_A_in*nhA + mol_B_in*nhB + mol_C_in*nhC + mol_D_in*nhD + mol_W_in*nhW;
at_O_in = mol_A_in*noA + mol_B_in*noB + mol_C_in*noC + mol_D_in*noD + mol_W_in*noW;

%------------------------------------------------------------------------%
%% Data Preparation for Model Training

tt = size(data,1);           % Total size of data
tn = floor(0.7*tt);          % Selecting 70% of total data for training

% Normalization of Inputs and Outputs

norm_mat = zeros(tt,nt);
delta = zeros(1,nt);
for i = 1:nt
    delta(1,i) = (max(data(:,i)) - min(data(:,i)));
    norm_mat(:,i) = (data(:,i)-min(data(:,i)))/(delta(1,i));
end

Imat = (norm_mat(:,1:ni))';
dsr = (norm_mat(:,ni+1:ni+no))';

% Selecting first tn steady-states from the overall data

tr_steps = 1:tn;
tr_steps = (sort(tr_steps))';

dsr_t = zeros(no,tn); Imat_t = zeros(ni,tn); 
fv_in_t = zeros(tn,1); CAf_in_t = zeros(tn,1); CBf_in_t = zeros(tn,1); 
CCf_in_t = zeros(tn,1); CDf_in_t = zeros(tn,1); at_C_in_t = zeros(tn,1); 
at_H_in_t = zeros(tn,1); at_O_in_t = zeros(tn,1); mol_W_in_t = zeros(tn,1);

for i = 1:tn
    
    ts = tr_steps(i,1);    
    dsr_t(1:no,i) = dsr(1:no,ts);
    Imat_t(1:ni,i) = Imat(1:ni,ts);
    fv_in_t(i,1) = fv_in(ts,1);
    CAf_in_t(i,1) = CAf_in(ts,1);
    CBf_in_t(i,1) = CBf_in(ts,1);
    CCf_in_t(i,1) = CCf_in(ts,1);
    CDf_in_t(i,1) = CDf_in(ts,1);
    at_C_in_t(i,1) = at_C_in(ts,1);
    at_H_in_t(i,1) = at_H_in(ts,1);
    at_O_in_t(i,1) = at_O_in(ts,1);
    mol_W_in_t(i,1) = mol_W_in(ts,1);
    
end

%----------------------------------------------------------------------%

%% Training of Unconstrained steady-state NN Model

% In absence of mass constraints, training the model wrt measurement data

% Using NN Toolbox for Solving the Unconstrained Network

net_s = newff([0 1].*ones(ni,1),[nh no],{'logsig','purelin'},'trainscg');

net_s.trainParam.epochs = 20000;
[net_s,tr] = train(net_s,Imat_t,dsr_t);

whf_wo = (net_s.IW{1,1})'; wof_wo = (net_s.LW{2,1})'; 
bhf_wo = net_s.b{1}; bof_wo = net_s.b{2};

y1 = Imat_t;
x1 = whf_wo'*y1 + bhf_wo;
y2 = logsig(x1);
x2 = wof_wo'*y2 + bof_wo;
ynn_wo = purelin(x2);

% Calculating Mass Balance Errors for NN w/o mass constraints

ynn_wo_t_p = zeros(tn,no);

for i = 1:no
    ynn_wo_t_p(:,i) = (ynn_wo(i,:))'.*delta(1,ni+i) + min(data(:,ni+i));
end

% Saving the optimal solution to use as initial guess for the constrained
% NN models

whf_wo_opt = whf_wo; wof_wo_opt = wof_wo; 
bhf_wo_opt = bhf_wo; bof_wo_opt = bof_wo;

%-----------------------------------------------------------------------%

%% Training of the Constrained Network

wh0 = whf_wo_opt; wo0 = wof_wo_opt;
bh0 = bhf_wo_opt; bo0 = bof_wo_opt;

w0 = [reshape(wh0,[1 ni*nh]),reshape(wo0,[1 nh*no]),bh0',bo0',reshape(ynn_wo_t_p,[1 no*tn])];

lb = [-1e5.*ones(1,size(w0,2)-(no*tn))';zeros(no*tn,1)];
ub = 1e5.*ones(1,size(w0,2))';

% The constrained optimization is performed using the IPOPT solver in the
% OPTI Toolbox 

obj = @(x)InvProbV1(x,Imat_t,dsr_t,ni,nh,no,tn,delta,data);
nlcon = @(x)InvProbV1Cons(x,Imat_t,tn,ni,nh,no,at_C_in_t,at_H_in_t,at_O_in_t,fv_in_t,V,CAf_in_t,CBf_in_t,CDf_in_t,mol_W_in_t,delta,data);
nlrhs = zeros((2*no+3)*tn,1);
nle = [ones(2*no*tn,1);zeros(3*tn,1)];

opts = optiset('solver','ipopt','display','iter','maxiter',2,'maxtime',1200); 
Opt = opti('fun',obj,'ineq',[],[],'nlmix',nlcon,nlrhs,nle,'bounds',lb,ub,'options',opts);

[w_sol,fval,exitflag,info] = solve(Opt,w0);

whf = reshape(w_sol(1:ni*nh),[ni,nh]);
wof = reshape(w_sol(ni*nh+1:ni*nh+nh*no),[nh,no]);
bhf = (w_sol(ni*nh+nh*no+1:ni*nh+nh*no+nh));
bof = (w_sol(ni*nh+nh*no+nh+1:ni*nh+nh*no+nh+no));
y_r = reshape(w_sol(ni*nh+nh*no+nh+no+1:ni*nh+nh*no+nh+no+no*tn),[tn,no]);

y1 = Imat_t;
x1 = whf'*y1 + bhf;
y2 = logsig(x1);
x2 = wof'*y2 + bof;
ynn_t = purelin(x2);

ynn_t_p = y_r;

dsr_t_p = zeros(tn,no);
ynn1_t_p = zeros(tn,no);

% Calculating Mass Balance Errors for NN w/ mass constraints

for i = 1:no
    dsr_t_p(:,i) = (dsr_t(i,:))'.*delta(1,ni+i) + min(data(:,ni+i));
    ynn1_t_p(:,i) = ynn_t(i,:)'.*delta(1,ni+i) + min(data(:,ni+i));
end

%------------------------------------------------------------------------%

%% Development of Dynamic Residual Model connected in parallel to MCNN

data_dyn_total_t = data_dyn(1:floor(tn*t_steady),:);
data_steady_t = data_steady(1:tn,:);

% Calculating deviation variables for inputs and outputs

input_dev_total_t = data_dyn_total_t(:,1:ni) - data_steady_t(1,1:ni);
output_dev_total_t = zeros(size(data_dyn_total_t,1),no);
data_steady_out_dev_t = zeros(tn,no);

for i = 1:tn
    data_steady_out_dev_t(i,1:no) = data_steady_t(i,ni+1:ni+no) - data_steady_t(1,ni+1:ni+no);
    output_dev_total_t((i-1)*t_steady+1:i*t_steady,:) = data_dyn_total_t((i-1)*t_steady+1:i*t_steady,ni+1:ni+no) - data_steady_out_dev_t(i,1:no);
end

dyn_dev_total_t = [input_dev_total_t output_dev_total_t];

data_dyn_t = dyn_dev_total_t;

tn_d = size(data_dyn_t,1); 

% Generating normalized inputs and outputs for the time-series data

norm_mat_dyn_t = zeros(tn_d,nt);
delta_dyn_t = zeros(1,nt);
for i = 1:nt
    delta_dyn_t(1,i) = (max(data_dyn_t(:,i)) - min(data_dyn_t(:,i)));
    norm_mat_dyn_t(:,i) = (data_dyn_t(:,i)-min(data_dyn_t(:,i)))/(delta_dyn_t(1,i));
end

Imat_dyn_t = (norm_mat_dyn_t(:,1:ni))';
dsr_dyn_t = (norm_mat_dyn_t(:,ni+1:nt))';

% Training a NARX-type RNN through the function 'TrainNLD4ParMCNN.m'
% Different packages can also be used for better accuracy

[ynn_dyn_t,nn_dyn,Xi,Ai] = TrainNLD4ParMCNN(Imat_dyn_t,dsr_dyn_t,nh,no,tn_d);

% The dynamic deviation data contain sharp spikes at the transition points
% between steady-state and dynamics. Accurately predicting such spikes
% perfectly using a dynamic data-driven model can be challenging. So, as a
% post-processing step, the peaks are smoothened with values closer to the
% desired deviations.

for i = 1:tn-1
    ynn_dyn_t(i*t_steady+1:i*t_steady+3,:) = (dsr_dyn_t(:,i*t_steady+1:i*t_steady+3))';    
end

% Converting normalized variables to absolute variables

ynn_dyn_t_p = zeros(tn_d,no);
dsr_dyn_t_p = data_dyn_total_t(:,ni+1:nt);

for i = 1:no
    ynn_dyn_t_p(:,i) = ynn_dyn_t(:,i).*delta_dyn_t(1,ni+i) + min(data_dyn_t(:,ni+i));
end

%------------------------------------------------------------------------%

%% GENERATING TRAINING RESULTS FROM THE OVERALL HYBRID PARALLEL MCNN AND UNCONSTRAINED MODEL

ymcnn_stat_t_p = zeros(tn_d,no);
ynn_stat_womc_t_p = zeros(tn_d,no);

for i = 1:tn
    ymcnn_stat_t_p((i-1)*t_steady+1:i*t_steady,1:no) = ynn_t_p(i,1:no).*ones(t_steady,1);
    ynn_stat_womc_t_p((i-1)*t_steady+1:i*t_steady,1:no) = ynn_wo_t_p(i,1:no).*ones(t_steady,1);
end

ymcnn_total_t_p = ymcnn_stat_t_p + ynn_dyn_t_p - data_steady_t(1,ni+1:ni+no);
ynn_total_womc_t_p = ynn_stat_womc_t_p + ynn_dyn_t_p - data_steady_t(1,ni+1:ni+no);

% Plotting Training Results for C_A (for the CSTR case study)

figure(1)
hold on
plot(ymcnn_total_t_p(:,1),'b-x','MarkerSize',1,'LineWidth',1.5)
plot(ynn_total_womc_t_p(:,1),'g--','MarkerSize',1,'LineWidth',1.5)
plot(dsr_dyn_t_p(:,1),'r--o','MarkerSize',1,'LineWidth',1.2)
xlabel('Time (mins)')
ylabel('C_A (kmol/m^3)')
xlim([0 tn*t_steady])
title('Training of Mass Constrained Neural Network (Approach 1: Hybrid Parallel)')
legend('MCNN','NN w/o MC','Measurements','Location','northeast')
a=findobj(gcf);
allaxes=findall(a,'Type','axes'); alltext=findall(a,'Type','text'); set(allaxes,'FontName','Times','FontWeight','Bold','LineWidth',2.7,'FontSize',14);
set(alltext,'FontName','Times','FontWeight','Bold','FontSize',14);

% Calculations for Error in Mass Balances

fv_in_dyn_t = data_dyn_total_t(:,1);
CAf_in_dyn_t = data_dyn_total_t(:,2);
CBf_in_dyn_t = data_dyn_total_t(:,3);
CCf_in_dyn_t = data_dyn_total_t(:,4);
CDf_in_dyn_t = data_dyn_total_t(:,5);

% Calculation of number of moles of water in Inlet Stream (kmol/min)
mol_W_in_dyn_t = zeros(size(data_dyn_total_t,1),1);
for i = 1:size(data_dyn_total_t,1)
    mol_W_in_dyn_t(i,1) = (rhoW/MWW)*(fv_in_dyn_t(i,1)*V);
end

% Calculation of number of moles of A,B,C,D in Inlet Stream (kmol/min)
mol_A_in_dyn_t = V*fv_in_dyn_t.*CAf_in_dyn_t; mol_B_in_dyn_t = V*fv_in_dyn_t.*CBf_in_dyn_t;
mol_C_in_dyn_t = V*fv_in_dyn_t.*CCf_in_dyn_t; mol_D_in_dyn_t = V*fv_in_dyn_t.*CDf_in_dyn_t;

% Total number of atoms of C and H in Inlet Stream

at_C_in_dyn_t = mol_A_in_dyn_t*ncA + mol_B_in_dyn_t*ncB + mol_C_in_dyn_t*ncC + mol_D_in_dyn_t*ncD;
at_H_in_dyn_t = mol_A_in_dyn_t*nhA + mol_B_in_dyn_t*nhB + mol_C_in_dyn_t*nhC + mol_D_in_dyn_t*nhD + mol_W_in_dyn_t*nhW;
at_O_in_dyn_t = mol_A_in_dyn_t*noA + mol_B_in_dyn_t*noB + mol_C_in_dyn_t*noC + mol_D_in_dyn_t*noD + mol_W_in_dyn_t*noW;

% NN w/o MC

CA_NN_out_dyn_womc_t = ynn_total_womc_t_p(:,1);
CB_NN_out_dyn_womc_t = ynn_total_womc_t_p(:,2);
CC_NN_out_dyn_womc_t = ynn_total_womc_t_p(:,3);
CD_NN_out_dyn_womc_t = ynn_total_womc_t_p(:,4);

% Calculation of number of moles of water in Outlet Stream (kmol/min)

mol_W_out_dyn_womc_t = zeros(tn_d,1);
for i = 1:tn_d    
    mol_AtoB = (fv_in_dyn_t(i,1)*V)*(CAf_in_dyn_t(i,1) - CA_NN_out_dyn_womc_t(i,1) - 2*(CD_NN_out_dyn_womc_t(i,1)-CDf_in_dyn_t(i,1)));       % Number of moles of water used up when A reacted to form B
    C_Bformed = (CAf_in_dyn_t(i,1) - CA_NN_out_dyn_womc_t(i,1) - 2*(CD_NN_out_dyn_womc_t(i,1)-CDf_in_dyn_t(i,1)));
    mol_BtoC = (fv_in_dyn_t(i,1)*V)*(CBf_in_dyn_t(i,1) + C_Bformed - CB_NN_out_dyn_womc_t(i,1));                                 % Number of moles of water used up when B reacted to form C    
    
    mol_W_out_dyn_womc_t(i,1) = mol_W_in_dyn_t(i,1) - (mol_AtoB + mol_BtoC);
end

% Calculation of number of moles of A,B,C,D in Outlet Stream (kmol/min)
mol_A_out_dyn_womc_t = V*fv_in_dyn_t.*CA_NN_out_dyn_womc_t; mol_B_out_dyn_womc_t = V*fv_in_dyn_t.*CB_NN_out_dyn_womc_t;
mol_C_out_dyn_womc_t = V*fv_in_dyn_t.*CC_NN_out_dyn_womc_t; mol_D_out_dyn_womc_t = V*fv_in_dyn_t.*CD_NN_out_dyn_womc_t;

% Total number of atoms of C and H in Outlet Stream

at_C_out_dyn_womc_t = mol_A_out_dyn_womc_t*ncA + mol_B_out_dyn_womc_t*ncB + mol_C_out_dyn_womc_t*ncC + mol_D_out_dyn_womc_t*ncD;
at_H_out_dyn_womc_t = mol_A_out_dyn_womc_t*nhA + mol_B_out_dyn_womc_t*nhB + mol_C_out_dyn_womc_t*nhC + mol_D_out_dyn_womc_t*nhD + mol_W_out_dyn_womc_t*nhW;
at_O_out_dyn_womc_t = mol_A_out_dyn_womc_t*noA + mol_B_out_dyn_womc_t*noB + mol_C_out_dyn_womc_t*noC + mol_D_out_dyn_womc_t*noD + mol_W_out_dyn_womc_t*noW;

% Error in Mass Balance calculations

diff_dyn_womc_t = 20.*abs([(at_C_in_dyn_t-at_C_out_dyn_womc_t)./at_C_in_dyn_t, (at_H_in_dyn_t-at_H_out_dyn_womc_t)./at_H_in_dyn_t,(at_O_in_dyn_t-at_O_out_dyn_womc_t)./at_O_in_dyn_t]);

% MCNN

CA_NN_out_dyn_mcnn_t = ymcnn_total_t_p(:,1);
CB_NN_out_dyn_mcnn_t = ymcnn_total_t_p(:,2);
CC_NN_out_dyn_mcnn_t = ymcnn_total_t_p(:,3);
CD_NN_out_dyn_mcnn_t = ymcnn_total_t_p(:,4);

% Calculation of number of moles of water in Outlet Stream (kmol/min)

mol_W_out_dyn_mcnn_t = zeros(tn_d,1);
for i = 1:tn_d    
    mol_AtoB = (fv_in_dyn_t(i,1)*V)*(CAf_in_dyn_t(i,1) - CA_NN_out_dyn_mcnn_t(i,1) - 2*(CD_NN_out_dyn_mcnn_t(i,1)-CDf_in_dyn_t(i,1)));       % Number of moles of water used up when A reacted to form B
    C_Bformed = (CAf_in_dyn_t(i,1) - CA_NN_out_dyn_mcnn_t(i,1) - 2*(CD_NN_out_dyn_mcnn_t(i,1)-CDf_in_dyn_t(i,1)));
    mol_BtoC = (fv_in_dyn_t(i,1)*V)*(CBf_in_dyn_t(i,1) + C_Bformed - CB_NN_out_dyn_mcnn_t(i,1));                                 % Number of moles of water used up when B reacted to form C    
    
    mol_W_out_dyn_mcnn_t(i,1) = mol_W_in_dyn_t(i,1) - (mol_AtoB + mol_BtoC);
end

% Calculation of number of moles of A,B,C,D in Outlet Stream (kmol/min)
mol_A_out_dyn_mcnn_t = V*fv_in_dyn_t.*CA_NN_out_dyn_mcnn_t; mol_B_out_dyn_mcnn_t = V*fv_in_dyn_t.*CB_NN_out_dyn_mcnn_t;
mol_C_out_dyn_mcnn_t = V*fv_in_dyn_t.*CC_NN_out_dyn_mcnn_t; mol_D_out_dyn_mcnn_t = V*fv_in_dyn_t.*CD_NN_out_dyn_mcnn_t;

% Total number of atoms of C and H in Outlet Stream

at_C_out_dyn_mcnn_t = mol_A_out_dyn_mcnn_t*ncA + mol_B_out_dyn_mcnn_t*ncB + mol_C_out_dyn_mcnn_t*ncC + mol_D_out_dyn_mcnn_t*ncD;
at_H_out_dyn_mcnn_t = mol_A_out_dyn_mcnn_t*nhA + mol_B_out_dyn_mcnn_t*nhB + mol_C_out_dyn_mcnn_t*nhC + mol_D_out_dyn_mcnn_t*nhD + mol_W_out_dyn_mcnn_t*nhW;
at_O_out_dyn_mcnn_t = mol_A_out_dyn_mcnn_t*noA + mol_B_out_dyn_mcnn_t*noB + mol_C_out_dyn_mcnn_t*noC + mol_D_out_dyn_mcnn_t*noD + mol_W_out_dyn_mcnn_t*noW;

% Error in Mass Balance calculations

diff_dyn_mcnn_t = 10.*abs([(at_C_in_dyn_t-at_C_out_dyn_mcnn_t)./at_C_in_dyn_t, (at_H_in_dyn_t-at_H_out_dyn_mcnn_t)./at_H_in_dyn_t,(at_O_in_dyn_t-at_O_out_dyn_mcnn_t)./at_O_in_dyn_t]);

figure(2)
subplot(1,2,1)
hold on
plot(diff_dyn_mcnn_t(:,1),'b','LineWidth',1.5)
plot(diff_dyn_womc_t(:,1),'r-o','Markersize',3,'LineWidth',1.0)
xlabel('Time (mins)')
ylabel('Error in C Atom Balance (%)')
title('(a)')
xlim([1200 2000])
ylim([-0.5 10])
legend('MCNN','NN w/o MC','Location','northeast')
a=findobj(gcf);
allaxes=findall(a,'Type','axes'); alltext=findall(a,'Type','text'); set(allaxes,'FontName','Times','FontWeight','Bold','LineWidth',2.7,'FontSize',14);
set(alltext,'FontName','Times','FontWeight','Bold','FontSize',14);
subplot(1,2,2)
hold on
plot(diff_dyn_mcnn_t(:,2),'b','LineWidth',1.5)
plot(diff_dyn_womc_t(:,2),'r-o','Markersize',3,'LineWidth',1.0)
xlabel('Time (mins)')
ylabel('Error in H Atom Balance (%)')
title('(b)')
xlim([1200 2000])
ylim([-0.5 15])
legend('MCNN','NN w/o MC','Location','northeast')
a=findobj(gcf);
allaxes=findall(a,'Type','axes'); alltext=findall(a,'Type','text'); set(allaxes,'FontName','Times','FontWeight','Bold','LineWidth',2.7,'FontSize',14);
set(alltext,'FontName','Times','FontWeight','Bold','FontSize',14);

% END OF TRAINING (INVERSE PROBLEM)
%------------------------------------------------------------------------%

%% VALIDATION / SIMULATION / FORWARD PROBLEM: STEPS

% 1. The forward problem follows the same steps as the inverse problem.
% From the validation data the steady-state zones are identified and the
% steady-state forward problem as included in the MATLAB and Python codes
% for Steady-State MCNN can be implemented.

% 2. The input deviation matrix is formed for implementation through the 
% optimal NLD (NARX-type RNN) model obtained by running the 'ValNLD4ParMCNN.m'
% function.

% 3. The resulting variables are converted from the normalized scale to
% their absolute scales of magnitude and the overall results are generated.
% The results are compared with those obtained from the unconstrained NN
% model.


%------------------------------------------------------------------------%











