clc
clear

%% Loading the Data for Model Development

% This code develops steady-state MCNN models for the cases when no noise
% or a constant bias with / without Gaussian noise is added to true data to
% generate training data

% This code requires the MATLAB Neural Network Toolbox to train the
% unconstrained neural network model for faster computation. The optimal
% solution of the unconstrained network serve as initial guesses for the
% constrained formulation of the inverse problem.

% Load the training and validation datasets and specify the input and
% output variables to the NN models
% Note that the user can consider any steady-state dataset for training and
% validation. The rows signify the observation indices for steady state data
% and the columns signify the input and output variables.

data = xlsread('Steady-State CSTR Data.xlsx','NoNoise');
% data = xlsread('Steady-State CSTR Data.xlsx','ConstantBias+Gaussian Noise');
data = data(:,2:end);

% For this specific system, the first five columns are the model inputs and
% the following 4 columns are the model outputs

input_data = data(:,1:5); output_data = data(:,6:end);

ni = size(input_data,2);             % Number of inputs = Number of neurons in input layer
no = size(output_data,2);            % Number of outputs = Number of neurons in output layer

% Default number of neurons in hidden layer taken equal to number of
% inputs. But it can be changed as per requirement.

nh = ni;

nt = ni + no;

% Total number of datasets available for model development

n = size(data,1);

%-----------------------------------------------------------------------%

%% Defining the System Model in terms of Process Variables

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

% Generating random training data for tn steps

tr_steps = randperm(tt,tn);
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

%% Training of Unconstrained NN Model

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

CA_NN_out_wo = ynn_wo_t_p(:,1); CB_NN_out_wo = ynn_wo_t_p(:,2);
CC_NN_out_wo = ynn_wo_t_p(:,3); CD_NN_out_wo = ynn_wo_t_p(:,4);

% Calculation of number of moles of water in Outlet Stream (kmol/min)

mol_W_out_wo = zeros(tn,1);
for i = 1:tn    
    % Calculation of number of moles of water used up in reaction (kmol/min)
    mol_AtoB = (fv_in_t(i,1)*V)*(CAf_in_t(i,1) - CA_NN_out_wo(i,1) - 2*(CD_NN_out_wo(i,1)-CDf_in_t(i,1)));            % Number of moles of water used up when A reacted to form B
    C_Bformed = (CAf_in_t(i,1) - CA_NN_out_wo(i,1) - 2*(CD_NN_out_wo(i,1)-CDf_in_t(i,1)));
    mol_BtoC = (fv_in_t(i,1)*V)*(CBf_in_t(i,1) + C_Bformed - CB_NN_out_wo(i,1));                                 % Number of moles of water used up when B reacted to form C    
    
    mol_W_out_wo(i,1) = mol_W_in_t(i,1) - (mol_AtoB + mol_BtoC);
end

% Calculation of number of moles of A,B,C,D in Outlet Stream (kmol/min)
mol_A_out_wo = V*fv_in_t.*CA_NN_out_wo; mol_B_out_wo = V*fv_in_t.*CB_NN_out_wo;
mol_C_out_wo = V*fv_in_t.*CC_NN_out_wo; mol_D_out_wo = V*fv_in_t.*CD_NN_out_wo;

% Total number of atoms of C and H in Outlet Stream

at_C_out_t_wo = mol_A_out_wo*ncA + mol_B_out_wo*ncB + mol_C_out_wo*ncC + mol_D_out_wo*ncD;
at_H_out_t_wo = mol_A_out_wo*nhA + mol_B_out_wo*nhB + mol_C_out_wo*nhC + mol_D_out_wo*nhD + mol_W_out_wo*nhW;
at_O_out_t_wo = mol_A_out_wo*noA + mol_B_out_wo*noB + mol_C_out_wo*noC + mol_D_out_wo*noD + mol_W_out_wo*noW;

% Error in Mass Balance calculations

diff_wo = 100.*abs([(at_C_in_t-at_C_out_t_wo)./at_C_in_t, (at_H_in_t-at_H_out_t_wo)./at_H_in_t,(at_O_in_t-at_O_out_t_wo)./at_O_in_t]);

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

opts = optiset('solver','ipopt','display','iter','maxiter',1e3,'maxtime',1200); 
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

CA_NN_out = ynn_t_p(:,1); CB_NN_out = ynn_t_p(:,2);
CC_NN_out = ynn_t_p(:,3); CD_NN_out = ynn_t_p(:,4);

% Calculation of number of moles of water in Outlet Stream (kmol/min)
mol_W_out = zeros(tn,1);
for i = 1:tn
    
    % Calculation of number of moles of water used up in reaction (kmol/min)
    mol_AtoB = (fv_in_t(i,1)*V)*(CAf_in_t(i,1) - CA_NN_out(i,1) - 2*(CD_NN_out(i,1)-CDf_in_t(i,1)));            % Number of moles of water used up when A reacted to form B
    C_Bformed = (CAf_in_t(i,1) - CA_NN_out(i,1) - 2*(CD_NN_out(i,1)-CDf_in_t(i,1)));
    mol_BtoC = (fv_in_t(i,1)*V)*(CBf_in_t(i,1) + C_Bformed - CB_NN_out(i,1));                                 % Number of moles of water used up when B reacted to form C
    
    mol_W_out(i,1) = mol_W_in_t(i,1) - (mol_AtoB + mol_BtoC);
end

% Calculation of number of moles of A,B,C,D in Outlet Stream (kmol/min)
mol_A_out = V*fv_in_t.*CA_NN_out; mol_B_out = V*fv_in_t.*CB_NN_out;
mol_C_out = V*fv_in_t.*CC_NN_out; mol_D_out = V*fv_in_t.*CD_NN_out;

% Total number of atoms of C and H in Outlet Stream

at_C_out_t = mol_A_out*ncA + mol_B_out*ncB + mol_C_out*ncC + mol_D_out*ncD;
at_H_out_t = mol_A_out*nhA + mol_B_out*nhB + mol_C_out*nhC + mol_D_out*nhD + mol_W_out*nhW;
at_O_out_t = mol_A_out*noA + mol_B_out*noB + mol_C_out*noC + mol_D_out*noD + mol_W_out*noW;

% Error in Mass Balance calculations

diff = 100.*abs([(at_C_in_t-at_C_out_t)./at_C_in_t, (at_H_in_t-at_H_out_t)./at_H_in_t, (at_O_in_t - at_O_out_t)./at_O_in_t]);

% END OF TRAINING
%------------------------------------------------------------------------%

%% Validation / Simulation of Unconstrained Network

flag = 1;
tv = tt - tn;
val_steps = zeros(tv,1);

for i = 1:tt
    check = ismember(i,tr_steps);    
    if check == 0
        val_steps(flag,1) = i;
        flag = flag+1;
    end
end

val_steps = sort(val_steps);

dsr_v = zeros(no,tv); Imat_v = zeros(ni,tv); 
fv_in_v = zeros(tv,1); CAf_in_v = zeros(tv,1); CBf_in_v = zeros(tv,1); 
CCf_in_v = zeros(tv,1); CDf_in_v = zeros(tv,1); at_C_in_v = zeros(tv,1); 
at_H_in_v = zeros(tv,1); at_O_in_v = zeros(tv,1); mol_W_in_v = zeros(tv,1);

for i = 1:tv
    
    ts = val_steps(i,1);    
    dsr_v(1:no,i) = dsr(1:no,ts);
    Imat_v(1:ni,i) = Imat(1:ni,ts);
    fv_in_v(i,1) = fv_in(ts,1);
    CAf_in_v(i,1) = CAf_in(ts,1);
    CBf_in_v(i,1) = CBf_in(ts,1);
    CCf_in_v(i,1) = CCf_in(ts,1);
    CDf_in_v(i,1) = CDf_in(ts,1);
    at_C_in_v(i,1) = at_C_in(ts,1);
    at_H_in_v(i,1) = at_H_in(ts,1);
    at_O_in_v(i,1) = at_O_in(ts,1);
    mol_W_in_v(i,1) = mol_W_in(ts,1);
    
end

y1 = Imat_v;
x1 = whf_wo_opt'*y1 + bhf_wo_opt;
y2 = logsig(x1);
x2 = wof_wo_opt'*y2 + bof_wo_opt;
ynn_wo_v = purelin(x2);

ynn_wo_v_p = zeros(tv,no);

for i = 1:no
    ynn_wo_v_p(:,i) = ynn_wo_v(i,:)'.*delta(1,ni+i) + min(data(:,ni+i));
end

CA_NN_out_wo = ynn_wo_v_p(:,1); CB_NN_out_wo = ynn_wo_v_p(:,2);
CC_NN_out_wo = ynn_wo_v_p(:,3); CD_NN_out_wo = ynn_wo_v_p(:,4);

% Calculation of number of moles of water in Outlet Stream (kmol/min)
mol_W_out_wo_v = zeros(tv,1);
for i = 1:tv
    
    % Calculation of number of moles of water used up in reaction (kmol/min)
    mol_AtoB = (fv_in_v(i,1)*V)*(CAf_in_v(i,1) - CA_NN_out_wo(i,1) - 2*(CD_NN_out_wo(i,1)-CDf_in_v(i,1)));            % Number of moles of water used up when A reacted to form B
    C_Bformed = (CAf_in_v(i,1) - CA_NN_out_wo(i,1) - 2*(CD_NN_out_wo(i,1)-CDf_in_v(i,1)));
    mol_BtoC = (fv_in_t(i,1)*V)*(CBf_in_v(i,1) + C_Bformed - CB_NN_out_wo(i,1));                                 % Number of moles of water used up when B reacted to form C
    
    
    mol_W_out_wo_v(i,1) = mol_W_in_v(i,1) - (mol_AtoB + mol_BtoC);
end

% Calculation of number of moles of A,B,C,D in Outlet Stream (kmol/min)
mol_A_out_wo = V*fv_in_v.*CA_NN_out_wo; mol_B_out_wo = V*fv_in_v.*CB_NN_out_wo;
mol_C_out_wo = V*fv_in_v.*CC_NN_out_wo; mol_D_out_wo = V*fv_in_v.*CD_NN_out_wo;

% Total number of atoms of C and H in Outlet Stream
at_C_out_v_wo = mol_A_out_wo*ncA + mol_B_out_wo*ncB + mol_C_out_wo*ncC + mol_D_out_wo*ncD;
at_H_out_v_wo = mol_A_out_wo*nhA + mol_B_out_wo*nhB + mol_C_out_wo*nhC + mol_D_out_wo*nhD + mol_W_out_wo_v*nhW;
at_O_out_v_wo = mol_A_out_wo*noA + mol_B_out_wo*noB + mol_C_out_wo*noC + mol_D_out_wo*noD + mol_W_out_wo_v*noW;

diff_wo_v = 100.*abs([(at_C_in_v-at_C_out_v_wo)./at_C_in_v, (at_H_in_v-at_H_out_v_wo)./at_H_in_v, (at_O_in_v-at_O_out_v_wo)./at_O_in_v]);

%------------------------------------------------------------------------%

%% Validation / Simulation of MCNN (including Dynamic Data Reconciliation Post-Processing Step)

y1 = Imat_v;
x1 = whf'*y1 + bhf;
y2 = logsig(x1);
x2 = wof'*y2 + bof;
ynn_v = purelin(x2);

ynn_v_p = zeros(tv,no);

for i = 1:no
    ynn_v_p(:,i) = ynn_v(i,:)'.*delta(1,ni+i) + min(data(:,ni+i));
end
    
ynn_v = ynn_v_p';

% Dynamic Data Reconciliation Post-Processing

disp('START: Data Reconciliation Step during Validation')

ynn_v0 = reshape(1*ynn_v,[1 no*tv]);

lb = -1e5.*ones(1,size(ynn_v0,2))';
ub = 1e5.*ones(1,size(ynn_v0,2))';

obj_v = @(x)ForwProbV1(x,ynn_v,tv,no);
nlcon_v = @(x)ForwProbV1Cons(x,ynn_v,tv,no,at_C_in_v,at_H_in_v,at_O_in_v,fv_in_v,V,CAf_in_v,CBf_in_v,CDf_in_v,mol_W_in_v);

nlrhs_v = zeros((no+3)*tv,1);
nle_v = [ones(no*tv,1);zeros(3*tv,1)];

opts_v = optiset('solver','ipopt','display','iter','maxiter',1e3,'maxtime',1000); 
Opt_v = opti('fun',obj_v,'ineq',[],[],'nlmix',nlcon_v,nlrhs_v,nle_v,'bounds',lb,ub,'options',opts_v);

[ynn_v_r,fval_r,exitflag_r,info_r] = solve(Opt_v,ynn_v0);

disp('END: Data Reconciliation Step during Validation')

ynn_v_r = reshape(ynn_v_r,size(ynn_v));

ynn_v_p = ynn_v_r';
dsr_v_p = zeros(tv,no);

for i = 1:no
    dsr_v_p(:,i) = (dsr_v(i,:))'.*delta(1,ni+i) + min(data(:,ni+i));
end

CA_NN_out_v = ynn_v_p(:,1); CB_NN_out_v = ynn_v_p(:,2);
CC_NN_out_v = ynn_v_p(:,3); CD_NN_out_v = ynn_v_p(:,4);

% Calculation of number of moles of water in Outlet Stream (kmol/min)
mol_W_out_v = zeros(tv,1);
for i = 1:tv    
    % Calculation of number of moles of water used up in reaction (kmol/min)
    mol_AtoB = (fv_in_v(i,1)*V)*(CAf_in_v(i,1) - CA_NN_out_v(i,1) - 2*(CD_NN_out_v(i,1)-CDf_in_v(i,1)));            % Number of moles of water used up when A reacted to form B
    C_Bformed = (CAf_in_v(i,1) - CA_NN_out_v(i,1) - 2*(CD_NN_out_v(i,1)-CDf_in_v(i,1)));
    mol_BtoC = (fv_in_v(i,1)*V)*(CBf_in_v(i,1) + C_Bformed - CB_NN_out_v(i,1));                                 % Number of moles of water used up when B reacted to form C
    
    mol_W_out_v(i,1) = mol_W_in_v(i,1) - (mol_AtoB + mol_BtoC);
end

% Calculation of number of moles of A,B,C,D in Outlet Stream (kmol/min)
mol_A_out_v = V*fv_in_v.*CA_NN_out_v; mol_B_out_v = V*fv_in_v.*CB_NN_out_v;
mol_C_out_v = V*fv_in_v.*CC_NN_out_v; mol_D_out_v = V*fv_in_v.*CD_NN_out_v;

% Total number of atoms of C and H in Outlet Stream
at_C_out_v = mol_A_out_v*ncA + mol_B_out_v*ncB + mol_C_out_v*ncC + mol_D_out_v*ncD;
at_H_out_v = mol_A_out_v*nhA + mol_B_out_v*nhB + mol_C_out_v*nhC + mol_D_out_v*nhD + mol_W_out_v*nhW;
at_O_out_v = mol_A_out_v*noA + mol_B_out_v*noB + mol_C_out_v*noC + mol_D_out_v*noD + mol_W_out_v*noW;

diff_v = 100.*abs([(at_C_in_v - at_C_out_v)./at_C_in_v, (at_H_in_v - at_H_out_v)./at_H_in_v, (at_O_in_v - at_O_out_v)./at_O_in_v]); 

%------------------------------------------------------------------------%

%% Plotting Error in Mass Balance Plots for Training

figure(1)
subplot(1,2,1)
hold on
plot(tr_steps,diff_wo(:,1),'r');
plot(tr_steps,diff(:,1),'b');
title('Isothermal CSTR System');
xlabel('Indices for Training Data');
ylabel('Error in C Atom Balance');
legend('NN w/o MC','MCNN','Location','northeast')
grid on
subplot(1,2,2)
hold on
plot(tr_steps,diff_wo(:,2),'r');
plot(tr_steps,diff(:,2),'b');
title('Isothermal CSTR System');
xlabel('Indices for Training Data');
ylabel('Error in H Atom Balance');
legend('NN w/o MC','MCNN','Location','northeast')
grid on

figure(1)
for i = 1:2
    subplot(1,2,i)
    a=findobj(gcf);
    allaxes=findall(a,'Type','axes'); alllines=findall(a,'Type','line'); alltext=findall(a,'Type','text'); set(allaxes,'FontName','Times','FontWeight','Bold','LineWidth',2.7,'FontSize',14); set(alllines,'Linewidth',1.2);
    set(alltext,'FontName','Times','FontWeight','Bold','FontSize',14);
end


%% Plotting Error in Mass Balance Plots for Validation / Simulation

figure(2)
subplot(1,2,1)
hold on
plot(val_steps,diff_wo_v(:,1),'r');
plot(val_steps,diff_v(:,1),'b');
title('Isothermal CSTR System');
xlabel('Indices for Validation Data');
ylabel('Error in C Atom Balance');
legend('NN w/o MC','MCNN','Location','northeast')
grid on
subplot(1,2,2)
hold on
plot(val_steps,diff_wo_v(:,2),'r');
plot(val_steps,diff_v(:,2),'b');
title({'Isothermal CSTR System'});
xlabel('Indices for Validation Data');
ylabel('Error in H Atom Balance');
legend('NN w/o MC','MCNN','Location','northeast')
grid on

figure(2)
for i = 1:2
    subplot(1,2,i)
    a=findobj(gcf);
    allaxes=findall(a,'Type','axes'); alllines=findall(a,'Type','line'); alltext=findall(a,'Type','text'); set(allaxes,'FontName','Times','FontWeight','Bold','LineWidth',2.7,'FontSize',14); set(alllines,'Linewidth',1.2);
    set(alltext,'FontName','Times','FontWeight','Bold','FontSize',14);
end

%------------------------------------------------------------------------%
