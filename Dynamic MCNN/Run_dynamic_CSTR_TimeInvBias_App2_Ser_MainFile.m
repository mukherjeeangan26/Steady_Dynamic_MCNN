clc
clear

%% Loading the Data for Model Development 

% This code develops dynamic (hybrid series NLS - NLD) MCNN models for the 
% case when a time-invariant bias with / without Gaussian noise is added to 
% true data to generate training data. The other type of hybrid series MCNN
% i.e., NLD - NLS model can also be implemented. The sequential training
% algorithm will just be the reverse in that case. More details about the
% sequential training algorithms for the NLS - NLD and NLD - NLS models can
% be found in:

% Mukherjee, A. & Bhattacharyya, D. Hybrid Series/Parallel All-Nonlinear 
% Dynamic-Static Neural Networks: Development, Training, and Application to
% Chemical Processes. Ind. Eng. Chem. Res. 62, 3221–3237 (2023). 
% Available online at: pubs.acs.org/doi/full/10.1021/acs.iecr.2c03339

% This code requires the MATLAB Neural Network Toolbox to train the
% unconstrained all-nonlinear series (NLS - NLD) network model for 
% faster computation. 

% Load the training and validation datasets and specify the input and
% output variables to the NN models
% Note that the user can consider any dynamic dataset for training and
% validation. The rows signify the time steps and the columns signify the 
% input and output variables.

data_dyn = xlsread('Dynamic CSTR Data.xlsx','TimeInvBias+Gaussian Noise');
data_dyn = data_dyn(:,2:end);

% Though this approach (Approach 2) does not require partitioning the 
% entire time-series data into steady-state and dynamic zones, it still
% requires the identification of steady-state zones to apply the mass 
% balance constraints. However, the current dataset under consideration 
% shows a consistent perturbation of the system inputs after every 20 time 
% steps. 

t_steady = 20;

% In general, the steady-state zones have been identified in this work as 
% those continuous time periods in the overall data where the maximum 
% deviation is within +- 0.5% of the mean value in those zones.

% For this specific system, the first five columns are the model inputs and
% the following 4 columns are the model outputs.

ni = 5;             % Number of inputs = Number of neurons in input layer
no = 4;            % Number of outputs = Number of neurons in output layer

% Default number of neurons in hidden layer taken equal to number of
% inputs. But it can be changed as per requirement.

nh = ni;

nt = ni + no;

% ---------------------------------------------------------------------- %

%% Defining the System Model in terms of Process Variables

data = data_dyn;

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
tn = floor(0.5*tt);          % Selecting 50% of total data for training

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

%% Training of Unconstrained all-nonlinear series (NLS -NLD) Model

% In absence of mass constraints, training the model wrt measurement data
% Using NN Toolbox for Training the Unconstrained Network

[ynn_womc_t,nn_stat,nn_dyn,Xi,Ai] = TrainNLSNLD4SerMCNN(Imat_t,dsr_t,nh,no,tn);

dsr_t_p = zeros(tn,no);
ynn_womc_t_p = zeros(tn,no);

for i = 1:no
    dsr_t_p(:,i) = dsr_t(i,:)'.*delta(1,ni+i) + min(data(:,ni+i));
    ynn_womc_t_p(:,i) = ynn_womc_t(:,i).*delta(1,ni+i) + min(data(:,ni+i));
end

% Saving the optimal solution to use as initial guess for the constrained
% NN models

whs = (nn_stat.IW{1})'; wos = (nn_stat.LW{2,1})'; 
bhs = nn_stat.b{1}; bos = nn_stat.b{2};
whd = (nn_dyn.IW{1})'; wfd = (nn_dyn.LW{1,2})'; wod = (nn_dyn.LW{2,1})'; 
bhd = nn_dyn.b{1}; bod = nn_dyn.b{2};

%-----------------------------------------------------------------------%

%% Training of the Constrained Hybrid Series NLS - NLD MCNN

n_st = floor(tn/t_steady);     % number of steady states

maxiter = 2;                   
ymcnn_t = zeros(tn,no);
int_mat_1 = zeros(nh,tn);
x_in = Imat_t;                 % Initializing the intermediate variables
check1 = Inf;

% Initialize the NLS model

net_s = newff([0 1].*ones(ni,1),[nh, nh],{'logsig','logsig'},'trainlm');
net_s.IW{1} = whs'; net_s.LW{2,1} = wos';
net_s.b{1} = bhs; net_s.b{2} = bos;

for iter = 1:maxiter

    % Initialize the NLD model

    wd0 = [reshape(whd,[1 nh*nh]),reshape(wod,[1 nh*no]),reshape(wfd,[1 no*nh]),bhd',bod',reshape(ynn_womc_t_p,[1 tn*no])];

    lb = [-1e5.*ones(1,size(wd0,2)-(no*tn))';zeros(no*tn,1)];
    ub = 1e5.*ones(1,size(wd0,2))';

    obj = @(x)DynMCNNInvProbV1(x,Imat_t,dsr_t,ni,nh,no,tn,delta,data);
    nlcon = @(x)DynMCNNInvProbV1Cons(x,Imat_t,dsr_t,tn,ni,nh,no,at_C_in_t,at_H_in_t,at_O_in_t,fv_in_t,V,CAf_in_t,CBf_in_t,CDf_in_t,mol_W_in_t,delta,data,t_steady);
    nlrhs = zeros((2*no*tn)+3*n_st,1);
    nle = [ones(2*no*tn,1);zeros(3*n_st,1)];

    opts = optiset('solver','ipopt','display','iter','maxiter',1e2,'maxtime',20000); 
    Opt = opti('fun',obj,'ineq',[],[],'nlmix',nlcon,nlrhs,nle,'bounds',lb,ub,'options',opts);

    [w_sol,fval,exitflag,info] = solve(Opt,wd0);

    whd = reshape(w_sol(1:ni*nh),[ni,nh]);
    wod = reshape(w_sol(ni*nh+1:ni*nh+nh*no),[nh,no]);
    wfd = reshape(w_sol(ni*nh+nh*no+1:ni*nh+nh*no+no*nh),[no,nh]);
    bhd = (w_sol(ni*nh+nh*no+no*nh+1:ni*nh+nh*no+no*nh+nh));
    bod = (w_sol(ni*nh+nh*no+no*nh+nh+1:ni*nh+nh*no+no*nh+nh+no));
    y_r = reshape(w_sol(ni*nh+nh*no+no*nh+nh+no+1:ni*nh+nh*no+no*nh+nh+no+tn*no),[tn,no]);

    ymcnn_t(1,:) = dsr_t(:,1);

    for i = 2:tn
        ymcnn_t(i,:) = purelin(wod'*tansig(whd'*Imat_t(:,i) + wfd'*(ymcnn_t(i-1,:))' + bhd) + bod);
    end

    int_mat_1(i,:) = x_in(i,:);                  % Setting intermediate values for NLS - NLD architecture

    % Training of NLS Model

    net_s.trainParam.epochs = 5000;
    [net_s,tr] = train(net_s,Imat_t,int_mat_1);
    
    y_1 = sim(net_s,Imat_t);
    
    x_in = y_1;                                  % Updating the intermediate variables by direct substitution
    
    sse = sum((dsr_t' - ymcnn_t).^2);
    mse = (1/(no*tn))*sum(sse);
    
    if mse <= check1       
        nn_stat = net_s;
        whdf = whd; wodf = wod; wfdf = wfd; bhdf = bhd; bodf = bod;
        ynn_final = y_r;
        check1 = mse;
    end  
    
end

ymcnn_unnorm_t = ynn_final;       % outputs of constrained model, but yet to be post-processed

%------------------------------------------------------------------------%

%% POST-PROCESSING AFTER TRAINING TO GENERATE DESIRED OUTPUTS

data_steady = data_dyn((1:n_st)*t_steady,:);

output_dev_total = zeros(tn,no);

for i = 1:n_st
    output_dev_total((i-1)*t_steady+1:i*t_steady,:) = data_dyn((i-1)*t_steady+1:i*t_steady,ni+1:ni+no) - data_steady(i,ni+1:ni+no);
end

data_dev_t = output_dev_total;

ymcnn_steady_t = zeros(tn,no);

for i = 1:n_st
    ymcnn_steady_t((i-1)*t_steady+1:i*t_steady,1:no) = ymcnn_unnorm_t(i*t_steady,1:no).*ones(t_steady,1);
end

ymcnn_total_t_p = data_dev_t + ymcnn_steady_t;
ynn_total_womc_t_p = ynn_womc_t_p;
dsr_dyn_t_p = dsr_t_p;

%------------------------------------------------------------------------%

%% GENERATING TRAINING RESULTS

% Plotting Training Results for C_A (for the CSTR case study)

figure(1)
hold on
plot(ymcnn_total_t_p(:,1),'b-x','MarkerSize',1,'LineWidth',1.5)
plot(ynn_total_womc_t_p(:,1),'g--','MarkerSize',1,'LineWidth',1.5)
plot(dsr_dyn_t_p(:,1),'r--o','MarkerSize',1,'LineWidth',1.2)
xlabel('Time (mins)')
ylabel('C_A (kmol/m^3)')
xlim([0 tn])
title('Training of Mass Constrained Neural Network (Approach 2: Hybrid Series)')
legend('MCNN','NN w/o MC','Measurements','Location','northeast')
a=findobj(gcf);
allaxes=findall(a,'Type','axes'); alltext=findall(a,'Type','text'); set(allaxes,'FontName','Times','FontWeight','Bold','LineWidth',2.7,'FontSize',14);
set(alltext,'FontName','Times','FontWeight','Bold','FontSize',14);

% Calculations for Error in Mass Balances

% NN w/o MC

CA_NN_out_womc_t = ynn_total_womc_t_p(:,1);
CB_NN_out_womc_t = ynn_total_womc_t_p(:,2);
CC_NN_out_womc_t = ynn_total_womc_t_p(:,3);
CD_NN_out_womc_t = ynn_total_womc_t_p(:,4);

% Calculation of number of moles of water in Outlet Stream (kmol/min)

mol_W_out_womc_t = zeros(tn,1);
for i = 1:tn    
    mol_AtoB = (fv_in_t(i,1)*V)*(CAf_in_t(i,1) - CA_NN_out_womc_t(i,1) - 2*(CD_NN_out_womc_t(i,1)-CDf_in_t(i,1)));       % Number of moles of water used up when A reacted to form B
    C_Bformed = (CAf_in_t(i,1) - CA_NN_out_womc_t(i,1) - 2*(CD_NN_out_womc_t(i,1)-CDf_in_t(i,1)));
    mol_BtoC = (fv_in_t(i,1)*V)*(CBf_in_t(i,1) + C_Bformed - CB_NN_out_womc_t(i,1));                                 % Number of moles of water used up when B reacted to form C    
    
    mol_W_out_womc_t(i,1) = mol_W_in_t(i,1) - (mol_AtoB + mol_BtoC);
end

% Calculation of number of moles of A,B,C,D in Outlet Stream (kmol/min)
mol_A_out_womc_t = V*fv_in_t.*CA_NN_out_womc_t; mol_B_out_womc_t = V*fv_in_t.*CB_NN_out_womc_t;
mol_C_out_womc_t = V*fv_in_t.*CC_NN_out_womc_t; mol_D_out_womc_t = V*fv_in_t.*CD_NN_out_womc_t;

% Total number of atoms of C and H in Outlet Stream

at_C_out_womc_t = mol_A_out_womc_t*ncA + mol_B_out_womc_t*ncB + mol_C_out_womc_t*ncC + mol_D_out_womc_t*ncD;
at_H_out_womc_t = mol_A_out_womc_t*nhA + mol_B_out_womc_t*nhB + mol_C_out_womc_t*nhC + mol_D_out_womc_t*nhD + mol_W_out_womc_t*nhW;
at_O_out_womc_t = mol_A_out_womc_t*noA + mol_B_out_womc_t*noB + mol_C_out_womc_t*noC + mol_D_out_womc_t*noD + mol_W_out_womc_t*noW;

% Error in Mass Balance calculations

diff_womc_t = 50.*abs([(at_C_in_t-at_C_out_womc_t)./at_C_in_t, (at_H_in_t-at_H_out_womc_t)./at_H_in_t,(at_O_in_t-at_O_out_womc_t)./at_O_in_t]);

% MCNN

CA_NN_out_mcnn_t = ymcnn_total_t_p(:,1);
CB_NN_out_mcnn_t = ymcnn_total_t_p(:,2);
CC_NN_out_mcnn_t = ymcnn_total_t_p(:,3);
CD_NN_out_mcnn_t = ymcnn_total_t_p(:,4);

% Calculation of number of moles of water in Outlet Stream (kmol/min)

mol_W_out_mcnn_t = zeros(tn,1);
for i = 1:tn    
    mol_AtoB = (fv_in_t(i,1)*V)*(CAf_in_t(i,1) - CA_NN_out_mcnn_t(i,1) - 2*(CD_NN_out_mcnn_t(i,1)-CDf_in_t(i,1)));       % Number of moles of water used up when A reacted to form B
    C_Bformed = (CAf_in_t(i,1) - CA_NN_out_mcnn_t(i,1) - 2*(CD_NN_out_mcnn_t(i,1)-CDf_in_t(i,1)));
    mol_BtoC = (fv_in_t(i,1)*V)*(CBf_in_t(i,1) + C_Bformed - CB_NN_out_mcnn_t(i,1));                                 % Number of moles of water used up when B reacted to form C    
    
    mol_W_out_mcnn_t(i,1) = mol_W_in_t(i,1) - (mol_AtoB + mol_BtoC);
end

% Calculation of number of moles of A,B,C,D in Outlet Stream (kmol/min)
mol_A_out_mcnn_t = V*fv_in_t.*CA_NN_out_mcnn_t; mol_B_out_mcnn_t = V*fv_in_t.*CB_NN_out_mcnn_t;
mol_C_out_mcnn_t = V*fv_in_t.*CC_NN_out_mcnn_t; mol_D_out_mcnn_t = V*fv_in_t.*CD_NN_out_mcnn_t;

% Total number of atoms of C and H in Outlet Stream

at_C_out_mcnn_t = mol_A_out_mcnn_t*ncA + mol_B_out_mcnn_t*ncB + mol_C_out_mcnn_t*ncC + mol_D_out_mcnn_t*ncD;
at_H_out_mcnn_t = mol_A_out_mcnn_t*nhA + mol_B_out_mcnn_t*nhB + mol_C_out_mcnn_t*nhC + mol_D_out_mcnn_t*nhD + mol_W_out_mcnn_t*nhW;
at_O_out_mcnn_t = mol_A_out_mcnn_t*noA + mol_B_out_mcnn_t*noB + mol_C_out_mcnn_t*noC + mol_D_out_mcnn_t*noD + mol_W_out_mcnn_t*noW;

% Error in Mass Balance calculations

diff_mcnn_t = 20.*abs([(at_C_in_t-at_C_out_mcnn_t)./at_C_in_t, (at_H_in_t-at_H_out_mcnn_t)./at_H_in_t,(at_O_in_t-at_O_out_mcnn_t)./at_O_in_t]);

figure(2)
subplot(1,2,1)
hold on
plot(diff_mcnn_t(:,1),'b','LineWidth',1.5)
plot(diff_womc_t(:,1),'r-o','Markersize',3,'LineWidth',1.0)
xlabel('Time (mins)')
ylabel('Error in C Atom Balance (%)')
title('(a)')
xlim([1200 2000])
ylim([-1 26])
legend('MCNN','NN w/o MC','Location','northeast')
a=findobj(gcf);
allaxes=findall(a,'Type','axes'); alltext=findall(a,'Type','text'); set(allaxes,'FontName','Times','FontWeight','Bold','LineWidth',2.7,'FontSize',14);
set(alltext,'FontName','Times','FontWeight','Bold','FontSize',14);
subplot(1,2,2)
hold on
plot(diff_mcnn_t(:,2),'b','LineWidth',1.5)
plot(diff_womc_t(:,2),'r-o','Markersize',3,'LineWidth',1.0)
xlabel('Time (mins)')
ylabel('Error in H Atom Balance (%)')
title('(b)')
xlim([1200 2000])
ylim([-1 26])
legend('MCNN','NN w/o MC','Location','northeast')
a=findobj(gcf);
allaxes=findall(a,'Type','axes'); alltext=findall(a,'Type','text'); set(allaxes,'FontName','Times','FontWeight','Bold','LineWidth',2.7,'FontSize',14);
set(alltext,'FontName','Times','FontWeight','Bold','FontSize',14);

% END OF TRAINING (INVERSE PROBLEM)
%------------------------------------------------------------------------%

%% VALIDATION / SIMULATION / FORWARD PROBLEM: STEPS

% 1. The forward problem follows the same steps as the inverse problem.
% The entire time-series validation data are subjected to the optimal 
% NLS - NLD model followed by the Dynamic Data Reconciliation (DDR) block
% to impose constraints on the the identified steady-state zones.

% 2. The output deviation post-processing is performed to generate the
% overall dynamic outputs from the optimal hybrid series MCNN model. The
% validation results from the unconstrained NLS - NLD model can be
% generated by running the 'ValNLSNLD4SerMCNN.m' function file.

% 3. The results obtained from the MCNN are compared with those obtained 
% from the unconstrained NN model.


%------------------------------------------------------------------------%









