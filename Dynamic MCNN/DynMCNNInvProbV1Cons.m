function c = DynMCNNInvProbV1Cons(x,Imat_t,dsr_t,tn,ni,nh,no,at_C_in_t,at_H_in_t,at_O_in_t,fv_in_t,V,CAf_in_t,CBf_in_t,CDf_in_t,mol_W_in_t,delta,data,t_steady)

whd = reshape(x(1:ni*nh),[ni,nh]);
wod = reshape(x(ni*nh+1:ni*nh+nh*no),[nh,no]);
wfd = reshape(x(ni*nh+nh*no+1:ni*nh+nh*no+no*nh),[no,nh]);
bhd = (x(ni*nh+nh*no+no*nh+1:ni*nh+nh*no+no*nh+nh))';
bod = (x(ni*nh+nh*no+no*nh+nh+1:ni*nh+nh*no+no*nh+nh+no))';

n_st = floor(size(dsr_t,2)/t_steady);
y_r = reshape(x(ni*nh+nh*no+no*nh+nh+no+1:ni*nh+nh*no+no*nh+nh+no+tn*no),[tn,no]);

bhdmat = zeros(nh,tn);
for i = 1:size(bhdmat,2)
    bhdmat(:,i) = bhd;
end

bodmat = zeros(no,tn);
for i = 1:size(bodmat,2)
    bodmat(:,i) = bod;
end

ynn_t = zeros(tn,no);

ynn_t(1,:) = dsr_t(:,1);

for i = 2:tn
    ynn_t(i,:) = purelin(wod'*tansig(whd'*Imat_t(:,i) + wfd'*(ynn_t(i-1,:))' + bhdmat(:,i)) + bodmat(:,i));
end

ynn2_t_p = zeros(tn,no);

for i = 1:no
    ynn2_t_p(:,i) = ynn_t(:,i).*delta(1,ni+i) + min(data(:,ni+i));
end

ynn1_t_p = y_r;

ncA = 5; ncB = 5; ncC = 5; ncD = 10;
nhA = 6; nhB = 8; nhC = 10; nhD = 12; nhW = 2;
noA = 0; noB = 1; noC = 2; noD = 0; noW = 1;

CA_NN_out = ynn1_t_p(:,1); CB_NN_out = ynn1_t_p(:,2);
CC_NN_out = ynn1_t_p(:,3); CD_NN_out = ynn1_t_p(:,4);

c1 = [CA_NN_out;CB_NN_out;CC_NN_out;CD_NN_out;ynn2_t_p(:,1);ynn2_t_p(:,2);ynn2_t_p(:,3);ynn2_t_p(:,4)];

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
mol_A_out = V*fv_in_t(1:tn,:).*CA_NN_out(1:tn,:); mol_B_out = V*fv_in_t(1:tn,:).*CB_NN_out(1:tn,:);
mol_C_out = V*fv_in_t(1:tn,:).*CC_NN_out(1:tn,:); mol_D_out = V*fv_in_t(1:tn,:).*CD_NN_out(1:tn,:);

% Total number of atoms of C and H in Outlet Stream

at_C_out_t = mol_A_out*ncA + mol_B_out*ncB + mol_C_out*ncC + mol_D_out*ncD;
at_H_out_t = mol_A_out*nhA + mol_B_out*nhB + mol_C_out*nhC + mol_D_out*nhD + mol_W_out*nhW;
at_O_out_t = mol_A_out*noA + mol_B_out*noB + mol_C_out*noC + mol_D_out*noD + mol_W_out*noW;

c2 = [at_C_in_t(t_steady.*(1:n_st),:) - at_C_out_t(t_steady.*(1:n_st),:); at_H_in_t(t_steady.*(1:n_st),:) - at_H_out_t(t_steady.*(1:n_st),:); at_O_in_t(t_steady.*(1:n_st),:) - at_O_out_t(t_steady.*(1:n_st),:)];

c = [c1;c2];

end