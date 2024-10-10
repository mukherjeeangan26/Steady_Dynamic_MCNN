function c = SolveDynMCNNCons(x,Imat_t,dsr_t,tn,ni,nh,no,at_C_in_t,F_out_t,V_ins_t,delta,data_dyn)

whd = reshape(x(1:ni*nh),[ni,nh]);
wod = reshape(x(ni*nh+1:ni*nh+nh*no),[nh,no]);
wfd = reshape(x(ni*nh+nh*no+1:ni*nh+nh*no+no*nh),[no,nh]);
bhd = (x(ni*nh+nh*no+no*nh+1:ni*nh+nh*no+no*nh+nh))';
bod = (x(ni*nh+nh*no+no*nh+nh+1:ni*nh+nh*no+no*nh+nh+no))';

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
    ynn2_t_p(:,i) = ynn_t(:,i).*delta(1,ni+i) + min(data_dyn(:,ni+i));
end

ynn1_t_p = y_r;

ncA = 5; ncB = 5; ncC = 5; ncD = 10;

CA_NN_out = ynn1_t_p(:,1); CB_NN_out = ynn1_t_p(:,2);
CC_NN_out = ynn1_t_p(:,3); CD_NN_out = ynn1_t_p(:,4);

c1 = [CA_NN_out;CB_NN_out;CC_NN_out;CD_NN_out;ynn2_t_p(:,1);ynn2_t_p(:,2);ynn2_t_p(:,3);ynn2_t_p(:,4)];

% Calculation of number of moles of A,B,C,D in Outlet Stream (kmol/min)
mol_A_out = F_out_t.*CA_NN_out; mol_B_out = F_out_t.*CB_NN_out;
mol_C_out = F_out_t.*CC_NN_out; mol_D_out = F_out_t.*CD_NN_out;

% Total number of atoms of C and H in Outlet Stream
at_C_out_t = mol_A_out*ncA + mol_B_out*ncB + mol_C_out*ncC + mol_D_out*ncD;

diff = zeros(tn-1,1);

for j = 2:tn
    diff(i-1,1) = (V_ins_t(i,1)*(CA_NN_out(i,1)*ncA + CB_NN_out(i,1)*ncB + CC_NN_out(i,1)*ncC + CD_NN_out(i,1)*ncD)) - ...
        (V_ins_t(i-1,1)*(CA_NN_out(i-1,1)*ncA + CB_NN_out(i-1,1)*ncB + CC_NN_out(i-1,1)*ncC + CD_NN_out(i-1,1)*ncD)) - ...
        (at_C_in_t(i,1)-at_C_out_t(i,1));
end

c2 = diff;

c = [c1;c2];

end