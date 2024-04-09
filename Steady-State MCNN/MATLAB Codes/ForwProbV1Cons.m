function c = ForwProbV1Cons(x,ynn_v,tv,no,at_C_in_v,at_H_in_v,at_O_in_v,fv_in_v,V,CAf_in_v,CBf_in_v,CDf_in_v,mol_W_in_v)

ynn_v_r = x(1:tv*no);
ynn_v_r = reshape(ynn_v_r,size(ynn_v));

ynn1_v_p = ynn_v_r';

ncA = 5; ncB = 5; ncC = 5; ncD = 10;
nhA = 6; nhB = 8; nhC = 10; nhD = 12; nhW = 2;
noA = 0; noB = 1; noC = 2; noD = 0; noW = 1;

CA_NN_out_r = ynn1_v_p(:,1); CB_NN_out_r = ynn1_v_p(:,2);
CC_NN_out_r = ynn1_v_p(:,3); CD_NN_out_r = ynn1_v_p(:,4);

c1 = [CA_NN_out_r;CB_NN_out_r;CC_NN_out_r;CD_NN_out_r];

% Calculation of number of moles of water in Outlet Stream (kmol/min)
mol_W_out = zeros(tv,1);
for i = 1:tv    
    % Calculation of number of moles of water used up in reaction (kmol/min)
    mol_AtoB = (fv_in_v(i,1)*V)*(CAf_in_v(i,1) - CA_NN_out_r(i,1) - 2*(CD_NN_out_r(i,1)-CDf_in_v(i,1)));            % Number of moles of water used up when A reacted to form B
    C_Bformed = (CAf_in_v(i,1) - CA_NN_out_r(i,1) - 2*(CD_NN_out_r(i,1)-CDf_in_v(i,1)));
    mol_BtoC = (fv_in_v(i,1)*V)*(CBf_in_v(i,1) + C_Bformed - CB_NN_out_r(i,1));                                 % Number of moles of water used up when B reacted to form C
    
    mol_W_out(i,1) = mol_W_in_v(i,1) - (mol_AtoB + mol_BtoC);
end

% Calculation of number of moles of A,B,C,D in Outlet Stream (kmol/min)
mol_A_out = V*fv_in_v.*CA_NN_out_r; mol_B_out = V*fv_in_v.*CB_NN_out_r;
mol_C_out = V*fv_in_v.*CC_NN_out_r; mol_D_out = V*fv_in_v.*CD_NN_out_r;

% Total number of atoms of C and H in Outlet Stream

at_C_out_v = mol_A_out*ncA + mol_B_out*ncB + mol_C_out*ncC + mol_D_out*ncD;
at_H_out_v = mol_A_out*nhA + mol_B_out*nhB + mol_C_out*nhC + mol_D_out*nhD + mol_W_out*nhW;
at_O_out_v = mol_A_out*noA + mol_B_out*noB + mol_C_out*noC + mol_D_out*noD + mol_W_out*noW;

c2 = [at_C_in_v - at_C_out_v; at_H_in_v - at_H_out_v; at_O_in_v - at_O_out_v]; 

c = [c1;c2];

end