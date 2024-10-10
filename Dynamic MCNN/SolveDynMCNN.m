function fun = SolveDynMCNN(x,Imat_t,dsr_t,ni,nh,no,tn,delta,data_dyn)

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

dsr_t_p = zeros(tn,no);
ynn1_t_p = zeros(tn,no);

for i = 1:no
    dsr_t_p(:,i) = (dsr_t(i,:))'.*delta(1,ni+i) + min(data_dyn(:,ni+i));
    ynn1_t_p(:,i) = ynn_t(:,i).*delta(1,ni+i) + min(data_dyn(:,ni+i));
end

fun = (1/(no*tn))*(sum(((dsr_t_p - y_r)'*eye(tn)*(dsr_t_p - y_r)),'all') + sum(((y_r - ynn1_t_p)'*eye(tn)*(y_r - ynn1_t_p)),'all'));

end