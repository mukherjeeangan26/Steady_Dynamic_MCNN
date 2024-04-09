function fun = InvProbV2(x,Imat_t,dsr_t,ni,nh,no,tn,delta,data)

wh = reshape(x(1:ni*nh),[ni,nh]);
wo = reshape(x(ni*nh+1:ni*nh+nh*no),[nh,no]);
bh = (x(ni*nh+nh*no+1:ni*nh+nh*no+nh))';
bo = (x(ni*nh+nh*no+nh+1:ni*nh+nh*no+nh+no))';
y_r = reshape(x(ni*nh+nh*no+nh+no+1:ni*nh+nh*no+nh+no+no*tn),[tn,no]);

a = x(ni*nh+nh*no+nh+no+no*tn+1:ni*nh+nh*no+nh+no+no*tn+no);

bhmat = zeros(nh,tn);
for i = 1:size(bhmat,2)
    bhmat(:,i) = bh;
end

bomat = zeros(no,tn);
for i = 1:size(bomat,2)
    bomat(:,i) = bo;
end

y1 = Imat_t;
x1 = wh'*y1 + bhmat;
y2 = logsig(x1);
x2 = wo'*y2 + bomat;
yNN = purelin(x2);

dsr_t_p = zeros(tn,no);
ynn1_t_p = zeros(tn,no);

for i = 1:no
    dsr_t_p(:,i) = (dsr_t(i,:))'.*delta(1,ni+i) + min(data(:,ni+i));
    ynn1_t_p(:,i) = yNN(i,:)'.*delta(1,ni+i) + min(data(:,ni+i));
end

y_str = [];

for i = 1:no
    y_str = [y_str, a(i).*y_r(:,i)];
end

fun = (sum(((dsr_t_p - (y_r+y_str))'*eye(tn)*(dsr_t_p - (y_r+y_str))),'all') + sum(((y_r - ynn1_t_p)'*eye(tn)*(y_r - ynn1_t_p)),'all'));

end

