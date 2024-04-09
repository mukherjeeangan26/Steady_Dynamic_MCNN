function fun = ForwProbV1(x,ynn_v,tv,no)

ynn_v_r = x(1:no*tv);
ynn_v_r = reshape(ynn_v_r,size(ynn_v));

fun = sum((ynn_v - ynn_v_r).^2,'all');

end