function [ynn_t,nn_stat,nn_dyn] = SolveNLSNLD(Imat_t,dsr_t,ni,nh,no,tn)

maxiter = 2;

ynn_t = zeros(tn,no);
int_mat_1 = zeros(nh,tn);

check1 = Inf;

x_in = 0.95.*Imat_t;

net_s = newff([0 1].*ones(ni,1),[nh, nh],{'logsig','logsig'},'trainlm');

for iter = 1:maxiter
    
    % Training of the dynamic network using narxnet RNN
       
    p = x_in;
             
    X = tonndata(p,true,false);
    T = tonndata(dsr_t,true,false);

    trainFcn = 'trainbfg';  

    inputDelays = 0;
    feedbackDelays = 1;
    hiddenLayerSize = nh;
    netnarx_d = narxnet(inputDelays,feedbackDelays,hiddenLayerSize,'closed',trainFcn);

    netnarx_d.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
    netnarx_d.trainParam.epochs = 5000;
    netnarx_d.trainParam.max_fail = 1000;
    netnarx_d.layers{1}.transferFcn = 'tansig';
    netnarx_d.layers{2}.transferFcn = 'purelin';

    [x,xi,ai,t] = preparets(netnarx_d,X,{},T);

    netnarx_d.performFcn = 'msereg';  

    [netnarx_d,tr_d] = train(netnarx_d,x,t,xi,ai);

    y = netnarx_d(x,xi,ai);
    
    yn = zeros(no,tn-1);
    for i = 1:tn-1
        yn(1:no,i) = y{i};
    end
    
    ynn_t(1,:) = dsr_t(:,1);
      
    for i = 1:no
        ynn_t(2:tn,i) = (yn(i,:))';
    end
       
    % Newton's Method for Updating Initial Guess
          
    for i = 1:nh
        int_mat_1(i,:) = x_in(i,:);
    end
    
    % Training of Static Network
    
    net_s.trainParam.epochs = 5000;
    inp = Imat_t;
    
    [net_s,tr] = train(net_s,inp,int_mat_1);
    
    y_1 = sim(net_s,inp);
    
    for i = 1:nh
        x_in(i,:) = y_1(i,:);
    end
    
    sse = sum((dsr_t' - ynn_t).^2);
    mse = (1/(no*tn))*sum(sse);   
    
    if mse <= check1
        
        nn_stat = net_s;
        nn_dyn = netnarx_d;
        ynn_final = ynn_t;
        check1 = mse;
    end
end

ynn_t = ynn_final;

end
