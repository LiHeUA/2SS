function [cen, label, T,iter] = kernpca_weightedClustering(cen0,labelKM,K1, V1, X1, sigma, maxIter, data)
if nargin<7
    maxIter = 1000;
    data = X1;
end

m1 = size(X1,1);
m2 = size(cen0,1);

B = V1;

label = labelKM;


%% New Code
cen = cen0;

iter = 1;
if ~exist('maxIter','var')
    maxIter = 50;
end

thr_cen = 1e-50;
diff_cen = 1e5;

alpha0 = 1e-3;

[~,~,last(:)] = unique(label);   % remove empty clusters
T = sparse(last,1:m1,1);

BBT = B*(B'*T');
TBBT = 2*T*BBT;
KBBT = 2*K1*BBT;


while diff_cen>thr_cen && iter<=maxIter
    last = cen;
    
    [dFdUij] = gradU_input(data,cen,sigma,TBBT,KBBT);

    sfactor = norm(dFdUij,'fro')/norm(cen,'fro');

    if sfactor>1
        alpha = alpha0/sfactor;
    else 
        alpha = alpha0;
    end
    cen = cen - alpha*dFdUij;
    
    diff_cen = norm(last-cen,'fro')/norm(last,'fro');
    iter = iter+1;
end