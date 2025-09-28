function [cen, label, F,iter] = optimization_loop_2SS(cen0,labelKM,K1, V1, X1, sigma, maxIter, dataTrain)
% This the main loop of Alg. 1 of [1]
%
% [1] Li He and Hong Zhang, Two-Step Nystrom Sampling for Large-scale 
% Kernel Approximation, to appear in IEEE Transactions on Big Data
%
% hel@sustech.edu.cn
% Sept. 28, 2025

if nargin<7
    maxIter = 10;
    dataTrain = X1;
end

m1 = size(X1,1);

label = labelKM;

if ~exist('maxIter','var')
    maxIter = 10;
end

thr_cen = 1e-3; % convergence threshold 
diff_cen = 1e5; % initialize convergence error

alpha0 = 1e-3; % gradient descent step length

[~,~,last(:)] = unique(label);   % remove empty clusters
F = sparse(last,1:m1,1); % X1-to-Z membership indicator matrix

VVF = V1*(V1'*F');
P = 2*F*VVF; % P in Alg. 2
Q = 2*K1*VVF; % Q in Alg. 2

iter = 1;
cen = cen0;
% Main loop
while diff_cen>thr_cen && iter<=maxIter
    last = cen;
    
    % Alg. 2. dFdZij is $\partial f(K_Z)/\partial z_{ij}$
    dFdZij = gradZ_input_Fast(dataTrain,cen,sigma,P,Q);

    sfactor = norm(dFdZij,'fro')/norm(cen,'fro');

    if sfactor>1
        alpha = alpha0/sfactor;
    else 
        alpha = alpha0;
    end
    % gradient descent update
    cen = cen - alpha*dFdZij;
    
    diff_cen = norm(last-cen,'fro')/norm(last,'fro');
    iter = iter+1;
end