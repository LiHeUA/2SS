function [cen2SS, FV1] = TwoStepNystromSampling(X1, sigma, dim, dataTrain, m2)
% This is the main function of two-step Nystrom sampling, or 2SS. 
%
% [1] Li He and Hong Zhang, Two-Step Nystrom Sampling for Large-scale 
% Kernel Approximation, to appear in IEEE Transactions on Big Data
%
% hel@sustech.edu.cn
% Sept. 28, 2025

% learn V1
dis = pdist2(X1,X1);
Ke = exp(-dis.^2/2/sigma^2);
[V1,~] = eigs(Ke,dim,'LM');

% K1 and K1*V1
dis = pdist2(dataTrain,X1);
K1 = exp(-dis.^2/2/sigma^2);

% initialize the small landmark set, or z in Alg. 1 of [1]
% In Alg. 1 of [1], cenKM2SS is represented as z
[~,cenKM2SS] = kmeans(dataTrain,m2,'EmptyAction','singleton');
% remove empty labels
labelLast = knnsearch(cenKM2SS, X1);
idxRemove = setdiff(1:m2,labelLast);
idx = true(m2,1);
idx(idxRemove) = false;
cenKM2SS = cenKM2SS(idx,:);

maxIter = 10;
% main loop in Alg. 1 of [1]
[cen2SS,~, F] = optimization_loop_2SS(cenKM2SS,labelLast,K1, V1, X1, sigma,maxIter,dataTrain);
FV1 = F*V1;