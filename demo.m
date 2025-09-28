function demo
% This is a demonstration of [1] that improves the Nystrom approximation
% performance.
%
% Nystrom approximation consists of two steps: low-rank decomposition and
% embedding.In Nystrom approximation, typically there is one landmark set. 
% Nystrom runs low-rank decompostion to get the eigenvectors of the 
% landmark set and embeds the entire data onto the eigenvectors. The 
% embedding time cost is the bottleneck for large-scale data set and is
% linearly with the size of the landmark set. 
%
% In [1], two landmark sets, a large one and a small-but-optimized one are
% built. The large landmark set is able to obtain accurate approximation, 
% with the cost of high time consumption in the embedding step.
% The small landmark set aims to have a similar accuracy as the large 
% landmark set but much lower time cost and memory requirement in the 
% embedding step.
%
% [1] Li He and Hong Zhang, Two-Step Nystrom Sampling for Large-scale 
% Kernel Approximation, to appear in IEEE Transactions on Big Data
%
% Li He, Dept. of Electronic and Electrical Engineering, Southern
% University of Science and Technology
%
% hel@sustech.edu.cn
% Sept. 28, 2025


% clc
% clear

%% 0. Initialization
% CCC_Capital data set. Please refer to [38] or https://www.idiap.ch/en/scientific-research/data/ccc/
load('CCC_Capital.mat');
data = dataCap;
numData = size(data,1); % number of entire data set
label = idxCap; % data labels

% dimension of embeddings, or the number of centers
dim = length(unique(label));

% randomly select 1K points to generate the ground truth. In the paper, 8K
sizeTraining = 1000;
idx = randperm(numData);
% in Alg. 1 of [1], dataTrain is represented as x
dataTrain = data(idx(1:sizeTraining),:); 

% set Gaussian scale parameter sigma as the median of data distances
dis = pdist2(dataTrain,dataTrain);
sigma = median(dis(:));


% Kernel matrix and its eigen-decomposition of training data
% Taking the eigenvectors of the training data as the ground truth
% eigenvectors
Ke = exp(-dis.^2/2/sigma^2);
[vecGT,val] = eig(Ke);
[~,idx] = sort(diag(val),'descend');
vecGT = vecGT(:,idx);
vecGT = vecGT(:,1:dim);
% Kernel matrix between the training data and the entire data
dis = pdist2(data,dataTrain);
K1 = exp(-dis.^2/2/sigma^2);
% Take the embeddings of the training points as the ground truth
embGT = K1*vecGT;

% size of the large landmark set, m1
m1 = 200;
% size of the small landmark set, m2
m2 = 30;

%% 1. k-means Nystrom
[~,cenKM] = kmeans(dataTrain,m1,'EmptyAction','singleton');
% k-means Nystrom is one SOTA method and takes k-means centers as the 
% landmark set for the standard Nystrom
embKM = StandardNyst(data, cenKM, sigma, dim); % standard Nystrom
errKM = runCalErr(embGT, embKM); % calculate embedding error


%% 2. Two-Step Nystrom Sampling, or 2SS
% Taking k-means centers as the initial landmarks
X1 = cenKM;
[cen2SS, FV1] = TwoStepNystromSampling(X1, sigma, dim, dataTrain, m2);

% Embedding
emb2SS = run2SSEmb(data, cen2SS, FV1, sigma);
err2SS = runCalErr(embGT, emb2SS);

disp(['Error of k-means is ' num2str(errKM) ', error of 2SS is ' num2str(err2SS)]);

% Nystrom Approximation
function emb = StandardNyst(pt, landmarks, sigma, dim)
% kernel matrix of training set
dis = pdist2(landmarks,landmarks);
K_c = exp(-dis.^2/2/sigma^2);

% kernel matrix of training set to entire data
dis = pdist2(pt,landmarks);
K_all2c = exp(-dis.^2/2/sigma^2);

[vec, val] = eig(K_c);
[val,idx] = sort(diag(val),'descend');
val = val(val>1e-7);
idx = idx(1:length(val));
vec = vec(:,idx);
vec = vec(:,1:dim);
emb = K_all2c*vec;

% Calculate Embedding Errors
function err = runCalErr(embGT, emb)
for i=1:size(embGT,2)
    if embGT(:,i)'*emb(:,i)<0
        emb(:,i) = -emb(:,i);
    end
end
err = norm(embGT-emb,'fro');

% Our embedding process is slightly different from the standard Nystrom
function emb = run2SSEmb(data, cen2SS, FV1, sigma)
dis = pdist2(data,cen2SS);
Kz = exp(-dis.^2/2/sigma^2);
emb = Kz*FV1;