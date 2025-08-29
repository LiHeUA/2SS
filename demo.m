function demo
clc
clear

%% 0. Initialization
% CCC_Capital data set. Please refer to [38] or https://www.idiap.ch/en/scientific-research/data/ccc/
load('CCC_Capital.mat');
data = dataCap;
numData = size(data,1);
label = idxCap;
% dimension of embeddings, or the number of centers
dim = length(unique(label));

% randomly select 1K points to generate the ground truth. In the paper, 8K
sizeTraining = 1000;
idx = randperm(numData);
dataTrain = data(idx(1:sizeTraining),:);

% set Gaussian scale parameter sigma
dis = pdist2(dataTrain,dataTrain);
sigma = median(dis(:));


% Kernel matrix and its eigen-decomposition of training data
Ke = exp(-dis.^2/2/sigma^2);
[vecGT,val] = eig(Ke);
[~,idx] = sort(diag(val),'descend');
vecGT = vecGT(:,idx);
vecGT = vecGT(:,1:dim);
% Kernel matrix between the training data and the entire data
dis = pdist2(data,dataTrain);
K1 = exp(-dis.^2/2/sigma^2);
% Take the embeddings of the 8K points as the ground truth
embGT = K1*vecGT;


% size of the large landmark set, m1
m1 = 200;
% size of the small landmark set, m2
m2 = 30;



%% 1. k-means Nystrom
[~,cenKM] = kmeans(dataTrain,m1,'EmptyAction','singleton');
embKM = StandardNyst(data, cenKM, sigma, dim);
errKM = runCalErr(embGT, embKM);


%% 2. Our 2SS
% Taking k-means centers as the initial landmarks
X1 = cenKM;
dis = pdist2(X1,X1);
% learn V1
Ke = exp(-dis.^2/2/sigma^2);
[V1,~] = eigs(Ke,dim,'LM');

% K1 and K1*V1
dis = pdist2(dataTrain,X1);
K1 = exp(-dis.^2/2/sigma^2);

% 
[~,cenKM2SS] = kmeans(dataTrain,m2,'EmptyAction','singleton');
% remove empty labels
labelLast = knnsearch(cenKM2SS, X1);
idxRemove = setdiff(1:m2,labelLast);
idx = true(m2,1);
idx(idxRemove) = false;
cenKM2SS = cenKM2SS(idx,:);

% main function
[cen2SS,~, T] = kernpca_weightedClustering(cenKM2SS,labelLast,K1, V1, X1, sigma,5,dataTrain);
TV1 = T*V1;

% Embedding
emb2SS = run2SSEmb(data, cen2SS, TV1, sigma);
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

% Our embedding process is different from the standard Nystrom
function emb = run2SSEmb(data, cenOur, TV1, sigma)
dis = pdist2(data,cenOur);
Kz = exp(-dis.^2/2/sigma^2);
emb = Kz*TV1;