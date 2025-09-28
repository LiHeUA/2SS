function dFdzij = gradZ_input_Fast(data,cen,sigma,P,Q)
% This is the main function of Alg. 2 of [1]
%
% [1] Li He and Hong Zhang, Two-Step Nystrom Sampling for Large-scale 
% Kernel Approximation, to appear in IEEE Transactions on Big Data
%
% hel@sustech.edu.cn
% Sept. 28, 2025

[n, dim] = size(data);
k = size(cen,1);

% Kz = ker(x, z)
% Here, data is represented as x, and cen as z, in Alg. 1 and 2 of [1]
dis = pdist2(data,cen);
Kz = exp(-dis.^2/2/sigma^2);

%% d f(Kz)/f(z_ij) = Tr( (d f(Kz)/d (Kz))^T * d (Kz)/d z_ij ), or Eq. (9)

%% dFdKz = 2*(Kz*F-K1)*V1V1'F', Eq. (10) or the 1st term in Eq. (16)
% dFdKz: n*k
dFdKz = Kz*P-Q;

%% d Kz / d z_ij
% dKzdUij: n*k

% One can use either a) fast but hard to understand or b) slow but easy to
% understand 
% a) Efficient but not easy to understand
dFdzij = zeros(dim,k);
hh = Kz.*dFdKz/sigma^2; % Eq. (16) execpt the second bracket
for j=1:k
    % the bsxfun part is the second bracket of Eq. (16)
    dFdzij(:,j) = bsxfun(@minus,data,cen(j,:))' * hh(:,j);
end
dFdzij = dFdzij';

% b) Easy to understand but low efficiency
% x = data';
% z = cen';
% 
% dFdzij = zeros(dim,k);
% 
% for i=1:dim
%     for j=1:k
%         dKzdUij = zeros(n,k);
%         for p=1:n
%             dKzdUij(p,j) = Kz(p,j)*(x(i,p)-z(i,j))/sigma^2;
%         end
%         dFdzij(i,j) = trace( dFdKz'*dKzdUij );
%     end
% end
% dFdzij = dFdzij';
