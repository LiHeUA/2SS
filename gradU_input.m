function dFdUij = gradU_input(data,cen,sigma,TBBT,KBBT)

[n, dim] = size(data);
k = size(cen,1);

dis = pdist2(data,cen);
Kz = exp(-dis.^2/2/sigma^2);

%% d g(U)/g (X_ij) = Tr( (d g(U)/d (U))^T d (U)/d X_ij )
% dFdKz = 2*(Kz*T-K)*BBT;
% dFdKz: n*k
dFdKz = Kz*TBBT-KBBT;

%% d Kz / d U_ij
% dKzdUij: n*k

% Easy to understand but low efficiency
% x = data';
% z = cen';
% 
% dFdUij = zeros(dim,k);
% 
% for i=1:dim
%     for j=1:k
%         dKzdUij = zeros(n,k);
%         for p=1:n
%             dKzdUij(p,j) = Kz(p,j)*(x(i,p)-z(i,j))/sigma^2;
%         end
%         dFdUij(i,j) = trace( dFdKz'*dKzdUij );
%     end
% end
% dFdUij = dFdUij';

% high efficiency but not easy to understand
dFdUij = zeros(dim,k);
for j=1:k
    b = bsxfun(@minus,data,cen(j,:));
    a = Kz(:,j).*b/sigma^2;
    dFdUij(:,j) = a'*dFdKz(:,j);
end

dFdUij = dFdUij';

