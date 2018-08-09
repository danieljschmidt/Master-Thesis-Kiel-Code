function y_transf = transform_y(y, psi)
% returns quasi-differenced data given the AR coefficients psi
% y_transf(:,1): the first q elements in y(:,1) are deleted but not trafo
% y_transf(t,i)= y(t,i) - psi_{i,1} y_{t-1,i} - ... - psi_{i,q) y_{t-q,i}
% for i=2:n, t=1:T
%
% function inputs:
% y  : (T+q) x n
% psi:     n x q
%
% function outputs:
% y_transf: T x n
%

[~, q] = size(psi);
[T1, n] = size(y);
T = T1 - q;

Psi = zeros((n-1)*q,n-1);
for i=1:q
    Psi((i-1)*(n-1)+1:i*(n-1),:) = diag(psi(2:n,i));
end

y_transf = nan(T,n);
y_transf(:,1)      = y(1+q:end,1);
y_transf(:,2:n)    = y(1+q:end,2:n) - lags(y(:,2:n),q)*Psi;

end

