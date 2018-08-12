function [phi0, lambda0, psi0, sigma2w0] = gibbs_mfdfm_init(y, p, q)
% TODO documentation
% TODO think more about how to deal with the first, quarterly variable

[T1, n] = size(y);

T2 = T1 - sum(any(isnan(y(:,2:n)),2)); % last index without NaN in row
f0 = mean(y(1:T2,2:10)./std(y(1:T2,2:10)),2);

lambda0 = randn(n,1);

for i=2:n
    [lambda0(i), ~] = ols(f0,y(1:T2,i),NaN);
end
lambda0(1) = lambda0(2);

[phi0, ~] = ols(lags(f0,p), f0(p+1:end), NaN);
phi0      = phi0';

sigma2v   = var(f0(p+1:end) - lags(f0,p)*phi0');
f0        = f0/sqrt(sigma2v);
lambda0   = lambda0*sqrt(sigma2v);

psi0    = zeros(n,q);
sigma2w0 = zeros(n,1);
for i=2:n
    u_i         = y(1:T2,i) - f0*lambda0(i);
    [psi0_i, ~] = ols(lags(u_i,q),u_i(q+1:end), NaN);
    w_i         = u_i(q+1:end) - lags(u_i,q)*psi0_i;
    sigma2w0(i)  = var(w_i);
    psi0(i,:)   = psi0_i';
end
psi0(1,:)  = zeros(1,q);
sigma2w0(1) = var(y(q+3:3:end-10,1)); % extremely sloppy

end