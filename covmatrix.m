function Sigma = covmatrix(phi, lags)
% returns the covariance matrix of the following process
% xi_t = 1/3*(u_t + 2*u_{t-1} + 3*u_{t-2} + 2*u_{t-3} + u_{t-4})
% where u_t  = phi_1 u_{t-1} + ... + phi_p u_{t-p} + epsilon_t
% epsilon_t ~ Normal(0, sigma2)
% up to the multiplicative factor sigma2 (!)
% for example, lags = 0:3:27 yields the covariance matrix for [u_3, u_6,
% u_9, u_12, ... u_30]
% phi needs to be a row vector!

maxlag = max(lags);
N = length(lags);

% calculate the necessary autocovariance of u_t
a = autocov_arp(phi, maxlag+4);

% calculate the autocovariance of xi_t at the lag lengths given by lags
autocovs = zeros(N+1,1);
for i=1:N
    l = lags(i);
    autocovs(i) = 1/9*(19*a(l+1) + 16*a(l+1+1) + 16*a(abs(l-1)+1) + ...
        10*a(l+2+1) + 10*a(abs(l-2)+1) + 4*a(l+3+1) + 4*a(abs(l-3)+1) + ...
        a(l+4+1) + a(abs(l-4)+1));
end

% construct a covariance matrix out of the autocovariances
Sigma = zeros(N);
for i=1:N
    for j=1:N
        Sigma(i,j) = autocovs(abs(i-j)+1);
    end
end

end