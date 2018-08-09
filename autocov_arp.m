function autocovs = autocov_arp(phi, maxlag)
% returns the autocovariances of an AR(p) process
% y_t = phi_1 y_{t-1} + ... + phi_p y_{t-p} + epsilon_t
% epsilon_t ~ Normal(0, sigma2)
% for lags 0 to maxlag up to the multiplicative factor sigma2 (!)
% note that the autocovariance for lag length i at the i+1th element of
% the returned vector
% phi needs to be a row vector!
% see Hamilton, chapter 3.4, p. 59

p = length(phi);
autocovs = zeros(1+maxlag,1);

% step 1: autocovariance for lags 0 to p-1
F = [phi; eye(p-1) zeros(p-1,1)];
e = [1; zeros(p^2-1,1)];
Sigma1p_vec = (eye(p^2) - kron(F, F))\e;
autocovs(1:p) = Sigma1p_vec(1:p);

% step 2: autocovariance for lags p to maxlag with Yule-Walker equations
for lag=p:maxlag
    autocovs(lag+1) = phi*autocovs(lag:-1:lag-p+1);
end

end