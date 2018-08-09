function y = simulate_arp(T, mu, phi, sigma2)
% returns the realization y (T x 1) of an AR(p) process
% y_t - mu = phi_1 (y_{t-1}-mu) + ... + phi_p (y_{t-p}-mu) + eps_t
% where eps_t ~ Normal(0, sigma2)
% phi is a row vector

p = length(phi);
sigma = sqrt(sigma2);

% get the covariance matrix for the initial p values
Phi = [phi; eye(p-1) zeros(p-1,1)];
Sigma1p = (eye(p^2)-kron(Phi, Phi))\[1; zeros(p^2-1,1)];
Sigma1p = reshape(Sigma1p,p,p);

% simulate AR(p) process
u = zeros(T,1);
u(1:p) = sigma*chol(Sigma1p, 'lower')*randn(p,1);
for t=p+1:T
    u(t) = phi*u(t-(1:p)) + sigma*randn(); 
end
y = mu + u;

end

