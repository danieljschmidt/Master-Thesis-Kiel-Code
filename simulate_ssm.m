function [x, y] = simulate_ssm(T, muSSM, A, B, Sigma_eps, R_eps, ...
    Sigma_eta, R_eta, mu_xf1, Sigma_xf1)
%
% returns simulated states x and observations y of the state space model
% 
% x_t = muSSM + A x_{t-1} + R_eps eps_t
% y_t = B x_t + R_eta eta_t
% x_1   ~ Normal(mu_xf1, Sigma_xf1)
% eps_t ~ Normal(0, Sigma_eps)
% eta_t ~ Normal(0, Sigma_eta)
%
% function inputs:
% muSSM:     n  x 1
% A:         k  x k
% B:         n  x k
% R_eps:     k  x k1
% Sigma_eps: k1 x k1
% R_eta:     n  x n1
% Sigma_eta: n1 x n1
% mu_xf1:    k  x 1
% Sigma_xf1: k  x k
%
% function outputs:
% x: T x k
% y: T x n
%

[~, k]  = size(B);
[n1, ~] = size(Sigma_eta);
[k1, ~] = size(Sigma_eps);

% sample the error terms
eta = randn(T, n1)*chol(Sigma_eta);
eps = randn(T-1, k1)*chol(Sigma_eps);

% sample the states x
x = zeros(T, k);
xc = mu_xf1 + chol(Sigma_xf1)'*randn(k,1);
x(1,:) = xc';
for t=2:T
    xc = A*xc + R_eps*eps(t-1,:)';
    x(t,:) = xc';
end

% sample y given x
y = ones(T,1)*muSSM'+x*B'+eta*R_eta';

end