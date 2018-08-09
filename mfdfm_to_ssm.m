function [muSSM, A, B, Sigma_eps, R_eps, Sigma_eta, R_eta, mu_xf1, Sigma_xf1] ...
    = mfdfm_to_ssm(mu, lambda, phi, sigma2v, psi, sigma2w)
%
% returns the parameters of the state space representation 
%
% x_t = muSSM + A x_{t-1} + R_eps eps_t
% y_t = B x_t + R_eta eta_t
% x_1   ~ Normal(mu_xf1, Sigma_xf1)
% eps_t ~ Normal(0, Sigma_eps)
% eta_t ~ Normal(0, Sigma_eta)
%
% for the mixed-frequency dynamic factor model
%
% y1_t  = 1/3 y1_t* + 2/3 y1_{t-1}* + y1_{t-2}* + 2/3 y1_{t-3}* + ...
%         1/3 y1_{t-4}*
% y1_t* = mu_1 + lambda_1 f_t + u1_t
% yi_t  = psi_i(1) mu_i + lambda_i psi_i(L) f_t + wi_t        for i=2:n
% u1_t  = psi_1 u1_{t-1} + ... + psi_q u1_{t-q} + w1_t
% f_t   = phi_1 f_{t-1}  + ... + phi_p f_{t-q}  + v_t
% wi_t ~ Normal(0, sigma2w_i), v_t ~ Normal(0, sigma2v)
%
% note that yi_t are the quasi-differenced actual observations such that
% the error term in the measurement equation is no longer autocorrelated
%
% function input:
%
% mu:      n x 1
% lambda:  n x 1
% phi:     1 x p
% sigma2v: 1 x 1
% psi:     n x q
% sigma2w: n x 1
%
% function output:
%
% muSSM:       n x   1
% A:          10 x  10
% B:           n x  10
% Sigma_eps:   2 x   2
% R_eps:      10 x   2
% Sigma_eta: n-1 x n-1
% R_eta:     n   x n-1
% mu_xf1:     10 x   1
% Sigma_xf1:  10 x  10
%

[~,q] = size(psi); % n x q
[~,p] = size(phi); % 1 x p

n  = length(lambda);
muSSM = [3*mu(1); (1-sum(psi(2:end,:),2)).*mu(2:end)];

A1   = [phi zeros(1,5-p); eye(4,4) zeros(4,1)];
A2  = [psi(1,:) zeros(1,5-q); eye(4,4) zeros(4,1)];
A    = blkdiag(A1, A2);

b1q  = lambda(1)/3*[1 2 3 2 1];
b2q  = 1/3*[1 2 3 2 1];
la  = reshape(lambda(2:end),n-1,1);
B    = [b1q b2q; la -psi(2:end,:).*la zeros(n-1,4-q) zeros(n-1,5)];

R_eps = [1 0; zeros(4,2); 0 1; zeros(4,2)];
Sigma_eps = diag([sigma2v; sigma2w(1)]);

R_eta = [zeros(1,n-1); eye(n-1)];
Sigma_eta = diag(sigma2w(2:end)); 

mu_xf1 = zeros(10,1);

% covariance matrix for [f_1, f_0, f_{-1}, f_{-2}, f_{-3}]
F = [phi zeros(1, 5-p); eye(4) zeros(4,1)];
e = [1; zeros(5^2-1,1)];
Sigma1p_vec = (eye(5^2) - kron(F, F))\e;
part1 = sigma2v*reshape(Sigma1p_vec,5,5);

% covariance matrix for [u1_1, u1_0, u1_{-1}, u1_{-2}, u1_{-3}]    
F = [psi(1,:) zeros(1, 5-q); eye(4) zeros(4,1)];
e = [1; zeros(5^2-1,1)];
Sigma1p_vec = (eye(5^2) - kron(F, F))\e;
part2 = sigma2v*reshape(Sigma1p_vec,5,5);

Sigma_xf1 = blkdiag(part1, part2);
    
end

