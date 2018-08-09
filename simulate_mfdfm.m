function [y, f, u] = simulate_mfdfm(T, mu, lambda, phi, sigma2v, ...
    psi, sigma2w)
%
% returns a realization of the following mixed-frequency dynamic factor
% model (1st variable is quarterly, all others monthly)
%
% y1_t  = 1/3 y1_t* + 2/3 y1_{t-1}* + y1_{t-2}* + 2/3 y1_{t-3}* + ...
%         1/3 y1_{t-4}*
% y1_t* = mu_1 + lambda_1 f_t + u1_t
% yi_t  = mu_i + lambda_i f_t + ui_t        for i=2:n
% ui_t  = psi_{i,1} ui_{t-1} + ... + psi_{i,q} ui_{t-q} + w1_t
% f_t   = phi_1 f_{t-1}  + ... + phi_p f_{t-q}  + v_t
% wi_t ~ Normal(0, sigma2w_i), v_t ~ Normal(0, sigma2v)
%
% NaNs for y1_t are not substituted for months when the quarterly variable
% is not observed!
%
% function inputs:
% T:       1 x 1
% mu:      n x 1
% lambda:  n x 1
% phi:     1 x p
% sigma2v: 1 x 1
% psi:     n x q
% sigma2w: n x 1
%
% function output:
%
% y: T   x n
% f: T+4 x 1 (contains factors for t=-3 to t=T)
% u: T+4 x n (contains error terms for t=-3 to t=T)

n = length(mu);

f = simulate_arp(T+4, 0, phi, sigma2v);

u = zeros(T+4, n);
for i=1:n
    u(:,i) = simulate_arp(T+4, 0, psi(i,:), sigma2w(i));
end

y = zeros(T, n);
y(:,1) = 3*mu(1) + lambda(1)* ...
      [f(5:T+4)   f(4:T+3)   f(3:T+2)   f(2:T+1)   f(1:T)  ]*[1;2;3;2;1]/3 ...
    + [u(5:T+4,1) u(4:T+3,1) u(3:T+2,1) u(2:T+1,1) u(1:T,1)]*[1;2;3;2;1]/3;
for i=2:n
    y(:,i) = mu(i) + lambda(i)*f(5:T+4) + u(5:T+4,i);
end

end