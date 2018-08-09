function [x_all, phi_all, mu_all, lambda_all, psi_all, sigma2w_all] = ...
    gibbs_mfdfm(y, S, p, q, ...
    phi0, mu0, lambda0, psi0, sigma2w0)
%
% returns MCMC samples for a mixed-frequency dynamic factor model
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
% TODO: describe tricks at the beginning of the sample
%
% sigma2v = 1 fixed in order to get rid of the undefined scale of lambda
% and f_t
%
% priors:
%
% p(lambda_i) \propto 1
% p(mu_i)     \propto 1
% p(phi)      \propto 1_{phi is stationary}
% p(psi_i)    \propto 1_{psi_i is stationary}
% p(sigma2w_i)\propto (sigma2w_i)^{-1/2}
%
% function input:
% 
% y : T+q x n (not quasi-differenced, NaNs at the end of the sample allowed)
% S :   1 x 1
% p :   1 x 1
% q :   1 x 1
%
% phi0    : 1 x p
% mu0     : n x 1
% lambda0 : n x 1
% psi0    : n x q
% sigma2w : n x 1
%
% function output:
%
% x_all       : S x T x 10
% phi_all     : S x p
% mu_all      : S x n
% lambda_all  : S x n
% psi_all     : S x n x q
% sigma2w_all : S x n
%
% problems:
% - if psi is close to a unit root, mu gets driven to large values
%

% TODO check that data is well-behaved

if q > 4
    error("q is larger than 4")
end

if p > 4
    error("p is larger than 4")
end

% maximum number of times that I try to sample from a truncated Normal 
% before an error message appears
maxtimes = 10000; 

[T1, n] = size(y);
T = T1 - q;

% fixed parameters
sigma2v = 1.;

% initialize storage matrices
x_all       = zeros(S,T,10);
phi_all     = zeros(S,p);
mu_all      = zeros(S,n);
lambda_all  = zeros(S,n);
psi_all     = zeros(S,n,q);
sigma2w_all = zeros(S,n);

% initialize parameters
phi     = phi0;     % 1 x p
mu      = mu0;      % n x 1
lambda  = lambda0;  % n x 1
psi     = psi0;     % n x q
sigma2w = sigma2w0; % n x 1

% quasi-difference y for i=2:n
y_transf = transform_y(y, psi);

T_obs    = sum(~isnan(y_transf));
idx      = 1:T;
y1_obs   = idx(~isnan(y_transf(:,1)));

for s=1:S

    % - - - - - - - - - - - - simulation smoothing - - - - - - - - - - - - 
    
    % (Durbin and Koopman, 2002, p. 67, y*-method

    [muSSM, A, B, Sigma_eps, R_eps, Sigma_eta, R_eta, ...
    mu_xf1, Sigma_xf1] = ...
    mfdfm_to_ssm(mu, lambda, phi, sigma2v, psi, sigma2w);

    [x_sim, y_transf_sim] = simulate_ssm(...
    T, muSSM, A, B, Sigma_eps, R_eps, Sigma_eta, R_eta, ...
    mu_xf1, Sigma_xf1);
    
    % TODO: replace Sigma_xf1 with eye(10) ??

    y_transf_sim(isnan(y_transf)) = nan;

    y_star = y_transf - y_transf_sim;

    xs_star    = fast_state_smoothing(y_star, ...
    zeros(n,1), A, B, Sigma_eps, R_eps, Sigma_eta, R_eta, mu_xf1, eye(10));
    % muSSM = zeros(n,1) for y*-method!

    x = xs_star + x_sim;

    % - - - - - - - - - - - - - draw psi for i=1 - - - - - - - - - - - - - 

    X_psi    = x(:,6+1:6+q);
    y_psi    = x(:,6);
    [mu_psi, invSigma2_psi] = ols(X_psi, y_psi, sigma2w(1));
    psi(1,:) = normrnd_stationary(mu_psi, invSigma2_psi, maxtimes);
    
    % - - - - - - - - - - - - - - - draw phi - - - - - - - - - - - - - - - 

    X_phi = x(2:end,2:p+1);
    y_phi = x(2:end,1);
    [mu_phi, invsigma2_phi] = ols(X_phi, y_phi, sigma2v);
    phi   = normrnd_stationary(mu_phi, invsigma2_phi, maxtimes)';
    % tranposing is important!
    
    %  - - - - - - - - - draw mu, lambda, sigma2w for i=1 - - - - - - - - -
        
    y_1           = y_transf(y1_obs,1);
    X_1           = [3*ones(T_obs(1),1), x(y1_obs,1:5)*[1;2;3;2;1]/3];
    covm          = covmatrix(psi(1,:), y1_obs-y1_obs(1));
    
    invSigma_beta = 1/sigma2w(1)*X_1'*(covm\X_1);
    mu_beta       = invSigma_beta\(1/sigma2w(1)*X_1'*(covm\y_1));
    beta          = normrnd2(mu_beta, invSigma_beta);
    mu(1)         = beta(1); 
    lambda(1)     = beta(2);
    
    u_1        = y_1 - X_1*beta;
    sigma2w(1) = 1/gamrnd(T_obs(1)/2, 2/(u_1'*(covm\u_1)));

    % - - - - - - - - - draw mu, lambda, sigma2w for i=2:n- - - - - - - - - 

    for i=2:n
        
        T_i    = T_obs(i);
        psiL_i = [1 -psi(i,:)];
        y_i    = y_transf(1:T_i,i);
        X_i    = [sum(psiL_i)*ones(T_i,1),x(1:T_i,1:q+1)*psiL_i'];
        
        [mu_beta, invSigma2_beta] = ols(X_i, y_i, sigma2w(i));
        beta      = normrnd2(mu_beta, invSigma2_beta);
        mu(i)     = beta(1); 
        lambda(i) = beta(2);
        
        u_i        = y_i - X_i*beta;
        sigma2w(i) = 1/gamrnd(T_i/2, 2/(u_i'*u_i));
        
    end

    % - - - - - - - - - - - - - draw psi for i=2:n - - - - - - - - - - - - 

    for i=2:n

        X_psi = lags(y(  1:T_obs(i)+q,i),q) - mu(i) - lambda(i)*x(1:T_obs(i),2:q+1);
        y_psi =      y(q+1:T_obs(i)+q,i)    - mu(i) - lambda(i)*x(1:T_obs(i),1);
        [mu_psi, invSigma2_psi] = ols(X_psi, y_psi, sigma2w(i));
        psi(i,:) = normrnd_stationary(mu_psi, invSigma2_psi, maxtimes);

    end

    % quasi-difference y for i=2:n
    y_transf = transform_y(y, psi);
    
    % - - - - - - - - - - - - - - store draws - - - - - - - - - - - - - -
    
    x_all(s,:,:)    = x;
    phi_all(s,:)    = phi;
    mu_all(s,:)     = mu;
    lambda_all(s,:) = lambda;
    psi_all(s,:,:)  = psi;
    sigma2w_all(s,:)= sigma2w;
    
end

end

