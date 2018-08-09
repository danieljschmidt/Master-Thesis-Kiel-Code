function [h_all, mu_all, phi_all, sigma2_all] = gibbs_sv_centered(yt, S, mu0, phi0, sigma20)
% draws samples from the posterior of a stochastic volatility model
% using the centered specification: 
% y(t) = exp(h(t)/2)*eps(t)
% h(t) = mu + phi*(h(t-1)-mu) + sigma*eta(t)
% eps(t) ~ N(0,1) and eta(t) ~ N(0,1)
% centered parametrization is inefficient for phi close to 0 and small
% sigma2
% the centered and the not-centered specifications are related by
% h_nc = (h_c - mu)/sigma
% yt = log(y.^2) where y are e.g. stock returns returns
% S is the number of samples
% mu0, phi0 and sigma20 are initial values for mu, phi and sigma2

T = size(yt,1);

% approximation of log(eps^2) for eps ~ N(0,1) as Normal mixture as in 
% Omori et al. (2007)

p  = [0.00609, 0.04775, 0.13057, 0.20674,  0.22715,  0.18842, ...
    0.12047,  0.05591,  0.01575,   0.00115]';
m  = [1.92677, 1.34744, 0.73504, 0.02266, -0.85173, -1.97278, ...
    -3.46788, -5.55246, -8.68384, -14.65000]';
s2 = [0.11265, 0.17788, 0.26768, 0.40611,  0.62699,  0.98583, ...
    1.57469,  2.54498,  4.16591,   7.33342]';

% initialize storage matrices

h_all      = zeros(S,T);
mu_all     = zeros(S,1);
phi_all    = zeros(S,1);
sigma2_all = zeros(S,1);

% initialize r

W = p*ones(1,T);
r = catrnd(W);

% initialize mu, phi and sigma2 with values given as function arguments

mu     = mu0;
phi    = phi0;
sigma2 = sigma20;

a_prior = 1.;
b_prior = 1.;

for s=1:S

    % draw h - centered

    c = (yt-m(r))./s2(r) + mu*(1-phi)^2/sigma2;
    c(1) = (yt(1)-m(r(1)))/s2(r(1)) + mu*(1-phi)/sigma2;
    c(T) = (yt(T)-m(r(1)))/s2(r(T)) + mu*(1-phi)/sigma2;

    Omega0 = 1./s2(r) + (1+phi^2)/sigma2;
    Omega0(1) = 1/s2(r(1)) + 1/sigma2;
    Omega0(T) = 1/s2(r(T)) + 1/sigma2;
    Omega1 = -phi/sigma2*ones(T,1);
    Omega = spdiags([Omega1 Omega0 Omega1],[-1,0,1],T,T);

    OmegaL = chol(Omega, 'lower');
    a = OmegaL\c;
    eps = randn(T,1);
    h = OmegaL'\(a+eps);
    
    % draw phi
    h2 = h - mu;
    mu_phi     = h2(1:T-1)'*h2(2:T)/(h2(1:T-1)'*h2(1:T-1));
    sigma2_phi = sigma2/(h2(1:T-1)'*h2(1:T-1));
    phi        = mu_phi + sqrt(sigma2_phi)*randn();
    
    while abs(phi) > 1
        phi        = mu_phi + sqrt(sigma2_phi)*randn();
    end
    
    % draw mu
    
    mu_mu     = mean(h);
    sigma2_mu = sigma2/T;
    mu = mu_mu + sqrt(sigma2_mu)*randn();
    
    % draw sigma2
    
    u_sigma2 = h(2:T) - mu - phi*(h(1:T-1)-mu);
    a_sigma2 = a_prior + (T-1)/2;
    b_sigma2 = 1/(1/b_prior + u_sigma2'*u_sigma2/2);
    sigma2 = 1/gamrnd(a_sigma2, b_sigma2);
    
    % draw r
    
    eps = yt - h;
    logW = log(p) - 1/2*log(s2) - (eps'-m).^2./(2*s2);
    W = exp(logW - max(logW));
    r = catrnd(W);
    
    % store everything
    
    h_all(s,:)      = h;
    mu_all(s,:)     = mu;
    phi_all(s,:)    = phi;
    sigma2_all(s,:) = sigma2;
    
end

end % function

