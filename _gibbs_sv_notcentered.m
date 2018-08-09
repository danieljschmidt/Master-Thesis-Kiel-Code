function [h_all, mu_all, phi_all, sigma2_all] = gibbs_sv_notcentered(yt, S, mu0, phi0, sigma20)
% draws samples from the posterior of a stochastic volatility model
% using the not-centered specification: 
% y(t) = omega*exp(sigma*h(t)/2)*eps(t) where omega = exp(mu/2)
% h(t) = phi*h(t-1) + eta(t)
% eps(t) ~ N(0,1) and eta(t) ~ N(0,1)
% not-centered parametrization is inefficient for phi close to 1 and large
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

for s=1:S

    % draw h - centered

    c = sqrt(sigma2)./s2(r).*(yt-m(r)-mu);
    
    Omega0 = sigma2./s2(r) + 1 + phi^2;
    Omega0(1) = sigma2./s2(r(1)) + 1;
    Omega0(T) = sigma2./s2(r(T)) + 1;
    Omega1 = -phi*ones(T,1);
    Omega = spdiags([Omega1 Omega0 Omega1],[-1,0,1],T,T);

    OmegaL = chol(Omega, 'lower');
    a = OmegaL\c;
    eps = randn(T,1);
    h = OmegaL'\(a+eps);
    
    % transform non-centered h to centered h
    
    h = mu + h*sqrt(sigma2);
    
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

