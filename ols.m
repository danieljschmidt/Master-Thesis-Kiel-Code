function [mu, invsigma2] = ols(X, y, sigma2)
%
% returns the OLS estimate mu and the corresponding inverse asymptotic
% variance invsigma2 for given data X, y and sigma2
% (also relevant for Bayesian estimation of linear regression paramters with
% Jeffrey's prior)
%
% function inputs:
% X: T x n
% y: T x 1
% sigma2: 1 x 1
%
% function outputs:
% mu: n x 1
% invsigma2: n x n
%

mu        = (X'*X)\(X'*y);
invsigma2 = X'*X/sigma2;

end

