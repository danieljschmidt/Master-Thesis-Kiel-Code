function i = inefficiency_factor(mcmc, maxlag)
% returns the inefficiency factor for the mcmc chain for a parameter
% (column vector)
% maxlag is the cutoff for the empirical autocorrelation function

i = 1 + 2*sum(empirical_acf(mcmc,maxlag));

end