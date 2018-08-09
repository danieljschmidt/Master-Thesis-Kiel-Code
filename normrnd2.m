function x = normrnd2(mu, invsigma2)
% returns a random sample from a Normal distribution with mean mu and
% variance (invsigma2)^{-1}
% I could not use the function name normrnd since it is already used by the
% Statistics Toolbox

n = length(mu);
x = mu + chol(invsigma2,'lower')\randn(n,1);

end