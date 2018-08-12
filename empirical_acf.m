function acf = empirical_acf(y, maxlag)
% returns the empirical autocorrelation function of the data y (column 
% vector!) for lags 0 to maxlag

T = size(y,1);
acf = zeros(maxlag+1,1);
ybar = mean(y);
var_y = 1/T*(y-ybar)'*(y-ybar);
for lag=0:maxlag
    acf(lag+1) = empirical_autocov(y, lag)/var_y;
end

end

function autocov = empirical_autocov(y, lag)
% returns the empirical autocovariance of the data y for the given lag
% length

T = size(y,1);
ybar = mean(y);
autocov = 1/(T-lag)*(y(lag+1:T)-ybar)'*(y(1:T-lag)-ybar);

end