% TODO update


gdp_realtime = readtable("data//realtime//GDPC1.csv");

rt_data = gdp_realtime;

origins = datetime(2000, 1, 1) + calmonths(0:3:18*12-1) + calmonths(2); 
targets = origins ;
% 2000Q1 - 2017Q4 when GDP of the last quarter was already published

n_forecasts = length(origins);

est_start = datetime(1980, 1, 1);

S = 1000;
pred_all = zeros(n_forecasts, S);

p = 4;

i = n_forecasts;

invSigma_phi_prior = diag([1; 100*ones(p-1,1)]);
mu_phi_prior = [1; zeros(p-1,1)];

for i=1:n_forecasts

    origin = origins(i);
    target = targets(i);

    cond1 = rt_data.realtime_start <= origin;
    cond2 = rt_data.realtime_end > origin;
    
    data = rt_data(cond1 & cond2, :);
    data.realtime_start = [];
    data.realtime_end   = [];
    
    data.date = data.date + calmonths(2); 
    % move data from the 1st month of a quarter to the 3rd month of a quarter
    
    data.GDPC1(2:end) = log(data.GDPC1(2:end)) - log(data.GDPC1(1:end-1));
    data(1,:) = [];
    
    data = data(data.date >= est_start, :);
    
    % check that there are missing GDPC1s in the real-time data?
    
    [mu_all, phi_all, sigma2_all] = gibbs_arp(data.GDPC1, S, p, ...
        mu_phi_prior, invSigma_phi_prior);
    
    pred_all(i,:) = mu_all + sum((data.GDPC1(end:-1:end-p+1)'-mu_all).*phi_all,2) + ...
        sqrt(sigma2_all).*randn(S,1);
    
end

pred_mean = mean(pred_all,2);

evaluation_date = datetime(2018, 3, 31);

cond1 = rt_data.realtime_start <= evaluation_date;
cond2 = rt_data.realtime_end > evaluation_date;

eval_data = rt_data(cond1 & cond2, :);
eval_data.realtime_start = [];
eval_data.realtime_end   = [];

eval_data.date = eval_data.date + calmonths(2); 

eval_data = table2timetable(eval_data, 'RowTimes', 'date');

eval_data = transform(eval_data, 'logdiff');

cond1 = eval_data.date >= targets(1);
cond2 = eval_data.date <= targets(end);

eval_data = eval_data(cond1 & cond2,:);

errors = pred_mean - eval_data{:,:};

rmse = sqrt(mean(errors.^2));

%observations = 

%save("test.mat", "pred_all")