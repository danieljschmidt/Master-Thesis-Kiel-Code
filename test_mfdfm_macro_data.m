
transformations = containers.Map();
transformations('RGDP')  = 'logdiff';
transformations('IP')    = 'logdiff';
transformations('RDPI')  = 'logdiff';
transformations('RRS')   = 'logdiff';
transformations('UR')    = 'diff';      % labor
transformations('EMP')   = 'logdiff';
transformations('AWH')   = 'logdiff';
transformations('CPI')   = 'logdiff';   % prices
transformations('PPI')   = 'logdiff';
transformations('HS')    = 'logdiff';   % housing
transformations('BP')    = 'logdiff';
transformations('CS')    = 'none';      % surveys
transformations('PMI')   = 'none';
transformations('PHIL')  = 'none';
transformations('SP500') = 'logdiff';   % financial data
transformations('OIL')   = 'logdiff';
transformations('USD')   = 'logdiff';
transformations('FF')    = 'diff';
transformations('SPREAD')= 'none';
transformations('VXO')   = 'none';

variables = ['RGDP', 'IP', 'RDPI', 'RRS', ...
    'UR', 'EMP', 'AWH', ...
    'CS', 'PHIL', 'PMI', ...
    'HS', 'BP', "CPI", 'PPI', ...
    'SP500', 'OIL', 'USD', 'FF', 'SPREAD', 'VXO'
];

n_predictors = length(variables);

origin = datetime(2017, 12, 31);

est_start = datetime(1986, 12, 1); % TODO q = 1

for i=1:n_predictors

    predictor = variables{i};

    tbl = readtable(fullfile('data', strcat(predictor, '.csv')));

    % construct a vintage at the date given by the origin variable
    cond1 = tbl.realtime_start <= origin;
    cond2 = tbl.realtime_end > origin;
    tbl = tbl(cond1 & cond2, :);
    tbl.realtime_start = [];
    tbl.realtime_end   = [];
    tbl = table2timetable(tbl, 'RowTimes', 'date');
    
    % check whether dates in the vintage are regular and sorted
    if ~isregular(tbl, 'month') || ~issorted(tbl)
        error('series is either not regular or not sorted.')
    end
    tbl = transform_table(tbl, transformations(predictor));

    % TODO no longer necessary
    if strcmp(predictor,"GDPC1")
        tbl.date = tbl.date + calmonths(2);
        % move quarterly obervations from beginning of quarter to end of quarter
    end
    
    % join vintages of different predictors
    if i==1
        big_tbl = tbl;
    else
        big_tbl = outerjoin(big_tbl, tbl);
    end

    % cut off all data before the start of the estimation sample
    cond = big_tbl.date >= est_start;
    big_tbl = big_tbl(cond,:);

end

y = big_tbl{:,:};

[T1, n] = size(y);

p = 4;
q = 1;

T = T1-q;

cutoff  = 100;
S       = 1000 + cutoff;
rho_max = 50;

% de-mean
for i=1:n
    y(:,i) = y(:,i) - mean(y(~isnan(y(:,i)),i));
end

% Gibbs sampler with initialization

[phi0, lambda0, psi0, sigma2w0] = gibbs_mfdfm_init(y, p, q);

[x_all, phi_all, lambda_all, psi_all, sigma2w_all] = ...
    gibbs_mfdfm(y, S, p, q, ...
    phi0, lambda0, psi0, sigma2w0);

x_all       = x_all(cutoff+1:end,:,:);
phi_all     = phi_all(cutoff+1:end,:);
lambda_all  = lambda_all(cutoff+1:end,:);
psi_all     = psi_all(cutoff+1:end,:,:);
sigma2w_all = sigma2w_all(cutoff+1:end,:,:);

% plots

fq = quantile(x_all(cutoff+1:end,:,1), [0.025, 0.975])';
figure
plot(big_tbl.date(2:end), fq, 'b-')
hold on
plot(big_tbl.date(2:length(f0)), f0(2:end), 'r--')
hold off

boxplot(phi_all)
hold on
plot(phi0, 'o', 'Color', 'black')
line([0; 0], [0; p])
hold off

boxplot(lambda_all)
hold on
plot(lambda0, 'o', 'Color', 'black')
line([0; 0], [0; n])
hold off

boxplot(psi_all(:,:,1))
hold on
plot(psi0, 'o', 'Color', 'black')
line([0; 0], [0; n])
hold off

boxplot(sigma2w_all)
hold on
plot(sigma2w0, 'o', 'Color', 'black')
line([0; 0], [0; n])
hold off

% plot inefficiency factors

plot_inefficiency_factors(phi_all,          rho_max)
plot_inefficiency_factors(lambda_all,       rho_max)
plot_inefficiency_factors(psi_all(:,:,1),   rho_max)
plot_inefficiency_factors(sigma2w_all(:,:), rho_max)
