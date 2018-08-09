% TODO: app quarterly gdp
% TODO: wrap into function (?)


transformations = containers.Map();
transformations('INDPRO')    = 'logdiff';
transformations('TCU')       = 'diff';
transformations('UNRATE')    = 'diff';
transformations('PAYEMS')    = 'logdiff';
transformations('CPIAUCSL')  = 'logdiff';
transformations('PPIACO')    = 'logdiff';
transformations('HOUST')     = 'logdiff';
transformations('PERMIT')    = 'logdiff';


% 'DFF'         'diff'    % federal funds rate
% 
% 'DTWEXM'      'logdiff' % trade weighted US Dollar index
% 'SP500'       'logdiff' % not from Alfred
% 'VIXCLS'      '???'     % CBOE volatility index
% 'DCOILWTICO'  'logdiff' % crude oil price


predictors = transformations.keys();
n_predictors = length(predictors);

origin = datetime(2017, 12, 31);

est_start = datetime(1990, 1, 1);

for i=1:n_predictors

    predictor = predictors{i};

    tbl = readtable(fullfile('data', 'realtime', strcat(predictor, '.csv')));

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

data = big_tbl{:,:};
data = (data - mean(data))./std(data);
[coeff,score,latent,tsquared,explained] = pca(data);

explained

coeff
big_tbl.Properties.VariableNames

scatter(coeff(:,1), coeff(:,3))


% for predictor=predictors
%     figure
%     plot(big_tbl.date, big_tbl{:,predictor})
%     title(predictor)
%     hold off
% end