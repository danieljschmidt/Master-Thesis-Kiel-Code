function timetable = transform_table(timetable, transformation)
% returns a transformed table where the type of transformation is given by 
% the input variable transformation

if strcmp(transformation, 'logdiff')
    timetable{2:end,:} = 100*(log(timetable{2:end,:}) - ...
        log(timetable{1:end-1,:}));
    timetable(1,:) = [];
    
elseif strcmp(transformation, 'log')
    timetable{2:end,:} = log(timetable{2:end,:});

elseif strcmp(transformation, 'diff')
    timetable{2:end,:} = timetable{2:end,:} - timetable{1:end-1,:};
    timetable(1,:) = [];

elseif strcmp(transformation, 'none')

else
    error('transformation unknown')

end

