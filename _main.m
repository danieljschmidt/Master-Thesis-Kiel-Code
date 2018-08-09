function results = main(par) % change name

if (isdeployed)
    maxNumCompThreads(1);
    if nargin==0
        par = input('par = ?');
    else
        par = str2double(par);
    end
end

% main part

save(['main_mfdfm' '_' int2str(par)], 'results') % ????

if isdeployed && nargin==0
    pause
end
end