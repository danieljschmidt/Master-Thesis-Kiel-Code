function phi = normrnd_stationary(mu_phi, invsigma2_phi, maxtimes)
% returns a random sample from a Normal distribution with mean mu_phi and
% variance (invsigma2_phi)^{-1} that has been truncated such that only 
% draws that correspond to a stationary AR process with lag polynomial 
% 1 - phi_1 L - ... phi_p L^p are allowed

phi = normrnd2(mu_phi, invsigma2_phi);

times = 0;
while any(abs(roots([1; -phi])) > 1)
    times = times + 1;
    phi = normrnd2(mu_phi, invsigma2_phi);
    if times > maxtimes
        error('cannot sample from truncated Normal distribution')
    end
end
    
end