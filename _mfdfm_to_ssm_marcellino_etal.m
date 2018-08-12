function [mu, A, B, Sigma_eps, R_eps, Sigma_eta, R_eta, mu_xf1, Sigma_xf1] ...
    = mfdfm_to_ssm_marcellino_etal(mu, lambda, phi, sigma2v, psi, sigma2w)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

    n  = length(lambda);
    mu = [3*mu(1); mu(2:end)];

    A1   = [phi zeros(1,4); eye(4,4) zeros(4,1)];
    A2q  = [psi(1) zeros(1,4); eye(4,4) zeros(4,1)];
    A2   = blkdiag(A2q, diag(psi(2:end)));
    A    = blkdiag(A1, A2);
    
    b1q  = lambda(1)/3*[1 2 3 2 1];
    b1m  = reshape(lambda(2:end),n-1,1);
    B1   = [b1q; b1m zeros(n-1,4)];
    b2q  = 1/3*[1 2 3 2 1];
    B2   = blkdiag(b2q, eye(n-1));
    B    = [B1 B2];
    
    R_eps = [1 zeros(1,n); zeros(4,n+1); 0 1 zeros(1,n-1); zeros(4,n+1);
        zeros(n-1,2) eye(n-1,n-1)];

    Sigma_eps = diag([sigma2v; sigma2w]);

    R_eta = zeros(n,1);

    Sigma_eta = 1.; % ugly hack

    % initialization

    mu_xf1 = zeros(10+n-1,1);

    temp = abs(ones(5,1)*[1 2 3 4 5]- [1;2;3;4;5]*ones(1,5));
    part1 = sigma2v/(1-phi^2)*phi.^temp;
    part2 = sigma2w(1)/(1-psi(1)^2)*psi(1).^temp;
    Sigma_xf1 = zeros(10+n-1,10+n-1);
    Sigma_xf1(1:5,1:5) = part1;
    Sigma_xf1(6:10,6:10) = part2;
    for i=1:n-1
        Sigma_xf1(10+i,10+i) = sigma2w(1+i)/(1-psi(1+i)^2);
    end  
    
end

