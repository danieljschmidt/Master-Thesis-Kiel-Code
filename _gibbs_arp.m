function [mu_all, phi_all, sigma2_all] = gibbs_arp(y, S, p,  ....
    mu_phi_prior, invSigma_phi_prior)

T = size(y,1);

mu_all     = zeros(S,1);
phi_all    = zeros(S,p);
sigma2_all = zeros(S,1);

% initialization with OLS estimates (conditional of first p values)
mu     = mean(y);
u      = y - mu;
phi    = (ylags(u,p)'*ylags(u,p))\(ylags(u,p)'*u(p+1:end));
e      = u(p+1:end) - ylags(u,p)*phi;
sigma2 = e'*e/(T-p);

for s=1:S

    Phi = [phi'; eye(p-1) zeros(p-1,1)];
    Sigma1p = (eye(p^2)-kron(Phi, Phi))\[1; zeros(p^2-1,1)];
    Sigma1p = reshape(Sigma1p,p,p);

    e = zeros(T,1);
    e(1:p) = chol(Sigma1p, 'lower')*u(1:p);
    e(p+1:T) = u(p+1:end) - ylags(u,p)*phi;

    % draw sigma2
    sigma2   = 1/gamrnd(T/2, 2/(e'*e));
    
    % draw mu
    phi1      = 1-ones(p,1)'*phi;
    sigma2_mu = sigma2/(ones(p,1)'*(Sigma1p\ones(p,1)) + (1-ones(p,1)'*phi)^2*(T-p));
    mu_mu     = (phi1*sum(y(p+1:end) - ylags(y,p)*phi)+ones(1,p)*(Sigma1p\y(1:p)))/(ones(p,1)'*(Sigma1p\ones(p,1)) + phi1^2*(T-p));
    mu        = mu_mu + sqrt(sigma2_mu)*randn();
    
    % phi
    u = y - mu;
    invSigma_phi = 1/sigma2*ylags(u,p)'*ylags(u,p) + invSigma_phi_prior;
    mu_phi = invSigma_phi\(1/sigma2*ylags(u,p)'*u(p+1:T) + invSigma_phi_prior\mu_phi_prior);
    phi_prop = mu_phi + chol(invSigma_phi, 'lower')\randn(p,1);
    
    while ~all(abs(roots([1; -phi_prop])) < 1)
        phi_prop        = mu_phi + chol(invSigma_phi, 'lower')\randn(p,1);
    end
    
    Phi = [phi_prop'; eye(p-1) zeros(p-1,1)];
    Sigma1p = (eye(p^2)-kron(Phi, Phi))\[1; zeros(p^2-1,1)];
    Sigma1p = reshape(Sigma1p,p,p);
    
    R = det(Sigma1p)^(-1/2)*exp(-1/(2*sigma2)*u(1:p)'*(Sigma1p\u(1:p)));
    if rand() < R
        phi = phi_prop;
    end
    
%     % phi
%     
%     ybar       = y - mu;
%     mu_phi     = ybar(1:T-1)'*ybar(2:T)/(ybar(1:T-1)'*ybar(1:T-1));
%     sigma2_phi = sigma2/(ybar(1:T-1)'*ybar(1:T-1));
%     phi        = mu_phi + sqrt(sigma2_phi)*randn();
%     
%     while abs(phi) > 1
%         phi        = mu_phi + sqrt(sigma2_phi)*randn();
%     end
    
    % store everything
    
    mu_all(s,:)     = mu;
    phi_all(s,:)    = phi;
    sigma2_all(s,:) = sigma2;
    
end

end

