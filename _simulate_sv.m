function [y, h] = simulate_sv(T, mu, phi, sigma2)
sigma = sqrt(sigma2);
h = zeros(T,1);
h(1) = mu + sigma/sqrt(1-phi^2)*randn();
for t=2:T
    h(t) = mu + phi*(h(t-1)-mu) + sigma*randn(); 
end
y = exp(h/2).*randn(T,1);
end

