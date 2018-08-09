T = 500;
mu_true = -10.;
phi_true = 0.95;
sigma2_true = (0.2)^2;

[y, h_true] = simulate_sv(T, mu_true, phi_true, sigma2_true);

yt = log(y.^2);
S = 10000;

[h_all, mu_all, phi_all, sigma2_all] = gibbs_sv_centered(yt, S, ...
    -9, 0.9, 0.1^2);

[h_all2, mu_all2, phi_all2, sigma2_all2] = gibbs_sv_notcentered(yt, S, ...
    mu_true, phi_true, sigma2_true);

cutoff = 1000;

hq  = quantile(h_all( cutoff:end,:), [0.025, 0.975])';
hq2 = quantile(h_all2(cutoff:end,:), [0.025, 0.975])';

figure
plot(1:T, h_true)
hold on
plot(1:T, hq, 'r')
hold on
plot(1:T, hq2, 'g')
hold off

figure
plot(1:T, y)
hold on
plot(1:T, [2 -2].*exp(median(h_all(cutoff:end,:))/2)', 'r')
hold off

% maxlags = 150;
% 
% figure
% plot(0:maxlags, autocorr(mu_all, 0:maxlags))
% title('autocorr of mu')
% 
% figure
% plot(0:maxlags, autocorr(phi_all, 0:maxlags))
% title('autocorr of phi')
% 
% figure
% plot(0:maxlags, autocorr(sigma2_all, 0:maxlags))
% title('autocorr of sigma2')