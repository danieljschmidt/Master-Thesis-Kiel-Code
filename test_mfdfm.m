
n = 20;
p = 4;
q = 1;
T = 300;
T1 = T + q;

cutoff  = 100;
S       = 1000 + cutoff;
rho_max = 50;

mu_true      = zeros(n,1); %randn(n,1);
lambda_true  = randn(n,1);
phi_true     = [0.4 0.2 0.1 0.];    % no phi close to unit root
sigma2v_true = 1.;
psi_true     = [0.85*rand(n,1)]; % no psi close to unit root
sigma2w_true = 1./gamrnd(ones(n,1), ones(n,1));

% simulate some data

[y_unobs, f_true, u_true] = simulate_mfdfm(T1, ...
    phi_true, sigma2v_true, lambda_true, psi_true, sigma2w_true);

x_true = [
    f_true(q+5:T+q+4)   f_true(q+4:T+q+3)   f_true(q+3:T+q+2) ...
    f_true(q+2:T+q+1)   f_true(q+1:T+q) ...
    u_true(q+5:T+q+4,1) u_true(q+4:T+q+3,1) u_true(q+3:T+q+2,1) ...
    u_true(q+2:T+q+1,1) u_true(q+1:T+q,1)
    ];
% TODO: more elegant

missing_m      = catrnd(ones(3,n)) - 1; % first element is meaningless
y              = nan(T1,n);
y(q+3:3:end,1) = y_unobs(q+3:3:end,1);  % after transformation 3:3:T
for i=2:n
    y(1:end-missing_m(i),i) = y_unobs(1:end-missing_m(i),i);
end

% Gibbs sampler

% TODO initialization

phi0     = zeros(1,p);
lambda0  = randn(n,1);
psi0     = zeros(n,q);
sigma2w0 = ones(n,1);

[x_all, phi_all, lambda_all, psi_all, sigma2w_all] = ...
    gibbs_mfdfm(y, S, p, q, ...
    phi0, lambda0, psi0, sigma2w0);

% old Gibbs sampler
% [x_all, lambda_all, phi_all, psi_all, sigma2w_all] = ...
%     gibbs_mfdfm2(y-mu_true', S);

% plots

% determine whether the sampled lambdas and factors have the opposite sign
% compared to the true ones
sgn = sign(median(lambda_all(cutoff+1:end,:))*lambda_true);

fq = quantile(x_all(cutoff+1:end,:,1), [0.025, 0.975])';
figure
plot(1:T, x_true(:,1))
hold on
plot(1:T, sgn*fq, '--r')
hold off

uq = quantile(x_all(cutoff+1:end,:,6), [0.025, 0.975])';
figure
plot(1:T, x_true(:,6))
hold on
plot(1:T, uq, '--r')
hold off

lambda_q = quantile(lambda_all(cutoff+1:end,:), [0.025, 0.975])';
figure
plot(0.5:n-0.5, lambda_true, 'x')
hold on
plot(0.5:n-0.5, sgn*lambda_q, '+r')
hold off

psi_q = quantile(psi_all(cutoff+1:end,:,1), [0.025, 0.975])';
figure
plot(0.5:n-0.5, psi_true(:,1), 'x')
hold on
plot(0.5:n-0.5, psi_q, '+r')
hold off

sigma2w_q = quantile(sigma2w_all(cutoff+1:end,:), [0.025, 0.975])';
figure
plot(0.5:n-0.5, sigma2w_true, 'x')
hold on
plot(0.5:n-0.5, sigma2w_q, '+r')
hold off

% plot inefficiency factors

if_lambda = zeros(n,1);
for i=1:n
    if_lambda(i) = inefficiency_factor(lambda_all(cutoff+1:end,i),rho_max);
end
figure
scatter(0.5:n-0.5, if_lambda)
hold off

% positive relationship between inefficiency factors and absolute value of
% lambda

% figure
% scatter(abs(lambda_true), if_lambda)
% hold off

if_psi = zeros(n,1);
for i=1:n
    if_psi(i) = inefficiency_factor(psi_all(cutoff+1:end,i,1),rho_max);
end
figure
scatter(0.5:n-0.5, if_psi)
hold off

if_sigma2w = zeros(n,1);
for i=1:n
    if_sigma2w(i) = inefficiency_factor(sigma2w_all(cutoff+1:end,i),rho_max);
end
figure
scatter(0.5:n-0.5, if_sigma2w)
hold off