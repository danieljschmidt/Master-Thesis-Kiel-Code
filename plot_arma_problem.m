n = 20;
p = 1;
q = 1;
T = 300;
T1 = T + q;

rng(100);

cutoff  = 1000;
S       = 20000 + cutoff;
rho_max = 1000;

mu_true      = randn(n,1);
lambda_true  = randn(n,1);
phi_true     = 0.8*randn(1);           % no phi close to unit root
sigma2v_true = 1.;
psi_true     = [0.2; 0.8*rand(n-1,1)]; % no psi close to unit root
sigma2w_true = ones(n,1);

% - - - simulate some data

[y_unobs, f_true, u_true] = simulate_mfdfm(T1, ...
    mu_true, lambda_true, phi_true, sigma2v_true, psi_true, sigma2w_true);

y              = nan(T1,n);
y(q+3:3:end,1) = y_unobs(q+3:3:end,1);  % after transformation 3:3:T
for i=2:n
    y(1:end,i) = y_unobs(1:end,i);
end

% - - - Gibbs sampler 

[x_all, phi_all, mu_all, lambda_all, psi_all, sigma2w_all] = ...
    gibbs_mfdfm(y, S, p, q, ...
    phi_true, mu_true, lambda_true, psi_true, sigma2w_true);
% initialization with true values to speed up convergence

% - - - inefficiency factors

if_psi_1     = inefficiency_factor(psi_all(    cutoff+1:end,1),rho_max);
if_sigma2w_1 = inefficiency_factor(sigma2w_all(cutoff+1:end,1),rho_max);

disp(['inefficiency factor for psi_1:     ', num2str(if_psi_1)])
disp(['inefficiency factor for sigma2w_1: ', num2str(if_sigma2w_1)])

% - - - save data for proper plot in julia

data_mcmc = [psi_all(cutoff+1:end,1), sigma2w_all(cutoff+1:end,1)];
data_true = [psi_true(1) sigma2w_true(1)];

csvwrite("data_plots\\data_arma_problem_mcmc.csv", data_mcmc);
csvwrite("data_plots\\data_arma_problem_true.csv", data_true);

% scatter plot of psi(1) vs sigma2w
% nicer version in julia (hexbinplot)

% - - - calculate line in plot

% the line describes all possible combinations of psi_1 and sigma2w_1 that
% lead to the same variance of 1/3 u_t + 2/3 u_{t-1} + ... + 1/3 u_{t-4}

var_true = sigma2w_true(1)*covmatrix(psi_true(1), 0);

psi_rng = -0.99:0.01:0.99;
n_points = length(psi_rng);
var_rng = zeros(n_points);
for i=1:n_points
    var_rng(i) = covmatrix(psi_rng(i), 0);
end
% var_psi is the variance apart from the factor sigma2w
 
sigma2w_rng = var_true./var_rng;

% - - - scatter plot with line

figure
scatter(data_mcmc(:,1), data_mcmc(:,2), '.')
hold on
plot(psi_true(1), sigma2w_true(1), 'rx')
plot(psi_rng, sigma2w_rng, 'r')
xlabel('\psi_1')
ylabel('\sigma^2_{w,1}')
hold off
 