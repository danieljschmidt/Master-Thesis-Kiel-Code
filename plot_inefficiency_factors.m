function plot_inefficiency_factors(mcmc, rho_max)
% TODO documentation

n = size(mcmc,2);

inefficiency_factors = zeros(n,1);
for i=1:n
    inefficiency_factors(i) = inefficiency_factor(mcmc(:,i),rho_max);
end

figure
scatter(1:n, inefficiency_factors)
xticks(1:n)
xlim([0.5 n+0.5])
hold off

end