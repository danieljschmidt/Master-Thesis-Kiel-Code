n_all = [1 5 10 20 50];
k_all = [1 5 10 20 50];

S = 100;

results = zeros(length(n_all), length(k_all));

for i=1:length(n_all)
    for j=1:length(k_all)
        
        n = n_all(i);
        k = k_all(j);
        
        muSSM = zeros(n,1);
        A = eye(k);
        B = randn(n,k);
        Sigma_eps = eye(k);
        Sigma_eta = eye(n);
        R_eps = eye(k);
        R_eta = eye(n);
        mu_xf1 = zeros(k,1);
        Sigma_xf1 = eye(k);
        
        [x_true, y] = simulate_ssm(T, ...
            muSSM, A, B, Sigma_eps, R_eps, Sigma_eta, R_eta, mu_xf1, Sigma_xf1);
        
        t1 = cputime();
        for s=1:S
            xs = fast_state_smoothing(y, muSSM, A, B, Sigma_eps, R_eps, ...
                Sigma_eta, R_eta, mu_xf1, Sigma_xf1);
        end
        t2 = cputime();
        results(i,j) = t2 - t1;
        
    end
end