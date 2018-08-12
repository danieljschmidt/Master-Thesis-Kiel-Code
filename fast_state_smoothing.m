function xs = fast_state_smoothing(y, A, B, Sigma_eps, R_eps, ...
    Sigma_eta, R_eta, mu_xf1, Sigma_xf1)
%
% returns the smoothed states calculated based on the data y for a known
% states space model
%
% x_t = A x_{t-1} + R_eps eps_t
% y_t = B x_t + R_eta eta_t
% x_1   ~ Normal(mu_xf1, Sigma_xf1)
% eps_t ~ Normal(0, Sigma_eps)
% eta_t ~ Normal(0, Sigma_eta)
%
% the implemented algorithm is the "fast state smoothing algorithm" from
% the Durbin and Koopman book
%
% function input:
%
% y:           T x   n (is allowed to contain NaN values!)
% A:          10 x  10
% B:           n x  10
% Sigma_eps:   2 x   2
% R_eps:      10 x   2
% Sigma_eta: n-1 x n-1
% R_eta:     n   x n-1
% mu_xf1:     10 x   1
% Sigma_xf1:  10 x  10
%
% function output:
% 
% xs           T x  10
%

V = R_eps*Sigma_eps*R_eps';
W = R_eta*Sigma_eta*R_eta';

[T, ~] = size(y);
[~, k] = size(B);

% recursion 1 (Kalman filter)

mu_xf    = mu_xf1;
Sigma_xf = Sigma_xf1;

q = zeros(T,k);
L = zeros(T,k,k);

for t=1:T

    yt = y(t,:)';
    st = ~isnan(yt);
    yt = yt(st);

    Bt  = B(st,:);
    Wt  = W(st, st);

    dyt = yt-Bt*mu_xf;
    Ft  = Bt*Sigma_xf*Bt' + Wt;
    Mt  = Bt'/Ft;             % <- computationally most expensive step
    Kt  = A*Sigma_xf*Mt;

    Lt  = A-Kt*Bt;
    qt = Mt*dyt;

    L(t,:,:) = Lt;
    q(t,:) = qt';

    mu_xf    = A*mu_xf + Kt*dyt;
    Sigma_xf = A*Sigma_xf*Lt'+ V;

end

% recursion 2 (backwards!)

r = zeros(T+1, k);
% index shifted by +1 !
% i.e. r(T+1,:) = 0

rt = zeros(k,1);

for t=T:-1:1
    Lt = squeeze(L(t,:,:));
    rt = q(t,:)' + Lt'*rt;
    r(t,:) = rt';
end

% recursion 3

xs = zeros(T,k);
xst = mu_xf1 + Sigma_xf1*r(1,:)';
xs(1,:) = xst';

for t=1:T-1
    xst = A*xst + V*r(t+1,:)';
    xs(t+1,:) = xst';
end
    
end

