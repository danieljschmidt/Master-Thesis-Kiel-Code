function r = catrnd(W)
% returns a T x 1 random vector of categorical variables r with weights 
% specified in the (unnormalized) n x T weights matrix W

T = size(W,2);
W_norm = W./sum(W,1);
W_cdf = cumsum(W_norm,1);
x = rand(1,T);
C = x > W_cdf;
r = sum(C,1)' + 1;

end

