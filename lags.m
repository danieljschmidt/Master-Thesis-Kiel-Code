function X = lags(Y,p)
% returns a matrix X = [Y[p:T-1,:] Y[p-1:T-2,:] ... Y[1:T-p]]
% where Y is a T x n matrix (n=1 is possible)

[T, n] = size(Y);
X = zeros(T-p, p*n);
for i=1:p
   X(:,(i-1)*n+1:i*n) = Y(p+1-i:end-i,:); 
end

end