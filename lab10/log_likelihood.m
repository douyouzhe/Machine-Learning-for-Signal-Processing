function result = log_likelihood(X,W,sigma,K)
[D,N] = size(X);
M = W'* W + sigma.*eye(K);
S = 0;
for i = 1:N
   S = S + norm(X(:,i))^2 ;
end
U = chol(M);
T = inv(U')*(W'*X);
sumT=0;
for i = 1:K
    for j = 1:N
        sumT=sumT+abs(T(i,j))^2;
    end
end
tmp1 = (S - sumT)/(sigma*N);
tmp = 2*sum(log(diag(U))) + (D - K)*log(sigma); 
result = (-N/2)*(D*log(2*pi) + tmp + tmp1);
end