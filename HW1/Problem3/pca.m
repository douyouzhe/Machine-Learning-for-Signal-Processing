%%%Function of PCA
function [mu,U,Y,S]=pca(X,d)
[D,N]=size(X);
mu=sum(X,2)/N;
X=X-mu*ones(1,N);
[U,S,V]=svd(X,0);
U=U(:,1:d);
Y=U'*X;
end