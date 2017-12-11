function [D,index] = dtw(S)
[M,N] = size(S);
D =zeros(M,N);
index = zeros(M,N); 
for m = 1: M
    for n = 1:N
        if m == 1 && n == 1
            D(m,n) = S(m,n);
            index(m,n) = 0;
        elseif m ==1 && n ~=1
            D(m,n) = S(m,n) + D(m,n-1);
            index(m,n) = 2;
        elseif m ~= 1 && n == 1
            D(m,n) = S(m,n) + D(m-1,n);
            index(m,n) = 1;
        else
            [V, I] = min([D(m-1,n), D(m,n-1), D(m-1,n-1)]);
            D(m,n) = S(m,n) + V;
            index(m,n) = I;
        end
    end
end
end
