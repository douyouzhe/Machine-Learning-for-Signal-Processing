file_name = 'Final_digits/';
matrix_name = dir([file_name '*.mat']);
data_all = {};
for k =1:length(matrix_name)
    load([file_name matrix_name(k).name]);
    tmp = [];
    for i = 1:(size(feat,2))
        for j = 1:(size(feat,3))
            tmp(i,j) = feat(1,i,j);
        end
    end
    data_all{k} = tmp;
end
%%
res = zeros(300,300);
for i = 1:300
    i
    for j = 1:300
        if j ~= i      
            T = data_all{i};
            R = data_all{j};    
            S = zeros(size(T,1),size(R,1));
            for m = 1:size(T,1)
                for n = 1:size(R,1)
                    S(m,n) = sum((T(m,:)-R(n,:)).^2).^0.5;
                end
            end
            [D,index] = dtw(S);
            [M,N] = size(D);
            m = M;
            n = N;
            count = 0;
            while m ~= 1 && n ~= 1
                count = count + 1;
                if index(m,n) == 1
                    m = m-1;
                elseif index(m,n) == 2
                    n = n-1;
                elseif index(m,n) == 3
                    m = m-1;
                    n = n-1;
                end
            end
            res(i,j) = D(M,N) / count;
        end
    end
end


%% 
confusion = zeros(300,300);
count = zeros(300,1);
for i = 1 :300
    [B,I] = sort(res(i,:));
    for j = 1:30
        if  (I(j)>=30*(ceil(i/30)-1)+1) &&...
                (I(j) <= 30*ceil(i/30)) &&...
            (I(j) ~=i)
            count(i) = count(i) + 1;
        end
    end
end
total_count = zeros(10,1);
accuracy = zeros(10,1);
for i = 1 : 10
    for j = 30*i-29:30*i
       total_count(i) = total_count(i) + count(j);
    end
    accuracy(i) = total_count(i) / (29*30);
end