%% load data
L = 24;
data = cell(24,3);
n = ones(24,1);
for i = 1: 24
    data{i,1} = load(['/Users/youzhedou/Desktop/Code_samples/Machine-Learning-for-Signal-Processing/HW2/Data/Train/Class_' num2str(i) '.txt']);
    data{i,2} = load(['/Users/youzhedou/Desktop/Code_samples/Machine-Learning-for-Signal-Processing/HW2/Data/DEV/Class_' num2str(i) '.txt']);
    data{i,3} = load(['/Users/youzhedou/Desktop/Code_samples/Machine-Learning-for-Signal-Processing/HW2/Data/Eval/Class_' num2str(i) '.txt']);
    n(i) = size(data{i,1},1);
end
%% Question 1
%% Normalizing
data_norm = data;
for i = 1:24
    for j = 1:3
        [l,m] =size(data{i,j});
        for k = 1 : l
            data_norm{i,j}(k,:) = data_norm{i,j}(k,:) ./ norm(data_norm{i,j}(k,:));
        end
    end
end
%% LDA Training
tmp = data_norm{1,1};
for i = 2 :24
    tmp = [tmp;data_norm{i,1}];
end
w_mean = mean(tmp,1);
w_cl = zeros(24,600);

Sb = zeros(600,600);
for i = 1:24
    w_cl(i,:) = mean(data_norm{i,1},1);
    Sb = Sb + n(i)* (w_cl(i,:)-w_mean)'*(w_cl(i,:)-w_mean);
end

Sw = zeros(600,600);
for i = 1 :24
    for j = 1 : n(i)
        Sw = Sw + (data_norm{i,1}(j,:)-w_cl(i,:))'* (data_norm{i,1}(j,:)-w_cl(i,:));
    end
end
[V,D] = eigs(Sb,Sw,L-1);
%% Classifier Training
for j = 1 : 24 
    [h,g] = size(data_norm{j,1});
    for k = 1 :h
        data_proj{j,1}(k,:) = V'* (data_norm{j,1}(k,:))'/norm(V'* (data_norm{j,1}(k,:))');
    end
    M{j,1} = 1/h * sum(data_proj{j,1},1) / norm(1/h * sum(data_proj{j,1},1));
end
%% Classifier Testing
w_test = cell(24,3);
for i = 2:3
    for j = 1:24
        [h,g] = size(data_norm{j,i});
        for k = 1:h
            w_test{j,i}(k,:) = V'* (data_norm{j,i}(k,:))'/ norm(V' * (data_norm{j,i}(k,:))');
        end
    end
end 
score = cell(24,3);
for i = 2:3
    for j = 1:24
        [h,g] = size(data_norm{j,i});
        for k = 1 :h
            for l = 1:24
                score{j,i}(k,l) = dot(w_test{j,i}(k,:),M{l,1});
            end
            [~,b] = max(score{j,i}(k,:));
            class{i,j}(k,1) = b;
        end
    end
end
%% acc
for i = 2:3
    num_a = 0;
    num_b = 0;
    for j = 1 : 24
        [h,g] = size(class{i,j});
        for k = 1: h
            if class{i,j}(h,1) == j
                num_a = num_a +1;
            end
            num_b = num_b+1;
        end
    end
    acc(i-1,1) = num_a/num_b;
end
% Question II
%%
A=(data{1,1})';
for i = 2 :24 
    A = [A,(data{i,1})'];
end
[D,N] = size(A);

%%
for i = 2:3
    i
    for j = 1:24
        j
        [h,g] = size(data{j,i});
        for k = 1:h
            k
            y = data{j,i}(k,:);
            a{j,i}(k,:) = SolveBP(A, y', N,10,1,1e-3);
            x1 = 1;
            for l = 1 : 24
                at{l} = a{j,i}(k,x1:x1+n(l)-1);
                x1 = x1+n(l);
            end
            for l = 1 :24
                Error_2{j,i}(l,1) = norm(y - at{l}*data{l,1},2);
            end
            [xx,yy] = min(Error_2{j,i});
            class_2{j,i}(k,1) = yy;
        end
    end
end
for i = 2:3
    num_a = 0;
    num_b = 0;
    for j = 1 : 24
        [h,g] = size(class_2{j,i});
        for k = 1: h
            if class_2{j,i}(h,1) == j
                num_a = num_a +1;
            end
            num_b = num_b+1;
        end
    end
    acc_2(i-1,1) = num_a/num_b;
end
%% Question III
for i = 2:3
    i
    for j = 1 : 24 
        j
        [h,g] = size(w_test{j,i});
        for k = 1 :h
            k
            y = w_test{j,i}(k,:);
            %a_test{j,i}(k,:) = SolveBP(V'*A, w_test{j,i}(k,:)', N,10,1,1e-3);
            a_test{j,i}(k,:) = SolveBP(V'*A/norm(V'*A), w_test{j,i}(k,:)', N,10,1,1e-3);
            %a_test{j,i}(k,:) = lasso(V'*A/norm(V'*A), w_test{j,i}(k,:)', 'Lambda',0.002);
            x1 = 1;
            for l = 1 : 24
                at_test{l} = a_test{j,i}(k,x1:x1+n(l)-1);
                x1 = x1+n(l);
            end
            for l = 1 :24
                Error_3{j,i}(l,1) = norm(y - at_test{l}*(V'*data{l,1}'/norm(V'*data{l,1}'))',2);
            end
            [xx,yy] = min(Error_3{j,i});
            class_3{j,i}(k,1) = yy;
        end
    end
end

for i = 2:3
    num_a = 0;
    num_b = 0;
    for j = 1 : 24
        [h,g] = size(class_3{j,i});
        for k = 1: h
            if class_3{j,i}(h,1) == j
                num_a = num_a +1;
            end
            num_b = num_b+1;
        end
    end
    acc_3(i-1,1) = num_a/num_b;
end
%% Question IV
for i = 1: 24
    if size(data{i,1},1) ~= 55
      [~,data_kmean{i,1}] = kmeans(data{i,1},55);
    else
        data_kmean{i,1} = data{i,1};
    end
end
AA=(data_kmean{1,1})';
for i = 2 :24 
    AA = [AA,(data_kmean{i,1})'];
end
for i = 2:3
    i
    for j = 1 : 24 
        j
        [h,g] = size(w_test{j,i});
        for k = 1 :h
            k
            y = w_test{j,i}(k,:);
            aa_test{j,i}(k,:) = SolveBP(V'*AA/norm(V'*AA), w_test{j,i}(k,:)', 1320,10,1,1e-3);
            %aa_test{j,i}(k,:) = lasso(V'*AA/norm(V'*AA), w_test{j,i}(k,:)', 'Lambda',0.002);
            x1 = 1;
            for l = 1 : 24
                ata_test{l} = aa_test{j,i}(k,x1:x1+55-1);
                x1 = x1+55;
            end
            for l = 1 :24
                Error_4{j,i}(l,1) = norm(y - ata_test{l}*(V'*data_kmean{l,1}'/norm(V'*data_kmean{l,1}'))',2);
            end
            [xx,yy] = min(Error_4{j,i});
            class_4{j,i}(k,1) = yy;
        end
    end
end

for i = 2:3
    num_a = 0;
    num_b = 0;
    for j = 1 : 24
        [h,g] = size(class_4{j,i});
        for k = 1: h
            if class_4{j,i}(h,1) == j
                num_a = num_a +1;
            end
            num_b = num_b+1;
        end
    end
    acc_4(i-1,1) = num_a/num_b;
end
%% Question V
meann=cell(1,24);
covv=cell(1,24);
share_cov=zeros(600,600);
for i =1:24
    meann{i}=mean(data{i,1},1);
    covv{i}=cov(data{i,1});
    share_cov=share_cov + (size(data{i,1},1)/22491)*covv{i};
end
score_d=zeros(2400,24);
data_dev=[];
for i =1:24
    data_dev = [data_dev;data{i,2}];
end
for i =1:24
    score_d(:,i)= mvnpdf(data_dev,meann{i},share_cov);
end
[~,result_d] = max(score_d,[],2);
acc_d=0;
for i =1:2400
    if result_d(i)==ceil(i/100)
        acc_d=acc_d+1;
    end
end
acc_d=acc_d/2400;

score_e=zeros(2400,24);
data_evl=[];
for i =1:24
    data_evl = [data_evl;data{i,3}];
end
for i =1:24
    score_e(:,i)= mvnpdf(data_evl,meann{i},share_cov);
end
[~,result_e] = max(score_e,[],2);
acc_e=0;
for i =1:2400
    if result_e(i)==ceil(i/100)
        acc_e=acc_e+1;
    end
end
acc_e=acc_e/2400;
acc_5=[acc_d,acc_e];