%% load image
for i=1:40
    train_all{i}=[];
    for j=1:9
        img=loadimage(i,j);
        train_all{i}(:,j)=img(:);
    end
end
for i=1:40
        img=loadimage(i,10);
        test_all(:,i)=img(:);
end 
[r,c] = size(img);
train_data=zeros(r*c,40*9);
for i=1:40
    for j=1:9
            train_data(:,9*(i-1)+j)=train_all{i}(:,j);
    end
end
%% PCA
dimension = 300;
[D,N]=size(train_data);
ave=sum(train_data,2)/N;
train_data=train_data-ave*ones(1,N);
[U,S,V]=svd(train_data,0);
U=U(:,1:dimension);
trainProj=U'*train_data;
test_all = test_all - sum(test_all,2)/40*ones(1,40);
test_data = U'*test_all;

%% LDA
mat = [];
xn=[];
Sw = zeros(300,300);
Sk = zeros(300,300);
Sb = zeros(300,300);
for i=1:40
    tmp = 9*(i-1);
    mat(:,i)=sum(trainProj(:,tmp+1:tmp+9),2)/9;
    for n=1:9
        xn=trainProj(:,tmp+n);
        Sk=Sk+(xn-mat(:,i))*(xn-mat(:,i))';
    end
    Sw=Sw+Sk;
end
mat_1=sum(mat,2)/40;
for i=1:40
    incre=9.*(mat(:,i)-mat_1)*(mat(:,i)-mat_1)';
    Sb=Sb+incre;
end
[V,D]=eigs(Sb, Sw, 40-1);

%% TEST
response=[];
for i=1:40
    tmp = 9*(i-1);
    response(1,tmp+1:tmp+9)=i;
end
NumNeighbors = 9;
acc = zeros(1,4);
for i = 10:10:40
    if(i==40) 
        i=39;
    end
    projection10=V(:,1:i)'*trainProj;
    test10=V(:,1:i)'*test_data;
    model = fitcknn(projection10',response','NumNeighbors',NumNeighbors,'Standardize',1,'Distance','cosine');
    result= predict(model,test10');
    count = 0;
    for j=1:40
        if result(j)==j
            count=count+1;
        end
    end
    acc(1,round((i+1)/10)) = count/40;
end