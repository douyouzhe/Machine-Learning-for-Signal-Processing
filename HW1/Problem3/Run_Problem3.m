%% 
%Load all data
f = 'face/male/train/';
ls = dir([f '*.jpg']);
maleTrainData = [];
for i =1:length(ls)
    img=imread([f ls(i).name]);
    maleTrainData(:,i)=img(:);
end

f = 'face/female/train/';
ls = dir([f '*.jpg']);
femaleTrainData = [];
for i =1:length(ls)
    img=imread([f ls(i).name]);
    femaleTrainData(:,i)=img(:);
end
allTrainData=[maleTrainData,femaleTrainData];

f = 'face/male/test/';
ls = dir([f '*.jpg']);
maleTestData = [];
for i =1:length(ls)
    img=imread([f ls(i).name]);
    maleTestData(:,i)=img(:);
end
f = 'face/female/test/';
ls = dir([f '*.jpg']);
femaleTestData = [];
for i =1:length(ls)
    img=imread([f ls(i).name]);
    femaleTestData(:,i)=img(:);
end
allTestData=[maleTestData,femaleTestData];
%%
%pca
n = 500;
[D,N]=size(allTrainData);
meanCol=sum(allTrainData,2)/N;
allTrainDataCentered=allTrainData-meanCol*ones(1,N);
[U,S,V]=svd(allTrainDataCentered,0);
U=U(:,1:n);

%% Plot eigenvalue 
x=1:500;
figure(1)
plot(x,diag(S(1:500,1:500)).^2);

%% Plot average face
[row,col] = size(img);
[D1,N1]=size(maleTrainData);
mu1=sum(maleTrainData,2)/N1;
[D2,N2]=size(femaleTrainData);
mu2=sum(femaleTrainData,2)/N2;
iu1=reshape(mu1,[row,col]);
iu2=reshape(mu2,[row,col]);
figure(2)
subplot(2,2,1); imagesc(iu1); title('male avg face');
subplot(2,2,2); imshow(iu1,[]); title('male avg face');
subplot(2,2,3); imagesc(iu2); title('female avg face');
subplot(2,2,4); imshow(iu2,[]); title('female avg face');

%% reconstrcut test face
[D,N]=size(allTestData);
mu3=sum(allTestData,2)/N;

noOfEigenFaces = 100;
eigenVector=U(:,1:noOfEigenFaces);
allTestDataCentered=allTestData-mu3*ones(1,N);
recon1 = eigenVector'*allTestDataCentered;
reconPic1 = eigenVector*recon1(:,1) + mu3;

noOfEigenFaces = 300;
eigenVector=U(:,1:noOfEigenFaces);
allTestDataCentered=allTestData-mu3*ones(1,N);
recon2 = eigenVector'*allTestDataCentered;
reconPic2 = eigenVector*recon2(:,1) + mu3;

noOfEigenFaces = 500;
eigenVector=U(:,1:noOfEigenFaces);
allTestDataCentered=allTestData-mu3*ones(1,N);
recon3 = eigenVector'*allTestDataCentered;
reconPic3 = eigenVector*recon3(:,1) + mu3;

figure(3)
subplot(2,2,1);imshow(reshape(reconPic1,size(img)),[]);
title('reconstructed face using 100 eigen faces')
subplot(2,2,2);imshow(reshape(reconPic2,size(img)),[]);
title('reconstructed face using 300 eigen faces')
subplot(2,2,3);imshow(reshape(reconPic3,size(img)),[]);
title('reconstructed face using 500 eigen faces')
subplot(2,2,4);imshow(reshape(allTestData(:,1),size(img)),[]);
title('original face')


%% simple gender detection system
ED = zeros(2,2000);
res = zeros(2000,1);
for i = 1:2000
    ED(1,i) = norm(allTestDataCentered(:,i)-mu1);
    ED(2,i) = norm(allTestDataCentered(:,i)-mu2);
    if(ED(1,i)>ED(2,i))
        res(i,1)=2;
    else
        res(i,1)=1;
    end
end

%%
acc=0;
for i=1:2000
    if i<=1000 && res(i)==1
        acc=acc+1;
    elseif i>1000 && res(i)==2
        acc=acc+1;
    end
end
acc=acc/2000;
%% Another gender detection system
%IDX=knnsearch(Y',y_test');
%IDX=knnsearch(Y',Project1');
acc1=0;
for i=1:2000
   if i<=1000 && IDX(i)<=1934
       acc1=acc1+1;
   elseif i>1000 && IDX(i)>=1934
       acc1=acc1+1;
   end
end
acc1=acc1/2000;