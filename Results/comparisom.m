clear
clc
load('T and V.mat');
load('faces32x400.mat');
result = T42*V42;
results = [];
k=1;
for i=1:5
    A=[];
    for j=1:5
        T = imresize(reshape(result(:,k),[32,32]),2);
        A=[A,T];
        k = k + 4;
        
    end
    results = [results;A];
end
faces = [];
k=1;
for i=1:5
    A=[];
    for j=1:5
        T = imresize(reshape(faces_new(:,k),[32,32]),2);
        A=[A,T];
        k = k + 4;
        
    end
    faces = [faces;A];
end

subplot(1,2,1);
imshow(uint8(faces));
title('Original Faces')
subplot(1,2,2);
imshow(uint8(results));
title('Recomposed Faces after NMF')

