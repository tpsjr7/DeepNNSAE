function test2()
close all;clc;clear all;
addpath minFunc\
load('trainFile.mat')
Xtrain=double(TrainBel);
clear trainbeliefs
STR1=load('C:\Users\Teddy\Desktop\DeepSparseNet\cifar-10-batches-mat\data_batch_1.mat');
STR2=load('C:\Users\Teddy\Desktop\DeepSparseNet\cifar-10-batches-mat\data_batch_2.mat');
STR3=load('C:\Users\Teddy\Desktop\DeepSparseNet\cifar-10-batches-mat\data_batch_3.mat');
STR4=load('C:\Users\Teddy\Desktop\DeepSparseNet\cifar-10-batches-mat\data_batch_4.mat');
STR5=load('C:\Users\Teddy\Desktop\DeepSparseNet\cifar-10-batches-mat\data_batch_5.mat');
Ytrain=[STR1.labels;STR2.labels;STR3.labels;STR4.labels;STR5.labels];
Ytrain=Ytrain+1;
clear STR1 STR2 STR3 STR4 STR5
Xtrain_mean = mean(Xtrain);
Xtrain_sd = sqrt(var(Xtrain)+0.01);
Xtrains = bsxfun(@rdivide, bsxfun(@minus, Xtrain, Xtrain_mean), Xtrain_sd);
Xtrains = [Xtrains, ones(size(Xtrains,1),1)];
C=10;
theta = train_svm(Xtrains, Ytrain,1/0.01);
[val,labels] = max(Xtrains*theta, [], 2);
fprintf('Train accuracy %f%%\n', 100 * (1 - sum(labels ~= Ytrain) / length(Ytrain)));
%% ___________________________________________________***************
load('testFile.mat')
Xtest=double(TestBeliefs);
% load('Testbeliefs.mat')
% Xtest=double(testbeliefs);
clear TestBeliefs
% Xtest=double(L);
%% ___________________________________________________***************
STR=load('C:\Users\Teddy\Desktop\DeepSparseNet\cifar-10-batches-mat\test_batch.mat');
Ytest=[STR.labels];%
Ytest=Ytest+1;
clear STR
Xtest_mean = mean(Xtest);
Xtest_sd = sqrt(var(Xtest)+0.01);
Xtests = bsxfun(@rdivide, bsxfun(@minus, Xtest, Xtest_mean), Xtest_sd);
Xtests = [Xtests, ones(size(Xtests,1),1)];
C=10;
% theta = train_svm(Xtests, Ytest, C);
[val,labels] = max(Xtests*theta, [], 2);
fprintf('Test accuracy %f%%\n', 100 * (1 - sum(labels ~= Ytest) / length(Ytest)));
%%
% disp(size(L))
end