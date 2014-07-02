function DeepNet()
clear all;close all; clc;
addpath minFunc;
fprintf('STARTING: SparseAutoEncoder DeSTIN\n');
warning off;
%function Net=Network(numL,centPerL,CifarInputObj)
%% Load Data
fprintf('Loading Data \n');
cifTrain=cifarData(10);% load cifar data_batch_1.mat as training set
cifTest=cifarData(6);% load cifar data_batch_3.mat as test set
%% Set network and Learning Parameters
maxIterTrain=20000;
maxIterTest=10000;
numL=4;
BeliefArray=[];
% TRAIN_BELIEFS=[];
saveOpt=true;
clearOpt=false;
%% Initialize Network Object
fprintf('Initializing Network \n');
sparseFeatureSizes=[48 300;4*300 200;4*200 100;4*100 100];
% sparseFeatureSizes=[48 64;4*64 64;4*64 64;4*64 64];
ntk=Network(numL,cifTrain,sparseFeatureSizes);
TrainBel=[];
% for cif=1:4
% cifTrain=cifarData(10);
%% Training
for iter=1:maxIterTrain
    % function doDestin(Net, LayerNum,saveOpt,clearOpt)
%     disp(iter)
%     pause
    if mod(iter,200)==0
        fprintf('Training Image %d of %d\n',iter, maxIterTrain);
    end
    for L=1:1
        [Row,Col]=size(ntk.Layers(L).autoEncoders);
        if L==1
            ntk.Layers(L).loadInput(reshape(cifTrain.getCurrentImg,[32,32,3]));
        else
            ntk.Layers(L).loadInput(ntk.Layers(L-1).autoEncoders);
        end
        
        for R=1:Row
            for C=1:Col
                if L==1
                    ntk.Layers(L).autoEncoders(R,C).doAutoEncoderLearning('training');
%                     ntk.Layers(L).autoEncoders(R,C).forwardProp();
%                     ntk.Layers(L).autoEncoders(R,C).doFeatureExtraction();
                else
                    ntk.Layers(L).autoEncoders(R,C).doAutoEncoderLearning('training');
%                     ntk.Layers(L).autoEncoders(R,C).forwardProp();
%                     ntk.Layers(L).autoEncoders(R,C).doFeatureExtraction();
                end
%                 ntk.Layers(L).autoEncoders(R,C).forwardProp();
%                 ntk.Layers(L).autoEncoders(R,C).doFeatureExtraction();
            end
        end
        if saveOpt==true
%             FID=fopen('testBeliefs.txt','a+');
            for II=1:size(ntk.Layers(L).autoEncoders,1)
                for JJ=1:size(ntk.Layers(L).autoEncoders,1)
                    BeliefArray=[BeliefArray(:);ntk.Layers(L).autoEncoders(II,JJ).features(:)];
                end
            
            end
        end
    end

      TRAIN_BELIEFS(iter,:)=double(BeliefArray(:)');
%       disp(size((BeliefArray(:)')))
%       pause
      BeliefArray=[];
%      cifTest.findNextImg();    
     cifTrain.findNextImg();
     disp(size(TRAIN_BELIEFS))
pause
end
TrainBel=[TrainBel;TRAIN_BELIEFS];
% clear TRAIN_BELIEFS
TRAIN_BELIEFS=[];
% end
% clear

save 'trainFile.mat' TrainBel
ntk.copyCifar(cifTest);
for iter=1:maxIterTest

    if mod(iter,1000)==0
        fprintf('Testing Image %d of %d \n',iter, maxIterTest);
    end
    for L=1:1
        [Row,Col]=size(ntk.Layers(L).autoEncoders);
        if L==1
            ntk.Layers(L).loadInput(reshape(cifTest.getCurrentImg,[32,32,3]));
        else
            ntk.Layers(L).loadInput(ntk.Layers(L-1).autoEncoders);
        end
        
        for R=1:Row
            for C=1:Col
                if L==1
                ntk.Layers(L).autoEncoders(R,C).doAutoEncoderLearning('testing');
%                     ntk.Layers(L).autoEncoders(R,C).doFeatureExtraction();
                else                
                ntk.Layers(L).autoEncoders(R,C).doAutoEncoderLearning('testing');
%                     ntk.Layers(L).autoEncoders(R,C).forwardProp();
%                     ntk.Layers(L).autoEncoders(R,C).doFeatureExtraction();
                end
            end
        end
        if saveOpt==true
            for II=1:size(ntk.Layers(L).autoEncoders,1)
                for JJ=1:size(ntk.Layers(L).autoEncoders,1)
                    BeliefArray=[BeliefArray(:);ntk.Layers(L).autoEncoders(II,JJ).features(:)];
                end
            
            end
        end
    end
%     disp(size(TRAIN_BELIEFS(iter,:)));
%     disp(size(BeliefArray(:)'));
     TestBeliefs(iter,:)=double(BeliefArray(:)');
     BeliefArray=[];
     cifTest.findNextImg();
end
save 'testFile.mat' TestBeliefs
% %%
% clear all;
% ft=load('C:\Users\Teddy\Desktop\DeepSparseNet\cifar-10-batches-mat\data_batch_1.mat');
% trainY=double(ft.labels+1);
% clear ft
% load('TestBeliefs10000Wide.mat');
% trainXC=TestBeliefs;
% trainXC_mean = mean(trainXC);
% trainXC_sd = sqrt(var(trainXC)+0.01);
% trainXCs = bsxfun(@rdivide, bsxfun(@minus, trainXC, trainXC_mean), trainXC_sd);
% trainXCs = [trainXCs, ones(size(trainXCs,1),1)];
% C = 100;
% theta = train_svm(trainXCs, trainY, C);
% [val,labels] = max(trainXCs*theta, [], 2);
% fprintf('Train accuracy %f%%\n', 100 * (1 - sum(labels ~= trainY) / length(trainY)));
% ft=load()
Fin='finished';
save 'Finished.mat' Fin
fprintf('Done\n');
end