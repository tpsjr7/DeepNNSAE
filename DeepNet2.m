function DeepNet2()
clear all;close all; clc;
addpath minFunc;
fprintf('SparseAutoEncoder DeSTIN begins here \n');
warning off;
%function Net=Network(numL,centPerL,CifarInputObj)
%% Load Data
fprintf('Loading Data \n');
cifTrain=cifarData(1);% load cifar data_batch_1.mat as training set
cifTest=cifarData(6);% load cifar data_batch_3.mat as test set
%% Set network and Learning Parameters
maxIterTrain=10000;
maxIterTest=10000;
numL=4;
%% Options to save and clear beliefs
% FID=fopen('testBeliefs.txt','w');
% fclose(FID);
% FID=fopen('trainBeliefs.txt','w');
% fclose(FID);
BeliefArray=[];
% TRAIN_BELIEFS=[];
saveOpt=true;
clearOpt=false;
%% Initialize Network Object
fprintf('Initializing Network \n');
sparseFeatureSizes=[48 26;4*26 26;4*26 26;4*26 26];
ntk=Network(numL,cifTrain,sparseFeatureSizes);
TrainBel=[];
for cif=1:5
cifTrain=cifarData(cif);
%% Training
for iter=1:maxIterTrain
    % function doDestin(Net, LayerNum,saveOpt,clearOpt)
%     disp(iter)
%     pause
    if mod(iter,50)==0
        fprintf('Training Image %d of %d %d \n',iter, maxIterTrain, cif);
    end
    for L=1:numL
        [Row,Col]=size(ntk.Layers(L).autoEncoders);
        if L==1
            ntk.Layers(L).loadInput(reshape(cifTrain.getCurrentImg,[32,32,3]));
        else
            ntk.Layers(L).loadInput(ntk.Layers(L-1).autoEncoders);
        end
        
        for R=1:Row
            for C=1:Col
                ntk.Layers(L).autoEncoders(R,C).doAutoEncoderLearning();
                ntk.Layers(L).autoEncoders(R,C).forwardProp();
                ntk.Layers(L).autoEncoders(R,C).doFeatureExtraction();
            end
        end
% %         if saveOpt==true && L>1
% %             FID=fopen('testBeliefs.txt','a+');
%             for II=1:size(ntk.Layers(L).autoEncoders,1)
%                 for JJ=1:size(ntk.Layers(L).autoEncoders,1)
%                     BeliefArray=[BeliefArray ntk.Layers(L).autoEncoders(II,JJ).features];
%                 end
%             
%             end
% %         end
    end

%       TRAIN_BELIEFS(iter,:)=double(BeliefArray(:)');
%       BeliefArray=[];
%      cifTest.findNextImg();    
     cifTrain.findNextImg();
end
% TrainBel=[TrainBel;TRAIN_BELIEFS];
% clear TRAIN_BELIEFS
% TRAIN_BELIEFS=[];
end
% % clear
for cif=1:5
cifTrain=cifarData(cif);
%% Training
for iter=1:maxIterTrain
    % function doDestin(Net, LayerNum,saveOpt,clearOpt)
%     disp(iter)
%     pause
    if mod(iter,50)==0
        fprintf('Training Image %d of %d %d \n',iter, maxIterTrain, cif);
    end
    for L=1:numL
        [Row,Col]=size(ntk.Layers(L).autoEncoders);
        if L==1
            ntk.Layers(L).loadInput(reshape(cifTrain.getCurrentImg,[32,32,3]));
        else
            ntk.Layers(L).loadInput(ntk.Layers(L-1).autoEncoders);
        end
        
        for R=1:Row
            for C=1:Col
%                 ntk.Layers(L).autoEncoders(R,C).doAutoEncoderLearning();
                ntk.Layers(L).autoEncoders(R,C).forwardProp();
                ntk.Layers(L).autoEncoders(R,C).doFeatureExtraction();
            end
        end
%         if saveOpt==true && L>1
%             FID=fopen('testBeliefs.txt','a+');
            for II=1:size(ntk.Layers(L).autoEncoders,1)
                for JJ=1:size(ntk.Layers(L).autoEncoders,1)
                    BeliefArray=[BeliefArray ntk.Layers(L).autoEncoders(II,JJ).features];
                end
            
            end
%         end
    end

      TRAIN_BELIEFS(iter,:)=double(BeliefArray(:)');
      BeliefArray=[];
%      cifTest.findNextImg();    
     cifTrain.findNextImg();
end
TrainBel=[TrainBel;TRAIN_BELIEFS];
% clear TRAIN_BELIEFS
TRAIN_BELIEFS=[];
end
save 'TrainBeliefs50000.mat' TrainBel
ntk.copyCifar(cifTest);
for iter=1:maxIterTest

    if mod(iter,50)==0
        fprintf('Testing Image %d of %d \n',iter, maxIterTest);
    end
    for L=1:numL
        [Row,Col]=size(ntk.Layers(L).autoEncoders);
        if L==1
            ntk.Layers(L).loadInput(reshape(cifTest.getCurrentImg,[32,32,3]));
        else
            ntk.Layers(L).loadInput(ntk.Layers(L-1).autoEncoders);
        end
        
        for R=1:Row
            for C=1:Col
%                 ntk.Layers(L).autoEncoders(R,C).doAutoEncoderLearning();
                ntk.Layers(L).autoEncoders(R,C).forwardProp();
                ntk.Layers(L).autoEncoders(R,C).doFeatureExtraction();
            end
        end
        if saveOpt==true && L>=1
            for II=1:size(ntk.Layers(L).autoEncoders,1)
                for JJ=1:size(ntk.Layers(L).autoEncoders,1)
                    BeliefArray=[BeliefArray ntk.Layers(L).autoEncoders(II,JJ).features];
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
save 'TestBeliefs10000.mat' TestBeliefs
fprintf('Done\n');
end