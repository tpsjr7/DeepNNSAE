classdef Network < handle
    properties(SetAccess = public, GetAccess = public)
        numLayers;
        cifarInputObj;% cifarInputObject
        FeatureSizes;
        Layers% array of layer objects
%         centWidth;
% numL,cifTrain,sparseFeatureSizes
    end
    methods
        function Net=Network(numL,CifarInputObj,sparseFeatureSizes)% Constructor
%             ntk=Network(numL,cifTrain,sparseFeatureSizes);
%               sparseFeatureSizes=[48 26;4*26 26;4*26 26;4*26 26];
            if nargin>0
                Net.numLayers=numL;%randn(numCents,width)*0.1;
                Net.FeatureSizes=sparseFeatureSizes;
%                 Net.centPerLayer=centPerL;
                Net.cifarInputObj=CifarInputObj;
                Net.Layers=layer.zeros(numL,1);
                for I=1:numL
                    if(I==1)
                        LSize=[8,8];
                        tempSize=sparseFeatureSizes(I,:);
                        Net.Layers(I)=layer(I,LSize,tempSize(1),tempSize(2));%LNew;
                    else
                        LSize=[(2^(numL-I)),(2^(numL-I))];
                        tempSize=sparseFeatureSizes(I,:);
                        Net.Layers(I)=layer(I,LSize,tempSize(1),tempSize(2));%LNew;
                    end
                end
            end
        end
        function copyCifar(Net,CIFAR)
            Net.cifarInputObj.cifarDim=CIFAR.cifarDim;
            Net.cifarInputObj.images=CIFAR.images;% mx3072
            Net.cifarInputObj.labels=CIFAR.labels;
            Net.cifarInputObj.currentImgIndex=CIFAR.currentImgIndex;% index of the current image
            Net.cifarInputObj.currentImg=CIFAR.currentImg; 
            
            
        end
    end   
end
