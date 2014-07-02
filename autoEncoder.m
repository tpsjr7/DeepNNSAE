classdef autoEncoder < handle
    properties(SetAccess = public, GetAccess = public)
        layerNum;
        features;
        featureStore;
        NNSAutoEnc;
        INPUT;
        epoches=16;%belief store length
        encparam=0.25;
    end
    
    methods
        function autoEnc=autoEncoder(layerNumber,visibleSize,HidSize)
            if nargin>0
            autoEnc.layerNum=layerNumber;
            autoEnc.NNSAutoEnc=NNSAE(visibleSize,HidSize,layerNumber);
            autoEnc.NNSAutoEnc.init();
            end
        end
        %Learn optimal Weights using AutoEncoder Neural Nets
        function doAutoEncoderLearning(autoEnc,mode)
            if strcmp(mode,'training')
                for I=1:16
    %                 disp(size(autoEnc.Input))
    %                 pause
    %                 autoEnc.NNSAutoEnc.inp=autoEnc.Input(I,:);
                    autoEnc.NNSAutoEnc.train(autoEnc.INPUT(I,:));
                    autoEnc.NNSAutoEnc.update();%update 
                    autoEnc.doFeatureExtraction();%extract features
                    autoEnc.featureStore(I,:)=autoEnc.features(:)';%loadIth beliefStore
                end
            else
                for I=1:16
                    autoEnc.doFeatureExtraction();%extract features
                    autoEnc.featureStore(I,:)=autoEnc.features(:)';%loadIth beliefStore
                end
            end
            autoEnc.features=mean(autoEnc.featureStore);
%             disp(size(autoEnc.features))
%             pause
        end
        
        function loadInput(autoEnc,Inp)
            autoEnc.INPUT=Inp;
%             autoEnc.NNSAutoEnc.inp=Inp;            
%             if autoEnc.layerNum>1
%                 Inp=Inp(:);
%             end
%             autoEnc.NNSAutoEnc.inp=Inp(:)';
        end
        % forward propagation the input over the auto encoder NNet
        function forwardProp(autoEnc)%% Needs a revision
%             g1=autoEnc.W1*autoEnc.a1'+autoEnc.b1;
%             autoEnc.a2=sigmoid(g1);
%             g2=autoEnc.W2*autoEnc.a2+autoEnc.b2;
%             autoEnc.a3=sigmoid(g2);
%             autoEnc.NNSAutoEnc.inp=autoEnc.Input;
            autoEnc.NNSAutoEnc.update();
%             IMG1=double(SAENet.apply());
        end
        function doFeatureExtraction(autoEnc)%% Needs a revision
            alpha=autoEnc.encparam;
%             disp(size(autoEnc.NNSAutoEnc.W))
%             disp(size(autoEnc.NNSAutoEnc.inp))
%             pause;
            z = autoEnc.NNSAutoEnc.W' * autoEnc.NNSAutoEnc.inp;
%             disp(size(z))
%             pause
            autoEnc.features = max(z - alpha, 0);%; -max(-z - alpha, 0) ];
%             disp(size(autoEnc.NNSAutoEnc.h))
%             pause;
%             autoEnc.features=autoEnc.NNSAutoEnc.h;
%             autoEnc.features=autoEnc.a2;
        end
        function copyAutoEnc(Enc,EncNew)
            Enc.layerNum=EncNew.layerNum;
            Enc.features=EncNew.features;
%             Enc.NNSAutoEnc.copyNNSAE(EncNew.NNSAutoEnc);
            Enc.NNSAutoEnc.inpDim = EncNew.NNSAutoEnc.inpDim;
            Enc.NNSAutoEnc.hidDim = EncNew.NNSAutoEnc.hidDim;
            Enc.NNSAutoEnc.inp = EncNew.NNSAutoEnc.inp;
            Enc.NNSAutoEnc.out = EncNew.NNSAutoEnc.out;
            Enc.NNSAutoEnc.g = EncNew.NNSAutoEnc.g; 
            Enc.NNSAutoEnc.h = EncNew.NNSAutoEnc.h; 
            Enc.NNSAutoEnc.a = EncNew.NNSAutoEnc.a; 
            Enc.NNSAutoEnc.b = EncNew.NNSAutoEnc.b; 
            Enc.NNSAutoEnc.W = EncNew.NNSAutoEnc.W; 
            Enc.NNSAutoEnc.lrateRO = EncNew.NNSAutoEnc.lrateRO; 
            Enc.NNSAutoEnc.regRO = EncNew.NNSAutoEnc.regRO;
            Enc.NNSAutoEnc.decayP = EncNew.NNSAutoEnc.decayP;
            Enc.NNSAutoEnc.decayN = EncNew.NNSAutoEnc.decayN;
            Enc.NNSAutoEnc.lrateIP = EncNew.NNSAutoEnc.lrateIP;
            Enc.NNSAutoEnc.meanIP = EncNew.NNSAutoEnc.meanIP;
        end
    end
    methods (Static)
        function z = zeros(varargin)
            if (nargin == 0)% For zeros('Color')
                z = autoEncoder;
            elseif any([varargin{:}] <= 0)% For zeros with any dimension <= 0
                z = autoEncoder.empty(varargin{:});
            else% For zeros(m,n,...,'Color')% Use property default values
                z = repmat(autoEncoder,varargin{:});
            end
        end
    end
end