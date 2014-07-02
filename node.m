classdef node < handle

properties(SetAccess = public)
    %kMeansIterations=50;
    InputDim;
    %AutoEncoderParameters
    visibleSize;
    hiddenSize;
    W1;
    W2;
    layerNum=0;
    Input;
    nodePos; % row and column values 
    beliefDim;
    beliefStore;
    centDim;
    belief;
    cents;
end

methods
            % initialize centroids
        function nd=node(numCent,width,LayerNum,pos1,pos2,visibleSize,hiddenSize)% a constructor function
%             if (nargin<3)
%                 LayerNum=0;
%                 disp('enter a valid layer number!');
%             end
            if nargin>0
                nd.visibleSize=visibleSize;
                nd.hiddenSize=hiddenSize;
                r  = sqrt(6) / sqrt(hiddenSize+visibleSize+1);   % we'll choose weights uniformly from the interval [-r, r]
                nd.W1 = rand(hiddenSize, visibleSize) * 2 * r - r;
                nd.W2 = rand(visibleSize, hiddenSize) * 2 * r - r;
                nd.nodePos=[pos1,pos2];
                nd.cents=randn(numCent,width)*0.1;
                nd.centDim=[numCent,width];
                nd.beliefDim=[numCent 1];
                nd.belief=ones(numCent,1)./length(nd.belief);
                nd.layerNum=LayerNum;
            end
        end
        function loadInput(nd,Inp)
            if nd.layerNum>1
                Inp=Inp(:)';
                Inp = bsxfun(@rdivide, bsxfun(@minus, Inp, mean(Inp,2)), sqrt(var(Inp,[],2)+10));
            end
            nd.Input=Inp;
        end
        function doClustering(nd)
        nd.cents=MBKMeans(nd.Input,nd.cents);
        end
        function doFeatureExtraction(nd)
        nd.belief=extractFeatures(nd.Input,nd.cents);
        end
        function clearBeliefs(nd)
            nd.belief=ones(nd.centDim)./length(nd.belief);
            % re-initialize node beliefs
        end
        function copyNode(nd,nd2)
            nd.InputDim=nd2.InputDim;
            nd.layerNum=nd2.layerNum;
            nd.Input=nd2.Input;
            nd.nodePos=nd2.nodePos; % row and column values
            nd.beliefDim=nd2.beliefDim;
            nd.centDim=nd2.centDim;
            nd.belief=nd2.belief;
            nd.cents=nd2.cents;
        end
end
methods (Static)
    % operator overloading over zeros() function; this makes the function
    % zeros() able to create an array of nodeObjects
        function z = zeros(varargin)
            if (nargin == 0)% For zeros('Color')
                z = node;
            elseif any([varargin{:}] <= 0)% For zeros with any dimension <= 0
                z = node.empty(varargin{:});
            else% For zeros(m,n,...,'Color')% Use property default values
                z = repmat(node,varargin{:});
            end
        end         
      
end
end