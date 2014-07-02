classdef layer < handle
%% Nodes are replacced by autoEncoder object
%  Net.Layers(I)=layer(I,LSize,sparseFeatureSizes(I,:))
    properties(SetAccess = public, GetAccess = public)
        layerNum;
        layerSize;
        autoEncoders;
    end
    methods
        function L=layer(lNumber,LSize,visibleSize,hiddenSize)% Costructor Method
            if nargin>0
                L.layerNum=lNumber;
                L.layerSize=LSize;
                L.autoEncoders=autoEncoder.zeros(LSize(1),LSize(2));
                for I=1:LSize(1)
                    for J=1:LSize(2)
%                         EncTemp=autoEncoder(lNumber,visibleSize,hiddenSize);
                        L.autoEncoders(I,J)=autoEncoder(lNumber,visibleSize,hiddenSize);
%                         disp('layer 18')
%                         disp(size(EncTemp.W1))
%                         L.autoEncoders(I,J).copyAutoEnc(EncTemp);%.copyNode(nTemp);
                    end
                end
               % L.cents=randn(numCents,width)*0.1;
            end
        end
        function loadInput(Lay,INPUT)% feed the nodes input from lower layer
%             disp(size(Lay.autoEncoders))
%             pause
            if (Lay.layerNum==1)
                cnt1=1;
%                 cnt2=1;
                Row=4;%  
                Col=4;%
              for I=1:size(Lay.autoEncoders,1)
                  cnt2=1;
                  for J=1:size(Lay.autoEncoders,2)
                     
                      Patches=returnPatches(cnt1,cnt2,INPUT);% returns 16 by 48 pixels
                      Lay.autoEncoders(I,J).loadInput(Patches);
                       cnt2=cnt2+4;
                  end
                  cnt1=cnt1+4;
              end               
%                 pause;
%                 Row=size(INPUT,1);Col=size(INPUT,2);
%                 R=4;
%                 cnt1=0;
%                 for I=1:R:Row
%                     cnt1=cnt1+1;cnt2=0;
%                     for J=1:R:Col
%                         cnt2=cnt2+1;
% %                         size(INPUT(I:(I+R-1),J:(J+R-1),:))    (size(INPUT,3)*(R^2))
%                         Lay.autoEncoders(cnt1,cnt2).loadInput(reshape(INPUT(I:(I+R-1),J:(J+R-1),:),1,48));
% %                         Lay.autoEncoders(cnt1,cnt2).layerNum=lNumber;
% %                         L.Input(cnt1,cnt2,:)=reshape(INPUT(I:(I+R-1),J:(J+R-1),:),1,(cifarDim(3)*(R^2)));
%                     end
%                 end                
            else
%                 inputNodes=INPUT.nodes;%
                Row=size(INPUT,1);Col=size(INPUT,2);
                R=2;
%                 inTemp=zeros(Row,Col);
                for I=1:Row
                    for J=1:Col
%                         disp(Lay.layerNum)
%                         disp(size(INPUT(I,J).features))
%                         pause
                        inTemp(I,J,:)=INPUT(I,J).features;
                    end
                end
                cnt1=0;
                for I=1:R:Row
                    cnt1=cnt1+1;cnt2=0;
                    for J=1:R:Col
                        cnt2=cnt2+1;
%                         Lay.autoEncoders(cnt1,cnt2).layerNum=lNumber;
                        Lay.autoEncoders(cnt1,cnt2).loadInput(reshape(inTemp(I:(I+R-1),J:(J+R-1),:),1,(size(inTemp,3)*(R^2))));
                    end
                end                
            end
        end
    end
methods (Static)
        function z = zeros(varargin)% overloads the function zeros to create array of layer objects
            % the command layerObj.zeros(r,c) returns an r by c array of
            % layer objects
            if (nargin == 0)
                z = layer;
            elseif any([varargin{:}] <= 0)
                z = layer.empty(varargin{:});
            else
                z = repmat(layer,varargin{:});
            end
        end        

end
end