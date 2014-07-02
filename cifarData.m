classdef cifarData < handle
    properties(SetAccess = public, GetAccess=public)
        cifarDim=[32,32,3];
        images;% mx3072
        labels;
        currentImgIndex;% index of the current image
        currentImg;
    end
    
    methods
        % constructor
        function cifar=cifarData(NUM)
            cifar.currentImgIndex=1;
            if NUM==10%
                STR1=load('C:\Users\Teddy\Desktop\DeepSparseNet\cifar-10-batches-mat\data_batch_1.mat');
                STR2=load('C:\Users\Teddy\Desktop\DeepSparseNet\cifar-10-batches-mat\data_batch_2.mat');
                STR3=load('C:\Users\Teddy\Desktop\DeepSparseNet\cifar-10-batches-mat\data_batch_3.mat');
                STR4=load('C:\Users\Teddy\Desktop\DeepSparseNet\cifar-10-batches-mat\data_batch_4.mat');
                STR5=load('C:\Users\Teddy\Desktop\DeepSparseNet\cifar-10-batches-mat\data_batch_5.mat');
                STR.data=double([STR1.data;STR2.data;STR3.data;STR4.data;STR5.data]);
                STR.labels=[STR1.labels;STR2.labels;STR3.labels;STR4.labels;STR5.labels];
                clear STR1 STR2 STR3 STR4 STR5
            elseif NUM==6
                STR=load('C:\Users\Teddy\Desktop\DeepSparseNet\cifar-10-batches-mat\test_batch.mat');
                STR.labels=[STR.labels];%STR2.labels;STR3.labels;STR4.labels;STR5.labels];
            else
                STR=load(strcat('C:\Users\Teddy\Desktop\DeepSparseNet\cifar-10-batches-mat\data_batch_',int2str(NUM),'.mat'));
                STR.labels=[STR.labels];
            end
            STR.data=double(STR.data);
            pmean=mean(STR.data,2);
            pvar=sqrt(var(STR.data,[],2)+10);
%             save 'mean_and_var' pmean pvar
            STR.data = bsxfun(@rdivide, bsxfun(@minus, STR.data, pmean), sqrt(pvar));
            STR.data = bsxfun(@rdivide, bsxfun(@minus, STR.data, mean(STR.data,2)), sqrt(var(STR.data,[],2)+10));
            M=mean(STR.data);
            C=cov(STR.data);
            [V,D] = eig(C);
            P = V * diag(sqrt(1./(diag(D) + 0.1))) * V';
            STR.data = bsxfun(@minus, STR.data, M) * P;            
            cifar.images=STR.data;
            cifar.labels=STR.labels;
            cifar.currentImg=STR.data(1,:);
            clear STR
%             currentImg=;
%             clear STR;
        end
        % increment index of the current image
        function findNextImg(cifar)
            if(cifar.currentImgIndex>=size(cifar.images,1))
                cifar.currentImgIndex=mod(cifar.currentImgIndex,size(cifar.images,1));
            end
            cifar.currentImgIndex=cifar.currentImgIndex+1;
            cifar.currentImg=reshape(cifar.images(cifar.currentImgIndex,:),cifar.cifarDim);
        end
        % returns the index of the current image
        % current image refers to the image over which the algorithm is
        % currently running
        function IDX=getCurrentImgIndex(cifar)
            IDX=cifar.currentImgIndex;
        end
        % returns the current image 
        function IMAGE=getCurrentImg(cifar)
            IMAGE=cifar.currentImg;
        end
        function destructCifarData(cifar)
            cifar.images=[];
            cifar.currentImg=[];
        end
        
    end
end