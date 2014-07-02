function [W1,W2,b1,b2]=sparseAutoEncoder(WW1,WW2,Bb1,Bb2,Aa1,VvisibleSize,HhiddenSize)
patches=Aa1;
visibleSize=VvisibleSize;
hiddenSize=HhiddenSize;
W1=WW1;
W2=WW2;
b1=Bb1;
b2=Bb2;
theta = [W1(:) ; W2(:) ; b1(:) ; b2(:)];
% theta;
% visibleSize = 8*8;   % number of input units 
% hiddenSize = 25;     % number of hidden units 
sparsityParam = 0.05;   % desired average activation of the hidden units.
                     % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		     %  in the lecture notes). 
% load('patches.mat');             
% patches=Aa1;
lambda = 0.0001;     % weight decay parameter       
beta = 1;            % weight of sparsity penalty term 
alpha=0.050981;

[cost, gradW1,gradW2,gradb1,gradb2] = sparseAutoencoderCost(theta,visibleSize, hiddenSize, lambda,sparsityParam, beta, patches);

W1=W1-alpha.*(gradW1+lambda.*W1);
b1=b1-alpha.*gradb1;
W2=W2-alpha.*(gradW2+lambda.*W2);
b2=b2-alpha.*gradb2;
% disp(W2)
end