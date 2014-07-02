function [cost,W1grad W2grad,b1grad,b2grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 
% pause
% hiddenSize
% pause
% visibleSize
% pause
W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 
m=size(data,2);
rho=zeros(size(b1))+0.000004;
m=1;
% for i=1:m
    %feedforward
%     disp('data')
%     disp(size(data))
    a1=data(:);
%     size(W1)
%     size(a1)
%     size(b1)
    z2=W1*a1+b1;
    a2=sigmoid(z2);
    z3=W2*a2+b2;
    a3=sigmoid(z3);
    %cost=cost+(a1-a3)'*(a1-a3)*0.5;
    rho=rho+a2;
% end
rho=rho/m;
sterm=beta*(-sparsityParam./rho+(1-sparsityParam)./(1-rho));
%sterm=beta*2*rho;
% for i=1:m
    %feedforward
    a1=data(:);
%     size(W1)
%     size(a1)
%     size(b1)    
    z2=W1*a1+b1;
    a2=sigmoid(z2);
    z3=W2*a2+b2;
    a3=sigmoid(z3);
    cost=cost+(a1-a3)'*(a1-a3)*0.5;
    %backpropagation
    delta3=(a3-a1).*a3.*(1-a3);
    delta2=(W2'*delta3+sterm).*a2.*(1-a2);
    W2grad=W2grad+delta3*a2';
    b2grad=b2grad+delta3;
    W1grad=W1grad+delta2*a1';
    b1grad=b1grad+delta2;
% end

kl=sparsityParam*log(sparsityParam./rho)+(1-sparsityParam)*log((1-sparsityParam)./(1-rho));
%kl=rho.^2;
cost=cost/m;
cost=cost+sum(sum(W1.^2))*lambda/2.0+sum(sum(W2.^2))*lambda/2.0+beta*sum(kl);
W2grad=W2grad./m+lambda*W2;
b2grad=b2grad./m;
W1grad=W1grad./m+lambda*W1;
b1grad=b1grad./m;

end