function belief=extractFeatures(inputData,cents)
inputData=double(inputData);
cents=double(cents);
%         nd.belief=extractFeatures(nd.Input,nd.cents);
%         generate a belief vector using the triangle encoding approach(stated in the paper by Coates and Ng)
    % compute 'triangle' activation function
% Example: extracting feature for patches
    xx = sum(inputData.^2, 2);
    cc = sum(cents.^2, 2)';
    xc = inputData * cents';
    
    z = sqrt( bsxfun(@plus, cc, bsxfun(@minus, xx, 2*xc)) ); % distances
    [v,inds] = min(z,[],2);
    mu = mean(z, 2); % average distance to centroids for each patch
    belief = max(bsxfun(@minus, mu, z), 0);

end