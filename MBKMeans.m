function outCents=MBKMeans(inputData,inputCents)
inputData=double(inputData);
inputCents=double(inputCents);
% Run an online K means clustering over inputData and return the updated centroids
% Step1: Compute the distance of the input from every centroid
% Step2: Update the position of the winnning centroid
% ** Starvation trace is not included at this time**
idx = findClosestCentroid(inputData, inputCents);
winCen=idx;
outCents=inputCents;
%prevCents=inputCents;
outCents(winCen,:)=inputCents(winCen,:)+(1/2)*(inputData-inputCents(winCen,:)); 
%=inputCents;
end