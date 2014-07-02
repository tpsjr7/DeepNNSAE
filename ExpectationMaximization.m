function centroids=ExpectationMaximization(patches,centroids,iteration)
if nargin<3
    iteration=1;
end
patches=patches(:)';
V=var(patches,[],2);

for iter=1:iteration
    Ex=zeros(size(patches,1),size(centroids,1));
    for i=1:size(patches,1)
      for j=1:size(centroids,1)
          
        Ex(i,j)=exp(-0.5*V(i)*(sum(patches(i,:))-sum(centroids(j,:))).^2);
        
      end
    s=sum(Ex(i,:));
    Ex(i,:)=Ex(i,:)/s;
    clear s;
    end

    %Maximization
    for j=1:size(centroids,1)
%         size(patches,1)
      for i=1:size(patches,1)
        cent=Ex(i,j).*patches(i);  
      end
      centroids(j)=sum(cent)/sum(Ex(i,j));
      clear cent;
    end
    clear  Ex i j;
end
clear V;
end