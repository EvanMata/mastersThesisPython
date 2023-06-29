function [Linkage,Dist_Metric] = Cluster_Linkage(Cluster_Index,Corr_Array)

    %%%
    %Function calculates distance metric, clustering linkage and dendrogram 
    %Plots correlation map, distance metric and dendrogram
    %
    %Input: Cluster_Index: Index of 'Cluster'-struct row-entry that will be
    %       processed (only relevant for plots)
    %       Correlation: Pair correlation of all frames assigned to the
    %       cluster that will be processed.
    %       
    %Input functions: Matlab 'Statistics and Machine Learning Toolbox',
    %'scaleContrast.m'
    %
    %Output: Clustering linkage to create hierarchy, distance metric
    %%%
    
fprintf('Calculating Linkage of cluster: %d (%d Frames)\n', Cluster_Index, size(Corr_Array,1))

%% Calculate distance metric
Dist_Metric = pdist(Corr_Array,'correlation');
Dist_Metric = squareform(Dist_Metric);

%Calculate linkage of correlations distance metric
Linkage   = linkage(Dist_Metric,'average','correlation'); 

disp('Linkage calculated!')

%% Output plots
figure('Position',[5 5 1600 800],'Name',sprintf('Feedback Linkage of cluster: %d\n', Cluster_Index)');

%Correlation map
ax(1) = subplot(1,3,1);
if size(Corr_Array,1) <= 5 %Prevent contrast scaling error
    imagesc(Corr_Array,'Parent',ax(1));
else
    [lower,upper] = scaleContrast(Corr_Array(Corr_Array ~= 1),[2 97]);
    imshow(Corr_Array,[lower upper],'Parent',ax(1));
end

axis xy; colorbar; colormap parula; axis on; 
title(sprintf('Initial correlation map - cluster: %d\n',Cluster_Index));
xlabel('Data index k'); ylabel('Data index k')

%Distance metric
ax(2) = subplot(1,3,2);
if size(Corr_Array,1) <= 5 %Prevent contrast scaling error
    imagesc(Dist_Metric,'Parent',ax(2));
else
    [lower,upper] = scaleContrast(Dist_Metric(Dist_Metric ~= 0),[2 97]);
    imshow(Dist_Metric,[lower upper],'Parent',ax(2));
end

axis xy; colorbar; colormap parula; axis on; 
title(sprintf('Distance metric - cluster: %d\n',Cluster_Index));
xlabel('Data index k'); ylabel('Data index k')

%Show Dendrogram
ax(3) = subplot(1,3,3); 
dendrogram(Linkage,200,'ColorThreshold','default');
title(sprintf('Dendrogram - cluster: %d\n',Cluster_Index))
xlabel('Data index k (unordered)'); ylabel('Hierarchical distance')