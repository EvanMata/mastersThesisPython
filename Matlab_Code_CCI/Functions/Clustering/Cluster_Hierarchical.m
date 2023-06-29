function [Cluster_Vec,Cluster_matrix] = Cluster_Hierachical(Linkage,ClusteringOption,Parameter)

    %%%
    %Function uses linkage tree to construct a given number of cluster 
    %(Option: 'maxcluster') or by cutting the tree at a given height
    %(Option: 'cutoff')
    %
    %Input: Linkage: Hierarchical linkage tree
    %       ClusteringOption: Select method to construct cluster:
    %           - 'cutoff': cut dendrogram at height 
    %           - 'maxclust': extract number of clusters
    %       Parameter: depends on ClusteringOption: Cutting height value 
    %           or number of clusters
    %       
    %Input functions: Matlab 'Statistics and Machine Learning Toolbox'
    %
    %Output: Cluster_Vec: Assignment of cluster to data index
    %        Cluster_Matrix: Binary assignment matrix [m x n]. If entry 
    %        (m,n) is '1' then data with index n is assigned to cluster m.
    %           - m: Cluster index 
    %           - n: Data index
    %        Example: Cluster_Vec: [1 2 1] = Cluster_Matrix: [1 0 1; 0 1 0]
    %%%

%Construct clusters, use either 'cutoff' or 'maxcluster' as options
Cluster_Vec = cluster(Linkage,ClusteringOption,Parameter,'criterion','distance');

%Get Nr of Cluster and get binary mask which encodes cluster
Cluster_Label = unique(Cluster_Vec);    %Label of different clusters
Cluster_matrix = zeros(length(Cluster_Label),size(Linkage,1)+1);

%Vary clusters
for L = 1:length(Cluster_Label)
    Cluster_matrix(L,:) = (Cluster_Vec == Cluster_Label(L));
end

fprintf(1,'%s Clusters were constructed!\n',string(length(Cluster_Label)));

%Plot cluster assignment (Row = Cluster assignment; column = data index)
figure;
ax = gca;
imagesc(Cluster_matrix); axis xy; 
xlabel('Data index'); ylabel('State assignment')

ax.YAxis.TickValues = [1:1:size(Cluster_matrix,1)];
hold off