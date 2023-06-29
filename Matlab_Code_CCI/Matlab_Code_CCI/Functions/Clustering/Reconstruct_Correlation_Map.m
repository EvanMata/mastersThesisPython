function [temp_corr] = Reconstruct_Correlation_Map(Cluster_Nr,Cluster,Corr_Array)

    %%%
    %Script Reconstruct the cluster's correlation map from the given
    %cluster's 'Frame_Index'-Vector and the (large) correlation map of all
    %frames.
    %
    %Input: - Cluster_Index: 
    %       - Cluster: 'Cluster'-struct saving all cluster information
    %       - Corr_Array: Correlation map of all frames
    %
    %Output: - temp_corr: Cluster's correlation map of relevant frames
    %%%

fprintf('Reconstructing correlation map of cluster: %d\n',Cluster_Nr)

%Get all Frames
Frame_Vec = Cluster(Cluster_Nr).Frame_Index;

%Setup new array
sz = size(Corr_Array);

%Get all permutations of frames representing all possible correlation pairs
[A,B] = meshgrid(Frame_Vec,Frame_Vec);
c=cat(2,A',B');
d=reshape(c,[],2);

%Get Index for direct indexing of correlation full
idx = sub2ind(sz,d(:,1),d(:,2));
temp_Vec = Corr_Array(idx);

%Reconstruct Cluster
temp_corr = reshape(temp_Vec,[length(Frame_Vec) length(Frame_Vec)]);
disp('Correlation Map Reconstructed!') 