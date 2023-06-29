function Cluster = Process_Cluster(Cluster,Cluster_Index,Correlation,Cluster_Matrix,Save,Plot)

    %%%
    %Function processes a given cluster assignment and adds the new
    %subclusters to the 'Cluster'-struct to store information. Shows also
    %correlation map, distance metric and resulting hierarchy of
    %subclusters as feedback. 
    %
    %Input: Cluster: 'Cluster'-struct that stores the clustering 
    %       data
    %       Cluster_Index: Index of cluster row-entry that will be
    %       processed
    %       Correlation: Pair correlation of all frames assigned to the
    %       initial cluster that will be processed.
    %       Cluster_Matrix: Cluster to frame assignment matrix
    %       Plot: Activate or disable output plots
    %       
    %Input functions: Matlab 'Statistics and Machine Learning Toolbox', 
    %'scaleContrast.m'
    %
    %Output: new 'Cluster'-Struct with appended sub-clusters
    %%%


fprintf('Processing cluster: %d\n', Cluster_Index)
    
%Initial frames assigned to the cluster that will be processed
Frame_Tracking = Cluster(Cluster_Index).Frame_Index;
    
%Initial length of cluster; used for appending new sub-clusters
Offset = length(Cluster);

%Varies sub-clusters
for ii = 1:size(Cluster_Matrix,1) 
    fprintf(1,'Creating sub-cluster: %d-%d\n',Cluster_Index,ii);
    
    %% Create mask which selects the section of the correlation that is
    %assigned to sub-cluster ii
    temp_mask = Cluster_Matrix(ii,:).*ones(size(Cluster_Matrix,2),'single');
    temp_mask = temp_mask.*transpose(Cluster_Matrix(ii,:));
    
    %Apply Mask to initial correlation map for feedback
    temp_corr = temp_mask.*Correlation;
    
    %% Show correlation of sub-cluster
    if Plot == 1
        figure('Position',[5 5 1600 800],'Name',sprintf('Feedback sub-cluster: %d',ii)');
        ax(1) = subplot(2,2,1);

        if sum(Cluster_Matrix(ii,:)) <= 4 %Prevent contrast scaling error
            imagesc(temp_corr,'Parent',ax(1)); axis xy; colorbar; colormap parula; axis on; 
        else
            [lower,upper] = scaleContrast(temp_corr(temp_corr ~= 1),[2 97]);
            imshow(temp_corr,[lower upper],'Parent',ax(1)); axis xy; colorbar; colormap parula; axis on
        end

        %Plot Settings
        title(sprintf('Section initial correlation map - subcluster: %d-%d\n',Cluster_Index,ii));
        xlabel('Data index k'); ylabel('Data index k')
    end
    
    %% Shrink array by deleting of rows and columns that are set to zero by
    %multiplication with temp mask
    temp_corr(:,~any(temp_corr,1)) = [];  %columns
    temp_corr(~any(temp_corr,2),:) = [];  %rows
    
    if Plot == 1
        %Show shrinked correlation map
        ax(2) = subplot(2,2,2);

        if sum(Cluster_Matrix(ii,:)) <= 5 %Prevent contrast scaling error
            imagesc(temp_corr,'Parent',ax(2)); axis xy; colorbar; colormap parula; axis on
        else
            [lower,upper] = scaleContrast(temp_corr(temp_corr ~= 1),[2 97]);
            imshow(temp_corr,[lower upper],'Parent',ax(2)); axis xy; colorbar; colormap parula; axis on
        end

        %Plot Settings
        title(sprintf('Correlation map - subcluster: %d-%d\n',Cluster_Index,ii))
        xlabel('Data index k-'); ylabel('Data index k-')
    end
    
    %% Calculate sub-cluster distance metric
    
    if Plot == 1
        temp_distance = pdist(temp_corr,'correlation');
        temp_distance = squareform(temp_distance);

        %Show distance metric of shrinked correlation map
        ax(3) = subplot(2,2,3);

        if sum(Cluster_Matrix(ii,:)) <= 5 %Prevent contrast scaling error
            imagesc(temp_distance,'Parent',ax(3)); axis xy; colorbar; colormap parula; axis on
        else
            [lower,upper] = scaleContrast(temp_distance(temp_distance ~= 1),[2 97]);
            imshow(temp_distance,[lower upper],'Parent',ax(3)); axis xy; colorbar; colormap parula; axis on
        end

        %Plot Settings
        title(sprintf('Distance metric - subcluster: %d-%d\n',Cluster_Index,ii))
        xlabel('Data index k-'); ylabel('Data index k-')
    end 
    %% Calculate Linkage of sub-cluster
    if Plot == 1
        if size(temp_distance,1) > 1
            temp_linkage = linkage(temp_distance,'average','correlation');
        
        
            %Show Dendrogram
            ax(4) = subplot(2,2,4); dendrogram(temp_linkage,200,'ColorThreshold','default');
            title(sprintf('Dendrogram - subcluster: %d-%d\n',Cluster_Index,ii))
            xlabel('Data index k- (unordered)'); ylabel('Hierarchical distance')
        end
    end
    %% delete 0 rows of mask, first row/column of mask is assignment to
    %sub-cluster ii
    temp_mask(~any(temp_mask,2),:) = [];  %rows 
    
    %Convert correlation map to vector for AVG and STD calculation
    temp_corr = temp_corr(:);

    %% Append sub-cluster to 'Cluster'-Struct (Stores subcluster)
    if Save == 1
        temp_Vector = temp_mask(1,:).*Frame_Tracking;
        temp_Vector(temp_Vector==0) = [];
        Cluster(ii+Offset).Frame_Index = temp_Vector;

        Cluster(ii+Offset).Corr_Avg = sum(temp_corr)/length(temp_corr);
        Cluster(ii+Offset).Corr_STD = std(temp_corr);
        
        fprintf('Saving subcluster %d-%d as new cluster %d\n',Cluster_Index, ii, ii+Offset)
   	else 
         disp('Saving sub-cluster disabled!')
  	end
end

%Delete old cluster information from 'Cluster'-struct
if Save == 1
    Cluster(Cluster_Index).Frame_Index = []; 
    Cluster(Cluster_Index).Corr_Avg = []; 
    Cluster(Cluster_Index).Corr_STD = [];
end
disp('Cluster processed!')