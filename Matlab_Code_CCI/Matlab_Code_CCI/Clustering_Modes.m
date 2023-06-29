%% Clustering_Analysis
    
    %%%
    %Script is clustering analysis of the correlation map to find
    %coincident magnetic domain modes of hologram frames. Clustering is 
    %based on Matlabs hierarchical clustering algorithm and the correlation
    %map.
    %
    %Clustering approach is based on distance metric and hierarchical
    %clustering algorithm. In brief, the following procedures are applied
    %iteratively:
    %
    %1: Calculate distance metric from correlation map.
    %2: Calculate Linkage tree (agglomerative hierarchical clustering tree) 
    %from distance metric
    %3: Plot dendrogram. If two identified sub-cluster represent pure or
    %mix-states, i.e., if they comprise frames from a single or multiple
    %states. Such natural divisions are indicated by large steps in the
    %dendrogram, also know as inconsistencies. We aim to make a robust
    %classification and therefore split the tree mostly at the top node
    %into 2 sub-clusters.
    %6: Decide if pair-correlation maps of sub-cluster are
    %pure or mixed-state clusters based on inconsistency of top-most node
    % If inconsistency value is below threshold then clusters are 
    % considered as pure state cluster. Pure-state cluster are 
    %identified as homogeneous correlation maps where any deviation from 
    %the average is just random noise.
    %7: Mixed-State cluster are clustered again starting step 1, the
    %calculation of the distance metric, but now using the mixed-state
    %clusters correlation map. 
    
    %Comment: Script loads 6.6 GB of data arrays. Runs with 16 GB RAM. 
    %Using less RAM requires code optimization.
    %   
    %Input Data: 
    % - Correlation Map
    % .../Cluster_Analysis/Correlation_Map.bin
    % - Initial distance metric
    % .../Cluster_Analysis/Initial_Dist_Metric.bin
    % - Initial Linkage Tree
    % .../Cluster_Analysis/Initial_Linkage.bin
    %
    %Input Functions: Matlab "Statistics and Machine Learning Toolbox", 
    %"Image Processing Toolbox", Cluster_Hierarchical.m, Cluster_Linkage.m, 
    %cluster_monitoring.m, Process_Cluster.m, Reconstruct_Correlation_Map.m
    %
    %Output: Mode assignment
    %
    %Code developed and tested with: Matlab Version 2018a, Statistics and
    %Machine Learning Toolbox Version 11.3, Image Processing Toolbox
    %Version 10.2
    %%%

%% Declaration of Path and Files
%Basic Directories (set Current Folder to Zenodo Export Folder)
workingDir = split(pwd,'\')
lstBaseDir = workingDir(1:length(workingDir)-1)
basePathCell = join(lstBaseDir, '\')
basePath = basePathCell{1}
baseFolder.Data     = fullfile(basePath,'Clustering_Analysis');             %Input Folder of 28800x28800 correlation map
baseFolder.Log      = fullfile(basePath);                                   %Input Folder of Log
addpath(genpath(pwd));

%baseFolder.Data     = fullfile(pwd,'Clustering_Analysis');                  %Input Folder of 28800x28800 correlation map
%baseFolder.Log      = fullfile(pwd);                                        %Input Folder of Log

%% Load Log
%Stores all relevant information
Log = readtable(fullfile(baseFolder.Log,'Log_Data_Coherent_Correlation_Imaging.txt'));

%% Load correlation map

    %%%
    %Correlation map is quite large [28800,28800] and needs 6.6GB of RAM 
    %with 'double' precision. Data will be converted to 'single' precision
    %to reduce the RAM requirement. 
    %%%

%Load
FileName = fullfile(baseFolder.Data,'Correlation_Map.bin');
fprintf(1, 'Now reading %s\n', FileName);
fid = fopen(FileName,'r');
Correlation = fread(fid,[28800,28800],'double');
fclose(fid);

%Convert correlation map to data type 'single' to save memory
Correlation = single(Correlation);

disp('Loading correlation metric finished!')
f = figure; ax = gca; imagesc(Correlation,'Parent', ax); axis image; axis xy; colorbar; title('Initial correlation map')
xlabel('Frame index'); ylabel('Frame index')

%% Comment: Load initial distance metric and initial linkage

    %%%
    %Calculation of the inital distance metric and the linkage tree is 
    %computation-intensive because the input correlation map is at maximum 
    %size. They were also exported to the clustering folder, so just load
    %them to save some time. The size of the sub-cluster correlation maps
    %reduces, and therefore the computational effort, with each iteration.
    %%%

%% Calculate initial linkage

% %We find that the damping of the autocorrelation (that is 1 per definition)
% %to the values that noisy, same-state single frame achieve improves the
% %separability of the first iteration.
% 
% Damping = 0.35;
% 
% %Calc Distance Metric and Linkage
% Dist_Metric  = pdist(Correlation-Damping*eye(size(Correlation)),'correlation');
% Linkage   = linkage(Dist_Metric,'average'); 

%% Load initial distance metric

%No need to load the initial distance metric, because linkage is already
%calculated. Except you want to see how the metric looks like. 

% FileName = fullfile(baseFolder.Data,'Initial_Dist_Metric.bin');
% fprintf(1, 'Now reading %s\n', FileName);
% fid = fopen(FileName,'r');
% Dist_Metric = fread(fid,[28800,28800],'double');
% fclose(fid);
% 
% %Convert distance metric to data type 'single' to save memory
% Dist_Metric = single(Dist_Metric);
% 
% disp('Loading distance metric finished!')
% imagesc(Dist_Metric,'Parent', ax); axis image; axis xy; colorbar; title('Initial distance metric')
% xlabel('Frame index'); ylabel('Frame index')

%% Load initial linkage

FileName = fullfile(baseFolder.Data,'Initial_Linkage.bin');
fprintf(1, 'Now reading %s\n', FileName);
fid = fopen(FileName,'r');
Linkage = fread(fid,[28799,3],'float64');
fclose(fid);

disp('Loading linkage finished!')

%% Clustering
close all

    %%%
    %Construct a given number of clusters (Option: 'maxcluster') or cut 
    %the dendrogram at a given height to construct clusters (Option: 
    %'cutoff'). Creates a cluster assignment.
    %
    %'Cluster_Index' is index setting cluster to analyze 
    %Cell Struct 'Cluster' will store all relevant clustering data. 
    %That is:
    %   - Frame_Index: List of all frames assigned to the cluster
    %   - Corr_Avg: Averaged pair-correlations between all frames assigned
    %   to the cluster.
    %   - Corr_STD: Standard deviation of the pair-correlation between all
    %   frames assigned to the cluster.
    %%%

%Struct that stores all relevant information of the clustering
clearvars Cluster
Cluster(1) = struct();

%Cluster_Index = i will get Cluster 'i' from struct as an iterator
Cluster_Index = 1;  %Analyze first cluster

%see above
Cluster(Cluster_Index).Frame_Index = 1:size(Correlation,1); %Initial cluster contains all frames
Cluster(Cluster_Index).Corr_Avg = [];   %Irrelevant for first cluster
Cluster(Cluster_Index).Corr_STD = [];   %Irrelevant for first cluster

%Show Dendrogram
figure('Position',[0 200 1000 800]); dendrogram(Linkage,500,'ColorThreshold','default'); 
xlabel('Data index k (unordered)'); ylabel('Hierarchical distance')

%Construct cluster and create frame to cluster assignment
[Cluster_Vec,Cluster_Matrix] = Cluster_Hierarchical(Linkage,'maxclust',2);

%Cluster_Matrix: Binary assignment matrix [m x n]. If entry 
%        (m,n) is '1' then data with index n is assigned to new cluster m.
%           - m: Cluster index 
%           - n: Data index
%        Example: Cluster_Vec: [1 2 1] = Cluster_Matrix: [1 0 1; 0 1 0]
%
%However, Cluster_Vec is no needed in this Version of the script

%% Process sub-clusters 
%Processing takes some time because the correlation and distance arrays are
%Still relatively large
close all

Plot = 0;
%Save = 1 appends new sub-clusters to 'Cluster'-struct
Save = 1;
Cluster = Process_Cluster(Cluster,Cluster_Index,Correlation,Cluster_Matrix,Save,Plot);

%Delete old cluster information from 'Cluster'-struct
if Save == 1
    Cluster(Cluster_Index).Frame_Index = []; 
    Cluster(Cluster_Index).Corr_Avg; 
    Cluster(Cluster_Index).Corr_STD = [];
end

%Based on the feedback plots, new Clusters 2 and 3 are mixed-states and 
%must be iterated again

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           Start of Iterations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Load 'Cluster'-struct containing the frame to cluster assignment

    %%%
    %Calculation of the distance metric and the linkage tree for first 
    % few iterations is computation-intensive because the input correlation 
    % maps of the clusters are still large. I calculated the first 4 
    % iterations in advance to save you computation time. Clustering 
    % algorithm continues with iteration step 5 (Cluster_Index = 5). 
    % You can also start with the first automated iteration if you want. 
    % Just skip this cell and set Cluster_Index = 2
    %%%

MatFileName = fullfile(baseFolder.Data,'Cluster_initial.mat');
fprintf(1,'Now loading %s\n', MatFileName);
load(MatFileName,'Initial_Cluster');
disp('Loading is done!')
    
%Cluster = Initial_Cluster;

%% Save 'Cluster'-struct containing the frame to cluster assignment
MatFileName = fullfile(baseFolder.Data,'Cluster_initial_2.mat');
fprintf(1,'Now loading %s\n', MatFileName);
save(MatFileName,'Cluster');
disp('Saving is done!')

%% Automated Iteration loop

%Our analysis starts with the pair-correlation. First step is to reduce the 
%noise of this metric. We therefore calculate the low-noise fourth-order 
%Pearson distance metric. 
%
%Our second iteration step is to employ an agglomerative UPGMA (unweighted 
%pair group method with arithmetic mean) algorithm to calculate a 
%hierarchical cluster tree (Linkage). In this bottom-up process, the 
%nearest two clusters are combined into a higher-level cluster, starting 
%with single frames on the lowest level and ending with a single final 
%cluster. The vertical level, or height, of each link in the dendrogram 
%corresponds to the distance between the linked clusters. If two linked 
%clusters represent the same physical state of the sample, the step size, 
%or height difference, between the old and the new hierarchy level is 
%small. Conversely, a large step, also known as an inconsistency, indicates
%a natural division in the data set, i.e., the merging of different 
%physical states. The largest step size is naturally observed at the 
%top-most node in the tree. We aim to make the most robust classification 
%and therefore split the tree only at the top node into two sub-clusters.
%
% Our third iteration step is to decide if our two identified sub-clusters 
%represent pure or mixed-states, i.e., if they comprise frames from a 
%single state or multiple states. To this end, we quantify the distinctness
%of the top-most link in the dendrogram by the inconsistency coefficient ð?œ‰,
%where ð?œ‰ is given as the ratio of the average absolute deviation of the 
%step height to the standard deviation of all lower-level links included in
%the calculation. On this basis, we identify all clusters that undercut an 
%inconistency threshold (here: Thres=1.85) as pure-state clusters while, in
%complementary, mixed states are present inclusters exceeding this 
%threshold. Iteration steps one to three are repeated for mixed-state 
%clusters until â€’ within our temporal resolution and available 
%signal-to-noise ratio â€’ complete separation into clusters containing only 
%a pure state is achieved.


% Plotting enabled if Plot == 1
Plot = 1;
Cluster_Index = 2; %Starting with index XX of Cluster
Thres = 1.85;

%Loop
stop = 0; %reset stop condition
while stop == 0 
    %% Reconstruct Clusters correlation map
    close all

    %Reconstructs section of correlation map that corresponds to frames of
    %given cluster. 
    [temp_corr] = Reconstruct_Correlation_Map(Cluster_Index,Cluster,Correlation);
    
    %Single frames can't be separated anymore
    if size(temp_corr,1) > 1
        %% Create Linkage
        %Calculates distance metric and hierarchical tree
        [Linkage,temp_dist] = Cluster_Linkage(Cluster_Index,temp_corr);

        %% Get feedback to decide if reclustering is necessary
        disp('Calculating cluster monitoring feedback!')

        %Ouput some parameter for monitoring of cluster, e.g. inconsistency
        temp_struct = cluster_monitoring(temp_corr,temp_dist,Linkage,Plot);

        Cluster_monitoring(Cluster_Index).STD_corr = temp_struct.STD_corr;
        Cluster_monitoring(Cluster_Index).STD_dist = temp_struct.STD_dist;
        Cluster_monitoring(Cluster_Index).Inconsistency = temp_struct.Inconsistency;

        %Relevant criteria to distinguish pure- and mixed-state cluster
        Cluster(Cluster_Index).Inconsistency = temp_struct.Inconsistency(end);

        %% Decision about reclustering

        %Recluster if inconsistency exceeds threshold. Here: Thres = 1.85
        if Cluster_monitoring(Cluster_Index).Inconsistency(4) > Thres
            disp('Reclustering condition satisfied!')
            %% Construct cluster and get cluster assignment  
            %Separates dendrogram at top-most node and to create two
            %sub-clusters
            [~,Cluster_Matrix] = Cluster_Hierarchical(Linkage,'maxclust',2);

            %% Process sub-clusters 
            %Save = 1 appends new sub-clusters to 'Cluster'-struct and
            %deletes old one Cluster(Cluster_Index)
            Save = 1;
            Cluster = Process_Cluster(Cluster,Cluster_Index,temp_corr,Cluster_Matrix,Save,Plot);

        else
            %Skip if it is a pure state cluster
            fprintf(1,'Cluster %d does not satisfy reclustering condition!\n',Cluster_Index)
        end
    else
        %Skip single frames
        disp('Small cluster size: Skipping!')
    end
    %% Update Cluster index
    Cluster_Index = Cluster_Index + 1;
    
    %Cancels while loop when all clusters are pure states clusters
    if Cluster_Index > length(Cluster)
        stop = 1;
        
        disp('End of cluster: Iteration stopps!')
    end
    
    %Next step when last figure is closed. Uncomment if you want to check
    %each step.

    %k=gcf;
    %waitfor(k)
    disp(' ')
end


disp('Iterative clustering algorithm finished!')

%% Downsize cluster
%Removes empty cells of iterated cluster from "Cluster"-Struct
Cluster = Cluster(~cellfun(@isempty,{Cluster.Frame_Index}));

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Calculate Fourier space pair-correlations of clusters

    %%%
    %Cluster pair correlation between cluster A and B is calculated by, 
    %first, reading the cluster's frame assignments, second, extracting the
    %pair correlation between all frames of cluster A and all frames of
    %cluster B (each combination/permutation of the index frames) and,
    %third, averaging this whole set of pair correlations.
    %%%

%Define Correlation Matrix. Autocorrelation is always 1
Cluster_Correlation_Fourier = eye(length(Cluster));

%Vary clusters, Loops considers symmetry of correlation map
for L = 1:(length(Cluster)-1) %Varies first cluster
    fprintf(1,'Calculating Mean with Cluster %d as Reference!\n', L)
    
    %Get frame indices of first cluster
    Frame_Vec_1 = Cluster(L).Frame_Index;

    %Vary second cluster, skip autocorrelation
    for k = (L+1):length(Cluster) 
        %Get frame indices of second cluster
        Frame_Vec_2 = Cluster(k).Frame_Index;
        
        %Get all possible combinations/permutations of frame indices
        [A,B] = meshgrid(Frame_Vec_1,Frame_Vec_2);
        c=cat(2,A',B');
        d=reshape(c,[],2);
        
        %Get Index for direct indexing of original, large correlation map
        idx = sub2ind(size(Correlation),d(:,1),d(:,2));
        temp_Vec = Correlation(idx);
        
        %Calc Mean and assign to Cluster correlation matrix
        temp = mean(temp_Vec);
        Cluster_Correlation_Fourier(L,k) = temp;
        Cluster_Correlation_Fourier(k,L) = temp;
    end
end

%Show cluster correlation map
figure; imshow(Cluster_Correlation_Fourier,[-0.2555 0.75]); axis image; axis xy; 
colormap parula; colorbar; title('Cluster correlation map'); xlabel('Cluster')
ylabel('Cluster')

disp('Cluster correlation map calculated!')

%% Further clustering to group the clusters further into domain modes
%close all

    %%%
    %Clustering analysis to group the clusters which actually represent 
    %the same domain configurations (and are, therefore, correlated) into
    %unique domain configuration, which we define as domain modes.
    %The Clustering method is the same as before but this iteration loop
    %aims to prevent the merging of inconsistent clusters, and insufficient 
    %grouping of cluster representing the same domain configurations. The 
    %reiteration focuses therefore on subsets of the most similar clusters 
    %determined in the previous iteration loop. 
    %
    %The subsets are determined from the hierarchy given by the 
    %Fourier-space correlation between all cluster calculated in the 
    %previous cell . These subsets are the different "branches".
    %
    %The global 'cutoff' value defines the hierarchical distance cutting the
    %dendrogram at the given height to separate dendrogram branches.
    %The dendrogram is separated into branches, which will be processed 
    %further individually. All frames of a branch are combined again into a 
    %single cluster as a preselection of frames and the iteratively
    %separated into pure states again.
    %%%
    
%Distance to cut the dendrogram and separate branches.    
Cutoff = 0.15; 

%Calculate linkage of cluster pair-correlations and show dendrogram
Linkage   = linkage(Cluster_Correlation_Fourier,'complete','correlation'); 
f = figure; dendrogram(Linkage,200,'ColorThreshold','default');

%Visualize cutoff value in dendrogram
line([0,150],[Cutoff,Cutoff])

%Separates clustering branches
[Cluster_Vec,Cluster_Matrix] = Cluster_Hierarchical(Linkage,'cutoff',Cutoff);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Create new cell of structs which combines frames of previous higher order clusters

%Setup cell to store cluster of each branch for further reclustering
Recluster = {};
Recluster_monitoring = {};

%Vary branches
for L = 1:size(Cluster_Matrix,1)
    cluster_idx = find(Cluster_Matrix(L,:) == 1);
    Frames = [];
    
    %Combine all frames of the branch
    for k = cluster_idx
        Frames = horzcat(Frames,Cluster(k).Frame_Index);
    end
    
    %Store as new cluster
    temp_struct.Frame_Index = Frames;
    Recluster{L,1} = temp_struct;
end

%% Automated Iteration loop

%%%
%Same procedure as above with the same threshold
%%%

%Set
Plot = 1;

%Loop
for Re = 1:size(Recluster,1)
    
    fprintf('Analysis of Recluster: %1.0d\n',Re)
    
    %Stop condition
    stop = 0;
    Cluster_Index = 1;
    
    Cluster_re = Recluster{Re,1};
    Cluster_monitoring_re = struct();
    
    while stop == 0 
        %% Reconstruct Clusters correlation map
        close all
        [temp_corr] = Reconstruct_Correlation_Map(Cluster_Index,Cluster_re,Correlation);

        if size(temp_corr,1) > 1
            %% Create Linkage
            [Linkage,temp_dist] = Cluster_Linkage(Cluster_Index,temp_corr);

            %% Get feedback to decide if reclustering is necessary
            disp('Calculating cluster monitoring feedback!')
            temp_struct = cluster_monitoring(temp_corr,temp_dist,Linkage,Plot);

            Cluster_monitoring_re(Cluster_Index).STD_corr = temp_struct.STD_corr;
            Cluster_monitoring_re(Cluster_Index).STD_dist = temp_struct.STD_dist;
            Cluster_monitoring_re(Cluster_Index).Inconsistency = temp_struct.Inconsistency;

            %% Decision about reclustering

            %Recluster
            if Cluster_monitoring_re(Cluster_Index).Inconsistency(4) > Thres
                disp('Reclustering condition satisfied!')
                %% Construct cluster and get cluster assignment  
                [~,Cluster_Matrix] = Cluster_Hierarchical(Linkage,'maxclust',2);

                %% Process sub-clusters 
                Save = 1; %appends new sub-clusters to 'Cluster'-struct
                Cluster_re = Process_Cluster(Cluster_re,Cluster_Index,temp_corr,Cluster_Matrix,Save,Plot);
                Cluster_re(Cluster_Index).Inconsistency = Cluster_monitoring_re(Cluster_Index).Inconsistency;
            else
                fprintf(1,'Cluster %d does not satisfy reclustering condition!\n',Cluster_Index)
                Cluster_re(Cluster_Index).Inconsistency = Cluster_monitoring_re(Cluster_Index).Inconsistency;
                Cluster_monitoring_re(Cluster_Index).Final = 1;
            end
        else
            disp('Small cluster size: Skipping!')
        end
        %% Update Cluster index
        Cluster_Index = Cluster_Index + 1;

        %Cancels while loop
        if Cluster_Index > length(Cluster_re)
            stop = 1;

            disp('End of cluster: Iteration stopps!')
        end

        %Next step when last figure is closed
%          fig=gcf;
%          waitfor(fig)
         disp(' ')
    end
    
    Recluster{Re,1} = Cluster_re;
    Recluster_monitoring{Re,1} = Cluster_monitoring_re; 
end

disp('Iterative clustering algorithm finished!')

%% Downsize cluster
%Removes empty cells of iterated cluster from "Cluster"-Struct

for L = 1:length(Recluster)
    temp_struct = Recluster{L,1};
    temp_struct = temp_struct(~cellfun(@isempty,{temp_struct.Frame_Index}));
    
    Recluster{L,1} = temp_struct;
end

disp('Empty fields removed!')

%% Transfer to single Cluster struct

%Overview of all clusters
Clustering_Reiterated = struct();
Clustering_Reiterated.Frame_Index = [];
Clustering_Reiterated.STD_corr = [];
Clustering_Reiterated.STD_dist = []; 


i = 0;
%Varies branch
for L = 1:length(Recluster)
    temp_struct = Recluster{L};
    
    %Varies clusters in branch
    for k = 1:length(temp_struct)
        i = i+1;
        tmp_struct = temp_struct(k);
        
        %Attach
        Clustering_Reiterated(i).Frame_Index = tmp_struct.Frame_Index;
    end
end

disp('Reiterated cluster struct created!')

%% How many clusters have bad statistics?

%%%
%Sort out all clusters with insufficient numbers of frames to achieve real
%images with reasonable signal-to-noise. Here only clusters with >15 frames
%%%

close all

%Struct that stores our modes
recluster_wo_noise = struct();
iterator = 1;
frames = 0; %Counts total number of assigned frames

for ii = 1:length(Clustering_Reiterated)
    %Considers only clusters with >15 frames as our final modes
    if length(Clustering_Reiterated(ii).Frame_Index) > 15
        frames = frames + length(Clustering_Reiterated(ii).Frame_Index);
        recluster_wo_noise(iterator).Frame_Index = Clustering_Reiterated(ii).Frame_Index;        
        iterator = iterator + 1;
    end
end

fprintf(1,'You assigned %d from %d frames which is %2.2f %%\n',frames,28800,100*frames/28800);
