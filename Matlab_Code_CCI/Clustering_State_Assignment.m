%%%
    %Script performs grouping of the domain modes into the magnetic domain
    %states which are spatially and temporally resolved. 
    % 
    %Classification is based on the sensitivity threshold that we derived 
    %from an analysis of the missclassification probability. We find that 
    %for modes that exhibit up to 93.8% similarity the missclassification 
    %probability of single frames is below 1%.  
    %We define this as the sensitivity threshold in our analysis. Internal 
    %domain modes with higher similarity were grouped allowing their 
    %temporal discrimination with an acceptable misclassification rate. 
    %These groups of modes are referred to as spatially and temporally 
    %resolved magnetic "states". In practice, we successively combined the 
    %nearest modes until the distance between any pair of modes of two 
    %dissimilar states exceeded the 100 % – 93.8 % = 6.2 % similarity 
    %distance defined by the sensitivity threshold.
    %   
    %Input Data: 
    % - Frame-to-mode assignment, Mode Reconstruction and Binary internal
    % modes
    % .../Cluster_Analysis/Reco_Mask_Final.mat
    %
    %Input Functions: Matlab "Statistics and Machine Learning Toolbox", 
    %"Image Processing Toolbox", Cluster_Hierarchical.m
    %
    %Output: State assignment
    %
    %Code developed and tested with: Matlab Version 2018a, Statistics and
    %Machine Learning Toolbox Version 11.3, Image Processing Toolbox
    %Version 10.2
    %%%

%% Declaration of Path and Files
%Basic Directories (set Current Folder to Zenodo Export Folder)
baseFolder.Data     = fullfile(pwd,'Clustering_Analysis');                  %Input Folder of 28800x28800 correlation map
baseFolder.Log      = fullfile(pwd);                                        %Input Folder of Log

%% Load Log
%Stores all relevant information
Log = readtable(fullfile(baseFolder.Log,'Log_Data_Coherent_Correlation_Imaging.txt'));

%% Load Clusters

    %%%
    %Matfile contains all relevant data of the particular modes, which are
    % - Complex holograms for positive and negative helicity
    % - Real part, imaginary part, absolute and phase of reconstructions
    % - low-pass filtered averaged difference holograms used to determine
    % the binary domain mask. Radi that defines max q in px.
    % - Real part, imaginary part, absolute and phase of reconstructions of
    % the low-pass filtered difference holograms
    % - The binary Domain masks
    %%%

ImageName   = sprintf('Reco_Mask_Final.mat');
OutPath     = fullfile(baseFolder.Data,ImageName);                          %Combine output name and folder to path
fprintf(1,'Now loading %s\n', ImageName);                                    %Display output path in command window 
load(OutPath, 'Cluster_Reco','Submode');
disp('Loading *.mat Done!')

%Additional binary domain configuration 73 was manually created (we
%attribute this to the fact that state 32 is the last state in our time 
%series and insufficient data was available to automatically decompose it)

Cluster_Reco{73} = Cluster_Reco{5};
Cluster_Reco{73}.Domain_Mask = Submode.Domain_Mask;

%% Separation of masks in basic configurations
close all

%%%
%To extract the positions of domain walls from sometimes noisy or 
%inhomogeneously illuminated real-space images, we discretized the 72 
%domain configurations identified in the cluster analysis.
%
%Some of the original real-space reconstructions of the 72 internal modes 
%show evidence of mixed-state superposition. Such mixed states inevitably 
%emerge if the domain state changes during the acquisition of a single 
%frame (i.e., if the dynamics is faster than our minimum temporal 
%resolution). Hence, simply binarizing a domain image as the one would 
%constitute a loss of information. However, we find that the whole set of 
%binarized internal modes forms a representative basis, based on which the 
%original gray-scale images can be decomposed. In practice, we evaluated 
%the similarity between the domain images and all binary internal modes 
%by the real-space correlation. In a next step, we applied a multilinear 
%regression to determine the individual weighting of the most similar 
%internal modes from the original image.
%%%

%% Evaluation of similarity between real-space images and binary internal modes

%Mask field of view
mask = createCirclesMask(size(Cluster_Reco{1}.Real_Reco),[size(Cluster_Reco{1}.Real_Reco,1)+2,size(Cluster_Reco{1}.Real_Reco,2)+1]/2,90);

%Stores Scalar product between real-space images and binary internal modes
Projection_Coeff = zeros(length(Cluster_Reco));

%Stores Normalized Reco and Domain Masks
Normed_Reco = zeros(size(Cluster_Reco{1}.Real_Reco,1)*size(Cluster_Reco{1}.Real_Reco,2),length(Cluster_Reco));
Normed_Mask = zeros(size(Cluster_Reco{1}.Real_Reco,1)*size(Cluster_Reco{1}.Real_Reco,2),length(Cluster_Reco));

%Get Projection of Reco onto Domain masks by Scalar product (non-symmetric due to different normalization!)
%Vary real-space images
for L = 1:length(Cluster_Reco)
    %Norm Reco
    temp_Reco = Cluster_Reco{L}.Real_Reco.*mask;
    temp_Reco = temp_Reco/sqrt(dot(temp_Reco(:),temp_Reco(:)));
    
    Normed_Reco(:,L) = temp_Reco(:);
    
    %Vary internal modes
    for k = 1:length(Cluster_Reco)
        %Norm Mask
        temp_Mask = Cluster_Reco{k}.Domain_Mask.*mask;
        temp_Mask = temp_Mask/sqrt(dot(temp_Mask(:),temp_Mask(:)));

        %Calc Scalar Product (Projection)
        Projection_Coeff(L,k) = dot(temp_Reco(:),temp_Mask(:));
        
        %Will be used in linear regression. 1d representation of the mask
        Normed_Mask(:,k) = temp_Mask(:);
    end
end 

figure; imagesc(Projection_Coeff); axis image; axis xy; title('Projection Coefficient')
colorbar; xlabel('Internal Mode'); ylabel('Real-space image')

disp('Normed!')

%% Linear Regression of internal modes
close all

%Only decomposition of those modes
Regress_Set = [3,4,5,6,8,9,11,12,14,16,20,21,23,24,25,29,32,36,37,38,...
               39,40,42,43,44,49,50,53,54,58,60,61,64,65,68]; 
          
%Vary all modes
for L = 1:length(Cluster_Reco) 
    Reco_Nr = L;
    
    %General threshold. Considers only internal modes with projection
    %coefficient > threshold for decomposition.
    Thres = 0.91;
    [row] = find(Projection_Coeff(:,Reco_Nr)> Thres);
    
    %Manually adapted thresholds to optimize decomposition
     if L == 3
        Thres = 0.91;
        [row] = find(Projection_Coeff(:,Reco_Nr)> Thres);
     elseif L == 5
        Thres = 0.94;
        [row] = find(Projection_Coeff(:,Reco_Nr)> Thres);
     elseif L == 8
        Thres = 0.92;
        [row] = find(Projection_Coeff(:,Reco_Nr)> Thres);
     elseif L == 9 
         Thres = 0.92;
         [row] = find(Projection_Coeff(:,Reco_Nr)> Thres);
     elseif L == 12 
         Thres = 0.92;
         [row] = find(Projection_Coeff(:,Reco_Nr)> Thres);
     elseif L == 14 
         Thres = 0.92;
         [row] = find(Projection_Coeff(:,Reco_Nr)> Thres);
     elseif L == 21 
         Thres = 0.91;
         [row] = find(Projection_Coeff(:,Reco_Nr)> Thres);
     elseif L == 22 
         Thres = 0.92;
         [row] = find(Projection_Coeff(:,Reco_Nr)> Thres);
     elseif L == 23 
         Thres = 0.90;
         [row] = find(Projection_Coeff(:,Reco_Nr)> Thres);
     elseif L == 24 
         Thres = 0.88;
         [row] = find(Projection_Coeff(:,Reco_Nr)> Thres);   
     elseif L == 25 
         Thres = 0.8934;
         [row] = find(Projection_Coeff(:,Reco_Nr)> Thres);
     elseif L == 29 
         Thres = 0.915;
         [row] = find(Projection_Coeff(:,Reco_Nr)> Thres);  
     elseif L == 32 
         Thres = 0.915;
         [row] = find(Projection_Coeff(:,Reco_Nr)> Thres);  
     elseif L == 36
         Thres = 0.895;
         [row] = find(Projection_Coeff(:,Reco_Nr)> Thres);
     elseif L == 37
         Thres = 0.92;
         [row] = find(Projection_Coeff(:,Reco_Nr)> Thres);
     elseif L == 39
         Thres = 0.89;
         [row] = find(Projection_Coeff(:,Reco_Nr)> Thres);
     elseif L == 39
         Thres = 0.89;
         [row] = find(Projection_Coeff(:,Reco_Nr)> Thres);
     elseif L == 40
         Thres = 0.92;
         [row] = find(Projection_Coeff(:,Reco_Nr)> Thres);
     elseif L == 43
         Thres = 0.90;
         [row] = find(Projection_Coeff(:,Reco_Nr)> Thres);
     elseif L == 49
         Thres = 0.9283;
         [row] = find(Projection_Coeff(:,Reco_Nr)> Thres);
     elseif L == 50
         Thres = 0.90;
         [row] = find(Projection_Coeff(:,Reco_Nr)> Thres);
     elseif L == 53
         Thres = 0.91;
         [row] = find(Projection_Coeff(:,Reco_Nr)> Thres);
     elseif L == 54
         Thres = 0.898;
         [row] = find(Projection_Coeff(:,Reco_Nr)> Thres);
     elseif L == 58
         Thres = 0.915;
         [row] = find(Projection_Coeff(:,Reco_Nr)> Thres);
     elseif L == 60
         Thres = 0.89;
         [row] = find(Projection_Coeff(:,Reco_Nr)> Thres);
     elseif L == 65
         Thres = 0.918;
         [row] = find(Projection_Coeff(:,Reco_Nr)> Thres);
     elseif L == 68
         Thres = 0.88;
         [row] = find(Projection_Coeff(:,Reco_Nr)> Thres);
     end
    
    % If there does not exist any internal mode with a projection 
    % coefficient exceeding the threshold, the one with the highest coeff
    % will be considered
    if isempty(row)
        [~,row] = max(Projection_Coeff(:,Reco_Nr));
    end
    
    %Indexing of Normed_mask
    temp_Vec = row;
    
    %Regression method, Uses 1d representation of mask
    Ref = Normed_Mask(:,temp_Vec);
    b = robustfit(Ref,Normed_Reco(:,Reco_Nr));
   
    %Norm (b)
    b = b/sum(b(2:end));
    
    Mask_Regress = zeros(size(Cluster_Reco{1}.Real_Reco));
    
    %Decomposition with coefficients determined by regression
    for k = 0:length(temp_Vec)
        if k == 0
            Mask_Regress = Mask_Regress + b(k+1)*zeros(size(Cluster_Reco{1}.Real_Reco)).*mask;
        else
            Index = temp_Vec(k);
            Mask_Regress = Mask_Regress + b(k+1)*Cluster_Reco{Index}.Domain_Mask;
        end
    end
    
    %Renormalize
    Mask_Regress(Mask_Regress > 1) = 1;
    Mask_Regress(Mask_Regress < -1) = -1;
    
    %Save only if mode is in Regress_Set (those that need to be decomposed)
    if ismember(Reco_Nr,Regress_Set) == 1
        Cluster_Reco{L}.Domain_Mask_Regress = Mask_Regress;
    elseif ismember(Reco_Nr,Regress_Set) == 0
        Cluster_Reco{L}.Domain_Mask_Regress = Cluster_Reco{Reco_Nr}.Domain_Mask;
    elseif L == 5
        Cluster_Reco{73}.Domain_Mask_Regress = Mask_Regress;
    end 
end

%% Calculate real space pair-correlations of masks
close all

    %%%
    %Cluster pair correlation between cluster A and B is calculated by, 
    %first, reading the cluster's frame assignments, second, extracting the
    %pair correlation between all frames of cluster A and all frames of
    %cluster B (each combination/permutation of the index frames) and,
    %third, averaging this whole set of pair correlations.
    %%%

%Define Correlation Matrix. Autocorrelation is always 1
Cluster_Correlation_Real_mask_regress = eye(length(Cluster_Reco)-1);

%Calculate real-space correlation
for L = 1:length(Cluster_Reco)-1
    Ref_Image = Cluster_Reco{L}.Domain_Mask_Regress;
    Norm_1 = sum(sum(Ref_Image.*Ref_Image));
    for k = 1:length(Cluster_Reco)-1
        Corr_Image = Cluster_Reco{k}.Domain_Mask_Regress;
        Norm_2 = sum(sum(Corr_Image.*Corr_Image));
        
        Cluster_Correlation_Real_mask_regress(L,k) = sum(sum(Ref_Image.*Corr_Image))/sqrt(Norm_1*Norm_2);
    end    
end

disp('Real Space Correlation Calculated!')

%Plot
figure; imagesc(Cluster_Correlation_Real_mask_regress); axis image; axis xy; colormap parula; axis on; colorbar
title('Real space correlation masks regress'); xlabel('Mode'); ylabel('Mode')


%% Load 'Cluster'-struct containing the frame to cluster assignment

%%%
%Recluster_wo_noise contains the different modes to frame assignments
%%%

MatFileName = fullfile(baseFolder.Data,'Cluster_hierachical_reit_wo_noise.mat');
fprintf(1,'Now loading %s\n', MatFileName);
load(MatFileName,'recluster_wo_noise');
disp('Loading is done!')

%% Assign real space data to modes

%Vary modes
for L = 1:72
    recluster_wo_noise(L).Frame_Index = sort(recluster_wo_noise(L).Frame_Index);
    recluster_wo_noise(L).Mode = L;
    recluster_wo_noise(L).Real_Reco = Cluster_Reco{L}.Real_Reco;
    recluster_wo_noise(L).Domain_Mask_Regress = Cluster_Reco{L}.Domain_Mask_Regress;
    recluster_wo_noise(L).Domain_Mask = Cluster_Reco{L}.Domain_Mask;
    recluster_wo_noise(L).Phase_Reco_Cropped = Cluster_Reco{L}.Phase_Reco_Cropped;
end
disp('Real space data assigned!')

%% Further clustering to group the clusters further into domain states
close all

    %%%
    %Similar to the second iteration of the mode clustering we will focus
    %only on subsets of the most similar modes as only those contain
    %potentially misclassified frames. 
    %
    %The subsets are determined from the hierarchy given by the correlation 
    %of the decomposed real-space images between all modes. These subsets 
    %are the different "branches".
    %
    %The global 'cutoff' value defines the hierarchical distance cutting the
    %dendrogram at the given height to separate dendrogram branches.
    %The dendrogram is separated into branches, which will be processed 
    %further individually.
    %%%
     
%Distance to cut the dendrogram and separate branches    
Cutoff = 0.05; %correlation

%Calculate linkage of cluster pair-correlations and show dendrogram
Linkage   = linkage(Cluster_Correlation_Real_mask_regress,'average','correlation'); 
f = figure; dendrogram(Linkage,100); title('Hierarchy between modes'); xlabel('Modes')

%Visualize cutoff value in dendrogram
line([0,150],[Cutoff,Cutoff])

%Separates clustering branches
[Cluster_Vec,Cluster_Matrix] = Cluster_Hierarchical(Linkage,'cutoff',Cutoff);

%% Create new Branch cell storing each branche's clusters

%Setup cell
Cluster_Branch = {};

%Vary Branches 
for L = 1:size(Cluster_Matrix,1)
    %Find modes assigned to a given branch
    Index = find(Cluster_Matrix(L,:) ~= 0);
    
    %Varies each branches modes
    for k = 1:length(Index)
        %Get mode k's cluster index in 'recluster_wo_noise'-struct
        ii = Index(k);
        
        %Assign mode to Cluster_Branch
        Cluster_Branch{L,k} = recluster_wo_noise(ii);
        
        %Save mode index
        Cluster_Branch{L,k}.Cluster_Index = ii;
    end
end

disp('Branches processed!')

%% Extract the branche's cluster pair correlations

    %%%
    %Extracts the cluster pair-correlation between all modes assigned to
    %coincident branches from the mode correlation map
    %%%
    
%Setup cell
Branch_Corr = {};
    
%Vary branches    
for L = 1:size(Cluster_Branch,1)
    %Prelocate branche's pair correlation map
    Branch_Corr{L}.Corr = zeros(size([Cluster_Branch{L,:}],2));
    Branch_Corr{L}.Cluster_Index = zeros(size([Cluster_Branch{L,:}],2),1);
    
    %Vary first ,mode of branch
    for k = 1:size([Cluster_Branch{L,:}],2)
        %Get mode Index of first mode
        Index_1 = Cluster_Branch{L,k}.Cluster_Index;
        
        %Stores analyzed mode indices an their order
        Branch_Corr{L}.Cluster_Index(k) = Index_1;
        
        %Vary second mode of branch
        for m = 1:size([Cluster_Branch{L,:}],2)
            %Get mode Index of second mode
            Index_2 = Cluster_Branch{L,m}.Cluster_Index;
            
            %Extract branche's cluster pair correlations from mode pair
            %correlation map
            Branch_Corr{L}.Corr(k,m) = Cluster_Correlation_Real_mask_regress(Index_1,Index_2);
        end
    end
end

disp('Branches mode correlation extracted!')

%% Separate based on metric

%Create state assignment based on sensitivity threshold here given as the
%cutoff value. 
%We successively combine the nearest modes until the distance between any 
%pair of modes of two dissimilar states exceeded the 100 % – 93.8 % = 6.2 % 
%Combining the modes is achieved by a combination of hierarchical
%single linkage clustering (combining clusters with min distance, i.e., 
% highest pair correlation) and the chebychev metric (maximum distance 
% between cluster elements). 


Cutoff = 0.938; %Mask regress 32 States

%Setup cell
States = {};

%Vary branches
iterator = 0;
for L = 1:length(Branch_Corr)
    %distance metric
    temp_matrix = 1 - Branch_Corr{L}.Corr;
    
    %Only if branch contains more than 1 element
    if size([Cluster_Branch{L,:}],2) > 1
        %Cut dendrogram when maximum distance between all cluster elements
        %is > 100%-Cutoff
        Linkage   = linkage(temp_matrix,'single','chebychev');        
        [assignment,Cluster_Matrix] = Cluster_Hierarchical(Linkage,'cutoff',1-Cutoff); 
    else
        assignment  = 1;
        Cluster_Matrix = 1;
    end
    
    %Prelocate
    for k = 1:size(Cluster_Matrix,1)
        States{iterator + k,1} = [];
    end
    
    %Varies modes in branches
    for k = 1:size(Cluster_Matrix,2)
        %State assignment of mode
        state = assignment(k);
        
        %Get inital mode
        mode = Cluster_Branch{L,k,1}.Mode;
        
        %Save in struct
        States{iterator + state,1} = horzcat(States{iterator + state,1},Cluster_Branch{L,k,1});
    end
    
    iterator = iterator + size(Cluster_Matrix,1);
end

disp('States created!')

%% Add Timestamp to new cluster, create Cluster Assignment and reorder
close all

%Convert everything to a single cell
%Setup
State_Assignment = zeros(length(Log.Timestamp_),1);
Time_Vector = Log.Timestamp_;

%Vary state
for state = 1:size(States,1)
    temp_Vec = 0;
    
    %Vary modes
    for mode = 1:length(States{state})
        %Sort frame order
        Frame_Vec = sort(States{state}(mode).Frame_Index);

        %Assign state to frames
        State_Assignment(Frame_Vec) = state;

        %Add Timestamp
        States{state,1}(mode).Timestamp = Time_Vector(Frame_Vec);
    end    
end    

%Plot
figure; plot(State_Assignment(State_Assignment~=0)); xlabel('Frame Index');
ylabel('States (unordered)')

fprintf('You assigned %1.0f from 28800 frames (%1.1f %%)\r\n',length(State_Assignment(State_Assignment~=0)),100*length(State_Assignment(State_Assignment~=0))/28800)

%Order with respect to temporal evolution in measurement
Final_States = {};
order = unique(State_Assignment(State_Assignment~=0),'stable');
for state = 1:length(order)
    Final_States{state,1} = States{order(state)};
end

%Final state assignment
Final_State_Assignment = zeros(length(Log.Timestamp_),1);

%Create final assignment
%Vary state
for state = 1:size(Final_States,1)
    temp_Vec = 0;
    
    %Vary modes
    for mode = 1:length(Final_States{state})
        Frame_Vec = Final_States{state}(mode).Frame_Index;
        Final_State_Assignment(Frame_Vec) = state;
    end    
end 

disp('State Assignment created!');

%Plot Temporal sequence of states
f = figure; plot(Time_Vector(Final_State_Assignment ~=0),Final_State_Assignment(Final_State_Assignment ~=0)); 
xlabel('Time (s)'); ylabel('State')

%% Export all real-space images of the states
close all

k = 1;
mask_scale = createCirclesMask(size(Cluster_Reco{k}.Real_Reco),[size(Cluster_Reco{k}.Real_Reco,1)+2,size(Cluster_Reco{k}.Real_Reco,2)+1]/2,75); 

folder = fullfile(baseFolder.Data,'Final_States_Ordered');

if ~exist(folder)
    mkdir(folder)
else
    rmdir(folder,'s')
    mkdir(folder)
end

%Vary state
for state = 1:size(Final_States,1)
    
    %Create folder
    nfolder = fullfile(folder,sprintf('State_%02.0f',state));
    
    if ~exist(nfolder)
        mkdir(nfolder)
    end
    
    %Vary modes
    for mode = 1:length(Final_States{state})
        temp_Reco = Final_States{state}(mode).Real_Reco.*mask;
        
        %Process real reco
        temp_Reco_scale = temp_Reco.*mask_scale + 10*(1-mask_scale);

        [lower,upper] = scaleContrast(temp_Reco_scale(temp_Reco_scale~=10),[0.1 99.9]);
        temp_Reco = 255*((temp_Reco - lower)/(upper-lower));
        temp_Reco = uint8(temp_Reco.*mask+255/2*(1-mask));        
        
        %Save Images
        Image_Name   = ['Reconstruction_State_' char(string(state)) '_Mode_' char(string(mode)) '_inital_mode_' char(string(Final_States{state}(mode).Mode))  '.png'];
        OutPath   = fullfile(nfolder,Image_Name);
        fprintf(1,'Now saving %s\n', Image_Name);
        imwrite(temp_Reco,OutPath);
        
        temp_reco = uint8(255/2*(Final_States{state}(mode).Domain_Mask+1));
        
        %Save Images
        Image_Name   = ['Mask_Reconstruction_State_' char(string(state)) '_Mode_' char(string(mode)) '_inital_mode_' char(string(Final_States{state}(mode).Mode))  '.png'];
        OutPath   = fullfile(nfolder,Image_Name);
        fprintf(1,'Now saving %s\n', Image_Name);
        imwrite(temp_reco,OutPath);

        temp_reco = uint8(255/2*(Final_States{state}(mode).Domain_Mask_Regress+1));
        
        %Save Images
        Image_Name   = ['Regressio_Mask_Reconstruction_State_' char(string(state)) '_Mode_' char(string(mode)) '_inital_mode_' char(string(Final_States{state}(mode).Mode))  '.png'];
        OutPath   = fullfile(nfolder,Image_Name);
        fprintf(1,'Now saving %s\n', Image_Name);
        imwrite(temp_reco,OutPath);
    end    
end

%Mode
temp_reco = uint8(255/2*(Submode.Domain_Mask+1));
        
%Save Images
Image_Name   = ['Mask_Reconstruction_State_' char(string(state)) '_Mode_' char(string(2)) '_inital_mode_' char(string(73))  '.png'];
OutPath   = fullfile(nfolder,Image_Name);
fprintf(1,'Now saving %s\n', Image_Name);
imwrite(temp_reco,OutPath);