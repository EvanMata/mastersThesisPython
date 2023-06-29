function [monitoring] = cluster_monitoring(corr_array,dist_metric,linkage,plot)

    %%%
    %Function some feedback parameters and optional plots
    %
    %Input: corr_array: Pair correlation map
    %       dist_metric: distance metric
    %       linkage: Agglomerative hierarchical cluster tree
    %       plot: if plot == 1 histograms of corr_array and distm_metric
    %       will be output
    %
    %Input functions: Matlab 'Statistics and Machine Learning Toolbox'
    %
    %Output: monitoring: struct that contains feedback values, i.e., 
    %standard deviation of corr_array and dist_metric, inconsistency 
    %coefficient of each node in hierarchy
    %%%

%% Setup

%Convert to 1d
corr_vec = corr_array(:);
dist_vec = dist_metric(:);

%% Histograms

if plot == 1
    k = figure('Position',[50 50 1600 700],'Name',sprintf('Cluster-Monitoring histograms'));
    
    ax(1) = subplot(2,1,1);
    histogram(corr_vec)
    title(sprintf('Correlation map'));
    xlabel('Correlation value'); 
        
    ax(2) = subplot(2,1,2);
    histogram(dist_vec)
    title(sprintf('Distance metric'));
    xlabel('distance value'); 
end

%% Feedback values
%Standard deviation
monitoring.STD_corr = std(corr_vec);
monitoring.STD_dist = std(dist_vec);

%inconsistency
Y = inconsistent(linkage,5);
monitoring.Inconsistency = Y(end,:);

%% Print

monitoring