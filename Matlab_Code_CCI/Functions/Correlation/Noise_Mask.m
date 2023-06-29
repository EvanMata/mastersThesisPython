function [Statistics] = Noise_Mask(Diff_Image,Mask_STD,STD_Limit)

    %%%
    %Function calculates a binary mask where pixel are true when their
    %intensity is significantly above the noise level. Here, noise level is
    %given by an bandpass calculated from the average and standard
    %deviation. Masks is true when: 
    % Limit_low < Mean-STD_Limit*STD, Limit_high > Mean+STD_Limit*STD,
    %
    %Input: Diff_Image: Difference hologram 
    %       Mask_STD: binary masks that defines area to estimate noise
    %       STD_Limit: Defines Limits of bandpass interval
    %
    %Output: flat_holo: flattened hologram
    %%%

%Get Area of Mask_STD for averaging
temp_Area = sum(sum(Mask_STD));
%Calc Average
temp_Avg = sum(sum(Diff_Image.*Mask_STD))/temp_Area;

%Calc standard deviation
STD = sqrt(1/temp_Area*sum(sum((Diff_Image.*Mask_STD-temp_Avg*ones(size(Diff_Image))).^2))); %Noise STD of Diff_Image

%inverse intensity bandpass-filtering to get statistics mask
Statistics = and(Diff_Image < temp_Avg+STD_Limit*STD, Diff_Image > temp_Avg-STD_Limit*STD); %Noise interval, Signal is not Element of Interval
Statistics = not(Statistics);