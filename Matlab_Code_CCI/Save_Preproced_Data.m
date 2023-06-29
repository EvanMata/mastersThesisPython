%% Calculate_Correlation_Map
    
    %%%
    %Script calculates the correlation between all reference-filtered 
    %difference holograms, which is the input of the clustering algorithm. 
    %   
    %Input Data: 
    % - reference-filtered difference holograms
    % .../Difference_Holograms_Reference_Filtered/Difference_Hologram_Reference_Filtered_XXXXX.bin
    % - reference-filtered topography holograms
    % .../Topography_Holograms_Reference_Filtered/Topography_Hologram_Reference_Filtered_XXX.bin
    % - Log txt for Coherent Correlation imaging
    % .../Log_Data_Coherent_Correlation_Imaging.txt
    %
    %Input Functions: Flatten_Matrix.m, Noise_Mask.m,
    %CCI_Correlation_Function.m
    %
    %Output: 28800x28800 correlation map
    %
    %Code developed and tested with: Matlab Version 2018a, Statistics and
    %Machine Learning Toolbox Version 11.3, Image Processing Toolbox
    %Version 10.2
    %%%

%% Declaration of Path and Files
%Basic Directories (set Current Folder to Zenodo Export Folder)
workingDir = split(pwd,'\');
lstBaseDir = workingDir(1:length(workingDir)-1);
basePathCell = join(lstBaseDir, '\');
basePath = basePathCell{1}
baseFolder.Data     = fullfile(basePath,'Difference_Hologram_Reference_Filtered');                   %Input Folder of raw holograms
baseFolder.Topo     = fullfile(basePath,'Topography_Holograms_Reference_Filtered');   
baseFolder.Log      = fullfile(basePath);                                   %Input Folder of Log
addpath(genpath(pwd));

procHoloFolder = 'Fully_Processed_Holograms\';
masksFolder = 'Hologram_Masks\';

%% Load Reference_Filtered Topo Holos and Create Topo Struct

Topo = struct();                                                       %Stores all relevant information about Topo holograms
Topo(1).Holo_Ref_Filtered = [];                                             %Stores Reference Filtered Holos

%Vary images
for ii = 1:144
    FileName = fullfile(baseFolder.Topo,sprintf('Topography_Hologram_Reference_Filtered_%03d.bin',ii));
    fprintf(1, 'Now reading %s\n', FileName);
    fid = fopen(FileName,'r');
    Topo(ii).Holo_Ref_Filtered = fread(fid,[70,70],'double');
    fclose(fid);
end

%Show
f = figure;
ax = gca;
%imagesc(Topo(1).Holo_Ref_Filtered,'Parent',ax); axis image; colorbar

%% Load Log and Creates Data Struct
Log = readtable(fullfile(baseFolder.Log,'Log_Data_Coherent_Correlation_Imaging.txt'));
Data = table2struct(Log);    %Stores all relevant information about holograms

Data(1).Holo_Ref_Filtered = []; %Stores Holos

%Select relevant frames for correlation calculation
%Frame_Vec = [3500:4000]; ORIGINAL, COMMENTED OUT FOR MINI SAMPLE
Frame_Vec = [1:2000]; %PROCESSES ONLY THESE HOLOS? What?? WHY?

%Load Reference-filtered Diff holos
for ii = Frame_Vec
    FileName = fullfile(baseFolder.Data,sprintf('Difference_Hologram_Reference_Filtered_%05d.bin',ii));
    if 0 == mod(ii,100)
        fprintf(1, 'Now reading %s\n', FileName);
    end
    fid = fopen(FileName,'r');
    Data(ii).Holo_Ref_Filtered = fread(fid,[70,70],'double');
    fclose(fid);
end

%Find data with holograms
len = find(~cellfun(@isempty,{Data.Holo_Ref_Filtered}));

%Show
%imagesc(Data(len(1)).Holo_Ref_Filtered,'Parent',ax); axis image; colorbar

%% Create Masks
close all

    %%%
    %Creates Masks to select only specific regions of the scattering images
    %
    %These masks are defined:
    % - Beamstop: Masks the central part of the Reference_Filtered
    % difference images where the beamstop is located. Necessary because
    % the beamstop area is not zero any more due to the reference
    % filtering.
    % - Dead-Pixel: Masks a dead tile of the detector.
    % - q-Filter: Excludes data exceeding a certain q-radius. Areas where 
    % the noise dominates over the signal.
    % - Flatten-Mask: Due to reference-filtering of topo mask, some
    % pixel are negativ, which is an artifact, or zero, leading to inf 
    % pixel values. Both types are excluded from analysis.
    % - Noise mask: Further excluding of areas where the noise dominates 
    % over the signal. They are defined based on the individual pixel
    % intensity by inverse bandpass filtering. Bandpass is determined 
    % individually for each images. Therefore, the noise level is estimated
    % by calculating the mean noise level and standard deviation of at high
    % q-radius.
    % Comment: Use either one or both options for noise filtering to reach
    % reasonable SNR in correlation analysis.
    %%%

%Parameter for masks
Dia_Beamstop = 8.5;     %Diameter of beamstop mask
Dia_q = 40;             %Diameter of q-Filter; True when q < q-Filter
Dia_STD = Dia_q;           %Diameter of q-filter to estimate Noise; 
                         %True when q > q-Filter
STD_Limit = 4;          %Intervall of noise mask filtering; 
                         %True when Intens is not in [Mean+/-STD_Limit*STD]                         
                                        
%Create global masks
%Define Zeros Mask for Filtering of Beamstop
sz = size(Data(len(1)).Holo_Ref_Filtered);

[lower, upper] = scaleContrast(Data(len(1)).Holo_Ref_Filtered,[10 90]);
figure; ax = gca;
imshow(Data(len(1)).Holo_Ref_Filtered,[lower upper],'Parent',ax);
Zeros = createMask(imellipse(ax, [(sz(2)-Dia_Beamstop+1)/2 (sz(1)-Dia_Beamstop)/2 Dia_Beamstop Dia_Beamstop]));

%Define Additional Mask for Deadpixel-Rows of detector
Zeros(65:69,1:40) = 1;

%Define Mask to get only Correlation areas within a certain absolute value of scattering vector q 
Zeros = or(Zeros,not(createMask(imellipse(gca, [(sz(2)-Dia_q)/2 (sz(1)-Dia_q)/2 Dia_q Dia_q]))));
Zeros = not(Zeros);

%Define Mask to get only Correlation areas beyond certain absolute value of scattering vector q 
Mask_STD = not(createMask(imellipse(ax, [(sz(2)-Dia_q)/2 (sz(1)-Dia_q)/2 Dia_q Dia_q])));


%Calc Flatten-Mask for topo images and apply Zeros-mask
Topo(1).Flatten_Matrix = [];  %Add Field
Topo(1).Flatten_Mask = [];  %Add Field

for ii = 1:length(Topo)
    %fprintf(1, 'Processing Topo-Holo: %d (%d)\n', ii,length(Topo));
    
    %Apply Zeros mask (beamstop, dead-pixel tile, ...)
    Topo(ii).Holo_Ref_Filtered = Topo(ii).Holo_Ref_Filtered.*Zeros;   %Apply mask
    
    %Calc Flattening topo matrix
    [Topo(ii).Flatten_Matrix,Topo(ii).Flatten_Mask] = Flatten_Matrix(Topo(ii).Holo_Ref_Filtered);
end

%Show Topo image with Masked areas
figure; imagesc(Topo(2).Holo_Ref_Filtered); axis image; 
title('Masked beamstop and dead tiles reference filtered topo holos')

%% Calc Flattened-Difference Holo for diff images
Holo_Flattened = zeros(sz(1),sz(2),length(len));
Noise_Masks = zeros(sz(1),sz(2),length(len));

for ii = len
    %fprintf(1, 'Processing Diff-Holo: %d [%d,%d]\n', ii,len(1),len(end));
        
    %Get Topo Index Nr that correspond to image frame i
    Topo_Index = Data(ii).Topography_Nr_; 
    
    %Flatten reference-filterend diff holos
    try
        Data(ii).Holo_Flattened = Data(ii).Holo_Ref_Filtered./sqrt(Topo(Topo_Index).Flatten_Matrix);
        Data(ii).Holo_Flattened = Data(ii).Holo_Flattened.*Topo(Topo_Index).Flatten_Mask;    %Note: Flatten_Mask also contains Zeros
    
        %Calc Noise mask
        Data(ii).Noise_Mask = Noise_Mask(Data(ii).Holo_Ref_Filtered,Mask_STD,STD_Limit);
        Data(ii).Noise_Mask = Data(ii).Noise_Mask.*Topo(Topo_Index).Flatten_Mask;
        
        %Create also Vector for faster processing
        Holo_Flattened(:,:,ii) = Data(ii).Holo_Flattened;
        Noise_Masks(:,:,ii) = Data(ii).Noise_Mask;
    catch 
        fprintf(1, 'Failed Processing Diff-Holo: %d [%d,%d]\n', ii,len(1),len(end));
        fprintf(1, '%i \n', Topo_Index);
    end
end

%Show Image with Masked areas
figure; imagesc(Data(len(1)).Holo_Flattened.*Data(ii).Noise_Mask); axis image; 
title('Final flattened holograms')
disp('Holograms flattened and masks created!')

%% Calculate complete 2-time-correlation matrix
close all

    %%%
    %Calculates the two-time correlation function of all flattened holograms
    %frames defined in Frame Vec. 
    %Note that for each combination of the frames the Noise masks define 
    %integration area in q-space to analyze. Therefore, the normalization
    %factor which norm the auto-correlation to the value 1, need to be
    %calculated for each combination of frames
    %If q-filter for noise filtering is used, matrix sizes can be reduced
    %significantly and script is much faster
    %%%
 
%ROI = Region of Interest
%Define ROI to reduce data matrix size because q-Filter is used    
ROI = [floor(size(Data(len(1)).Holo_Ref_Filtered,1)/2 - (Dia_q/2+1)):ceil(size(Data(len(1)).Holo_Ref_Filtered,1)/2 + (Dia_q/2+1))];  %Symmetric ROI
fprintf(1, 'ROI\n');
fprintf(1, ROI)

%%%
%TEST IF ERROR OCCURS BECAUSE OF OFF-INDEXING 
%for ii = 1:(length(len)-1)
%    index_1 = len(ii);
%    Nr = ii/(length(len)-1);
%    Status = 100*(2*Nr - Nr^2); %Percentage of calculated pairs
%    if 0 == mod(ii,100)
%        fprintf(1, 'Running holo: %d [%d,%d]; Finished: %0.3f %% \n', index_1,len(1),len(end),Status);
%    end
%    
%    try
%        holoPreprocessed = Data(index_1).Holo_Flattened(ROI,ROI);
%        holoMask = Data(index_1).Noise_Mask(ROI,ROI);
%        fileHoloPath = fullfile(basePath, procHoloFolder, sprintf('preprocessed_Holo_%05d.bin', ii));
%        fileIDHolo = fopen(fileHoloPath,'wb');
%        fileHoloMaskPath = fullfile(basePath, masksFolder, sprintf('holo_mask_%05d.bin', ii));
%        fileIDHoloMask = fopen(fileHoloMaskPath,'wb');
%        fwrite(fileIDHolo,holoPreprocessed,'float64');
%        fwrite(fileIDHoloMask,holoMask,'float64');
%        fclose(fileIDHolo);
%        fclose(fileIDHoloMask);
%
%    catch
%        fprintf(1, '%d is a bad index \n', index_1);
%    end
%end
%%%
disp('Script finished!')
