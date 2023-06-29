%% Calculate_Topography_Holograms
    
    %%%
    %Script calculates the sum of holograms recorded with positive and
    %negative helicity. Only holograms without dynamics during the
    %acquisition were used for the topography data. Therefore, only sets
    %where the magnetic reconstruction from the difference of positive and
    %negative helicity holograms showed full contrast across the entire
    %field of view. 
    
    %Input Data: 
    % - raw holograms of positive and negative helicity
    % .../Raw_Holo/Raw_Holograms_XXXX.bin
    % - Log txt for Coherent Correlation imaging
    % .../Log_Data_Coherent_Correlation_Imaging.txt
    %
    %Input Functions: Create_Topo.m or Create_Topo_Stack.m
    %
    %Output: Topography_Holograms [960,972,144]    
    %
    %Code developed and tested with: Matlab Version 2018a, Statistics and
    %Machine Learning Toolbox Version 11.3, Image Processing Toolbox
    %Version 10.2
    %%%

%% Declaration of Path and Files
%Basic Directories
workingDir = split(pwd,'\')
lstBaseDir = workingDir(1:length(workingDir)-1)
basePathCell = join(lstBaseDir, '\')
basePath = basePathCell{1}
baseFolder.Data     = fullfile(basePath,'Raw_Holograms');                   %Input Folder of raw holograms
baseFolder.Log      = fullfile(basePath);                                   %Input Folder of Log
addpath(genpath(pwd));

%% Load Log
Log = readtable(fullfile(baseFolder.Log,'Log_Data_Coherent_Correlation_Imaging.txt'));

%% Load Sum Array

    %We recorded 288 hologram stacks with alternating x-ray helicity, each 
    %consisting of 100 image frames.
    %Averages all frames that correspond to coincident image stacks. Then
    %calculates the sum of consecutive image stack with alternating x-ray
    %helicity

%Arrays that save averaged holograms of positive and negative helicity
Pos = zeros(960,972,size(Log,1)/200);
Neg = zeros(960,972,size(Log,1)/200);

%Temporary Array
temp_Array = zeros(960,972,100);
    
%Vary image stack
for i = 1:size(Log,1)/100
    Frame = 100*(i-1) + 1;                                                  %First Image of each Stack
    Helicity = Log.Helicitiy_(Frame);
    
    %Vary images in stack
    for j = 1:100
        Frame = 100*(i-1) + j;

        %Load hologram frames
        FileName = fullfile(baseFolder.Data,sprintf('Raw_Hologram_%05d.bin',Frame));
        fprintf(1, 'Now reading %s\n', FileName);
        fid = fopen(FileName,'r');
        temp_Array(:,:,j) = fread(fid,[960,972],'double');
        fclose(fid);
    end
    
    %Separate Stacks of Positive and negative Helicity
    if Helicity == 1                                                        %If Helicity is positive
        Pos(:,:,ceil(i/2)) = sum(temp_Array,3);
    elseif Helicity == -1                                                   %If Helicity is negative
        Neg(:,:,ceil(i/2)) = sum(temp_Array,3);
    end
end

Sum = Pos + Neg;
disp('Sum Array calculated!')

%% Define "Topo-Images" for Calculation of Difference Hologram

    %%%
    %Creates array [960,972,144] of topography holograms based on the set 
    %of sum holograms. The holograms were defined manually based on the 
    %mentioned at the beginning of the script. 
    %%%
    
[Topo] = Create_Topo(Sum);
disp('Topo calculated!');