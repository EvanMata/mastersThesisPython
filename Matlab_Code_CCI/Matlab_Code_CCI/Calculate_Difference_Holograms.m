%% Calculate_Difference_Holograms
    
    %%%
    %Script calculates the difference holograms which is the difference of
    %the generic hologram recorded with positive or negative helicity and
    %the corresponding topography hologram
    %Script also calculates the reference filtered difference and topo
    %holograms which are the inputs of the correlation analysis
    
    %Input Data: 
    % - raw holograms of positive and negative helicity
    % .../Raw_Holo/Raw_Holograms_XXXXX.bin
    % - Topography holograms
    % .../Topography_Holograms/Topography_Hologram_XXX.bin
    % - Log txt for Coherent Correlation imaging
    % .../Log_Data_Coherent_Correlation_Imaging.txt
    %
    %Input Functions: Matlab "Image Processing Toolbox", create_beamstop.m,
    %calc_translation.m, filter_reference.m
    
    %Output:Difference_Holograms, Difference_Holograms_Reference_Filtered,
    %       Topography_Holograms_Reference_Filtered
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
baseFolder.Data     = fullfile(basePath,'Raw_Holograms');                   %Input Folder of raw holograms
baseFolder.Stack    = fullfile(basePath,'Topography_Hologram_Stack');   
baseFolder.Log      = fullfile(basePath);                                   %Input Folder of Log
addpath(genpath(pwd));

%baseFolder.Data     = fullfile(pwd,'Raw_Holo');                             %Input Folder of raw holograms
%baseFolder.Stack     = fullfile(pwd,'Topography_Hologram_Stack');                 %Input Folder of Topo holograms
%baseFolder.Log      = fullfile(pwd);                                        %Input Folder of Log

%% Load Sum Array

    %We recorded 288 hologram stacks with alternating x-ray helicity, each 
    %consisting of 100 image frames.
    %Averages all frames that correspond to coincident image stacks. Then
    %calculates the sum of consecutive image stack with alternating x-ray
    %helicity

%Arrays that save averaged holograms of positive and negative helicity
Pos = zeros(972,960,size(Log,1)/200);
Neg = zeros(972,960,size(Log,1)/200);
    
%Load Image stacks
for i = 1:size(Log,1)/200
    Stack = i-1;                                                  %First Image of each Stack;
    
    %Load Pos helicity image stack
    FileName = fullfile(baseFolder.Stack,'Pos_Helicity',sprintf('%04d.dat',Stack));
    fprintf(1, 'Now reading %s\n', FileName);
    fid = fopen(FileName,'r');
    temp_Array = fread(fid,[972,960],'double');
    fclose(fid);

    Pos(:,:,i) = temp_Array;
    
    %Load Neg helicity image stack
    FileName = fullfile(baseFolder.Stack,'Neg_Helicity',sprintf('%04d.dat',Stack));
    fprintf(1, 'Now reading %s\n', FileName);
    fid = fopen(FileName,'r');
    temp_Array = fread(fid,[972,960],'double');
    fclose(fid);
    
    Neg(:,:,i) = temp_Array;
end

Sum = Pos + Neg;
Sum = rot90(Sum);
disp('Sum Array calculated!')

%% Load Topo Holos and Create Topo Struct

%%% What is this 144? The number of items from Create_Topo? Why is the
%%% last? slot replaced with this struct? Also topo is a 3d array so ????
Topo(144) = struct();                                                       %Stores all relevant information about Topo holograms
Topo(1).Holo = [];                                                          %Stores Holograms
Topo(1).Holo_Ref_Filtered = [];                                             %Stores Reference Filtered Holos
Topo(1).Drift_Vec = [];                                                     %Stores Relative Drift between Data Holos and its corresponding topo holo

[tTopo] = Create_Topo_Stack(Sum);

%Assign to struct
for i = 1:length(Topo)
    Topo(i).Holo = tTopo(:,:,i);
end

%Show Topo holo
f = figure;
ax = gca;
imagesc(Topo(1).Holo,'Parent',ax); axis image; colorbar

%% Load Log and Creates Data Struct
Log = readtable(fullfile(baseFolder.Log,'Log_Data_Coherent_Correlation_Imaging.txt'));
Data = table2struct(Log);                                                   %Stores all relevant information about holograms

Data(1).Holo = [];                                                          %Stores Holos
Data(1).Drift_Vec = [];                                                     %Stores Relative Drift between Data Holos and its corresponding topo holo
Data(1).Dyn_Factor = [];                                                    %Stores Dynamics intensity correction factor
Data(1).Diff_Holo = [];                                                     %Stores Difference Hologram
Data(1).Diff_Holo_Ref_Filtered = [];                                        %Stores Reference Filtered Holos
Data(1).Diff_Holo_Flattened = [];                                           %Stores Flattened Difference Holos

%% Load hologram stack
Stack_ID = 1;                                                               %Loading of i-th Image Stack (100 images)

for i = 1:100
    %Load hologram frame
    Frame = (Stack_ID - 1)*100 + i;
    
    FileName = fullfile(baseFolder.Data,sprintf('Raw_Hologram_%05d.bin',Frame));
    fprintf(1, 'Now reading %s\n', FileName);
    fid = fopen(FileName,'r');
    Data(Frame).Holo = fread(fid,[960,972],'double');
    fclose(fid);
end

%% Mask beamstop of Topo data

    %%% 
    %A smoothed circular region with Radius Radi_bs of the imput image is 
    %set to zero. Covers the experimental beamstop to prevent artifacts in 
    %the Fourier-Inversion of the Holograms (Patterson Map). Edges are 
    %smoothed by Filter to decrease edge artifactis
    %%%

%Parameter of beamstop
Radi_bs = 55;                                                               %Beamstop radius                                                                %Parameter for smooting; the higher sigma, the stronger the smoothing

%Create beamstop mask
[lower, upper] = scaleContrast(Topo(1).Holo,[1 99.9]);
f = figure('visible','off'); 
ax = gca;
imshow(Topo(1).Holo,[lower upper],'Parent',ax);
mask_bs = createMask(imellipse(gca, [(size(Topo(1).Holo,2)-Radi_bs)/2 (size(Topo(1).Holo,1)-Radi_bs)/2 Radi_bs Radi_bs]));
        
%Apply beamstop to Topo Holos
for i = 1:length(Topo)
    Topo(i).Holo = Topo(i).Holo.*(1-mask_bs);    
end

%Show Topo holo
%imagesc(Topo(1).Holo,'Parent',ax); axis image; colorbar
disp('Topo holo beamstop masked!')
figure; imagesc(Topo(1).Holo)

%% Mask beamstop of hologram data

%Find data with holograms
len = find(~cellfun(@isempty,{Data.Holo}));

%Apply beamstop to data holos
for i = len
    Data(i).Holo = Data(i).Holo.*(1-mask_bs);    
end

disp('Data holo beamstop masked!')

%% Calculate Drift between Data holograms

    %%%
    %Relative drift between data holograms and their corresponding topo
    %holograms is calculated by Matlab image registration algorithm. 
    %Necessary to get well defined difference holograms
    %%%
    
%Mask
%Define Reference_Image and Mask
Reference_Image = Topo(1).Holo;
[lower, upper] = scaleContrast(Reference_Image,[10 99]);
figure('visible','off'); imshow(Reference_Image,[lower upper]);
Radi = 120;
mask = createMask(imellipse(gca, [size(Reference_Image,2)/2-Radi/2 size(Reference_Image,1)/2-Radi/2 Radi Radi]));
mask = 1-double(mask);

Reference_Image = Topo(1).Holo.*mask;        

%Store drift data
Drift_Vec = zeros(size(len,2),2);

%Varies loaded data
for i = 1%len
    fprintf(1,'Calculating Image Registration for Data Holo: %d\n',i);
    
    %Get Topo Index Nr that correspond to image frame i
    Topo_Index = Data(i).Topography_Nr_;

    %Define Reference Image; Static Topo holo i
    Ref_Image = Topo(Topo_Index).Holo.*mask;
    
    %Define Moving Image in Alignment algorithm; holo frame i
    temp_Image  = Data(i).Holo.*mask;
    
    %Calc Relative Drift
    Drift_Vec(i,:) = calc_translation(temp_Image,Ref_Image);
    Data(i).Drift_Vec = Drift_Vec(i,:);
end

disp('Drift between Data Holos calculated!')

%% Shift images to apply drift-correction 

    %%%
    %Remove relative drift between data holo and its corresponding
    %reference topo holo is corrected. Alignment of data holo and topo holo.
    %%%

%Varies loaded data
for i = 1%len
    Data(i).Holo  = imtranslate(Data(i).Holo,Drift_Vec(i,:));
end  

disp('Drift corrected!')

%% Calculate drift-corrected difference holograms

    %%%
    %Calculates drift-corrected difference hologram which is the basis for
    %the FTH image reconstruction and CCI. Uses a dynamics intensity factor
    %to match intensity of data holo and topo holo. Based on approximation
    %that magnetization and topography scattering are uncorrelated and
    %therefore orthogonal.
    %%%

%Varies loaded data
for i = 1%len
    fprintf(1,'Calculating Difference Hologram of Data Holo: %d\n',i);
    
    %Get Topo Index Nr that correspond to image frame i
    Topo_Index = Data(i).Topography_Nr_;
    
    %Calculate dynamic intensity correction factor
    Data(i).Dyn_Factor = sum(sum((Data(i).Holo.*Topo(Topo_Index).Holo)))/...
                         sum(sum((Topo(Topo_Index).Holo.*Topo(Topo_Index).Holo)));

    %Calculate difference hologram
    Data(i).Diff_Holo = Data(i).Holo - Data(i).Dyn_Factor.*Topo(Topo_Index).Holo;
end

%Show diff holo
imagesc(Data(i).Diff_Holo,'Parent',ax); axis image; colorbar
disp('Difference Holos calculated!')

%% Filter reference modulations from difference holograms

    %%%
    %Extracts the cross-correlation between the magnetization and 
    %topography scattering from the difference holograms by cropping the
    %central part of the Patterson map.
    %%%
    
%Filtering Parameter 
Dia = 140;
crop_sz = 65;
sigma = 5;

%Create mask for filtering of Patterson Map
temp_Image = Data(len(1)).Holo;                                             %Arbitrary image with 960x972 format
figure('visible','off');
imshow(temp_Image);
mask_pm = createMask(imellipse(gca, [(size(temp_Image,2)-Dia)/2 (size(temp_Image,1)-Dia)/2 Dia Dia]));

h = fspecial('gaussian', 4*sigma+1 , sigma);
mask_pm = imfilter(double(mask_pm), h, 'symmetric', 'conv');


%Varies loaded data
for i = 1%len
    fprintf(1,'Filtering difference holo: %d\n', i)
    
    %Filter_Reference from difference holograms
    [Data(i).Patterson_Map,Data(i).Diff_Holo_Ref_Filtered] = filter_reference(Data(i).Diff_Holo,mask_pm,Dia,crop_sz);
end

%Show reference-filtered diff holo
%imagesc(real(Data(i).Diff_Holo_Ref_Filtered),'Parent',ax); axis image; colorbar
figure; imagesc(real(Data(i).Diff_Holo_Ref_Filtered)); axis image; colorbar
disp('Reference modulations filtered from difference holograms!')

%% Filter reference modulations from topography holograms

    %%%
    %Extracts the auto-correlation of the topography scattering from the 
    %topography holograms by cropping the central part of the Patterson 
    %map.
    %%%

%Varies Topo data
for i = 1:length(Topo)
    fprintf(1,'Filtering topography holo: %d\n', i)
    
    %Filter_Reference from difference holograms
    [Patterson,Reference_Filtered] = filter_reference(Topo(i).Holo,mask_pm,Dia,crop_sz);
    Topo(i).Patterson_Map = Patterson;
    Topo(i).Holo_Ref_Filtered = Reference_Filtered;
end

%Show reference-filtered topo holo
figure; ax = gca; imagesc(real(Topo(1).Holo_Ref_Filtered),'Parent',ax); axis image; colorbar
disp('Reference modulations filtered from topography holograms!')

%% Flatten difference holograms

    %%%
    %The reference-filtered difference holograms exhibit high intensity at
    %small recipocal vectors due to the heterodyne mixing of the
    %magnetization scattering with the topography mask scattering 
    %(Airy-Pattern). They need to be flattened to extract the pure magnetic
    %scattering.
    %%%
    
%Varies loaded data    
for i = 1%len
        %Get Topo Index Nr that correspond to image frame i
        Topo_Index = Data(i).Topography_Nr_;
        
        %Calc flattened holo
        Data(i).Diff_Holo_Flattened = real(Data(i).Diff_Holo_Ref_Filtered)./sqrt(abs(real(Topo(Topo_Index).Holo_Ref_Filtered)));
end

%Show flattended diff holo
figure; ax = gca; imagesc(Data(i).Diff_Holo_Flattened,'Parent',ax); axis image; colorbar
disp('Difference holograms flattened!')