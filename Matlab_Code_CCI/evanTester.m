%arr = rand(3,3,'double');
%arr2 = rand(3,3,'double');
%mask1 = [[0,1,1],[1,1,1],[0,1,1]];
%mask2 = [[0,1,1],[0,1,1],[1,1,1]];

arr = rand(9,1,'double');
arr2 = rand(9,1,'double');
mask1 = [0,0,1,1,1,1,1,1,1];
mask2 = [0,1,0,1,1,1,1,1,1];



info = CCI_Correlation_Function(arr, arr2, mask1, mask2);
%filepath = fullfile(pwd,'testFile.bin');
%fId = fopen(filepath,'w');
%fwrite(fId,arr,'float64');
%fclose(fId);


function [Corr_Value,Corr_Array,Norm_Factor_1,Norm_Factor_2] = CCI_Correlation_Function(Flattened_Holo_1,Flattened_Holo_2,Statistics_1,Statistics_2)
    
    %%%
    %Function calculates the CCI Fourier-space correlation function using
    %the reference-filtered, flattened, magnetic difference holograms
    %recorded with FTH
    %
    %Input: Flattened_Holo_1,2: reference-filtered, flattened, 
    %       magnetization difference holograms between correlation will be
    %       calculated.
    %       Statistics_1,2: Masks defining relevant areas in q space for 
    %       averaging of correlation(q)
    %
    %Output:Corr_Value: Scalar Correlation Value
    %       Corr_Array: Normalized product of flattened holos
    %       Norm_Factor_1,2: correlation-based normalization Factors of 
    %       flattened holos 
    %%%
    
    
%Process Masks
%Exclude only pixels that relevant neither in holo1 nor holo2 
Statistics = zeros(size(Flattened_Holo_1));
Statistics = or(Statistics_1,Statistics_2);

Flattened_Holo_1 = Flattened_Holo_1.*Statistics;
Flattened_Holo_2 = Flattened_Holo_2.*Statistics;

%Calculate Autocorrelations to calc normalization
Corr_Array = zeros(size(Flattened_Holo_1));

Corr_Array = Flattened_Holo_1.*Flattened_Holo_1;
Norm_Factor_1 = sum(sum(Corr_Array));

Corr_Array = Flattened_Holo_2.*Flattened_Holo_2;
Norm_Factor_2 = sum(sum(Corr_Array));

%Calculate Cross_Correlation
Corr_Array = Flattened_Holo_1.*Flattened_Holo_2;
Corr_Array = Corr_Array/sqrt(Norm_Factor_1*Norm_Factor_2);
Corr_Value = sum(sum(Corr_Array));
end