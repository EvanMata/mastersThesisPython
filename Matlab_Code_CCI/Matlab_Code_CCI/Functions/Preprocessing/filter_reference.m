function [Patterson,Reference_Filtered] = filter_reference(holo,mask_patterson,Dia_crop,crop_sz)

    %%%
    %Function calculates inverse Fourier-transform of holograms to obtain the
    %Patterson map. Then, the central part of the map, defined by
    %mask_patterson, is extracted to isolate the central auto-/
    %cross-correlation. A subsequent Fourier-transform yields the
    %reference-filtered holograms.
    %
    %Input: holo: FTH hologram [m x n]
    %       mask_patterson: Binary [m x n] mask that defines the area
    %       cropped from the Patterson map (==1)
    %       Dia_crop: Diameter of the cropping area of mask_patterson
    %       crop_sz: Size of the reference-filtered hologram
    %
    %Output: Patterson_Map and Refererence-Filtered holo
    %%%
    
%Fourier Transform: Holo --> Patterson Map and crop reference convolution
Patterson = fftshift(ifft2(ifftshift(holo)));                               %FT of holo
Patterson = Patterson.*complex(mask_patterson,mask_patterson);              %Crops central part from Patterson Map

%Use only Relevant section size from Patterson_Map
sz = size(Patterson);                                                 %Additional correction of Patterson map center
Patterson = Patterson(  (floor((sz(1)-Dia_crop)/2)):...
                        (floor((sz(1)+Dia_crop)/2)),...
                        (floor((sz(2)-Dia_crop)/2)):...
                        (floor((sz(2)+Dia_crop)/2)));   %Reduces array to crop size

%Inverse Fourier-Transformation & crop filtered holo
Correction_Shift = [4 4];  
Reference_Filtered = ifftshift(fft2(fftshift(Patterson)));                  %iFFT
sz = size(Reference_Filtered);
Reference_Filtered = Reference_Filtered( floor((sz(1)-crop_sz)/2):...       %Crop to final size of reference filtered holo
                                        (floor((sz(1)+crop_sz)/2)+Correction_Shift(1)),...        %excludes parts without scattering signal
                                         floor((sz(2)-crop_sz)/2):...
                                        (floor((sz(2)+crop_sz)/2)+Correction_Shift(2)));
