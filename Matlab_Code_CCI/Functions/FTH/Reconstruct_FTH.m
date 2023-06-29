function [Reconstruct] = Reconstruct_FTH(Reconstruct,Log,Topo,mask_bs,ROI,Phase)

    %%%
    %Function reconstructs magnetization transmission function from
    %holograms recorded using Fourier-transform holography. 
    %
    %Input: Reconstruct: Stores all relevant information for reconstruction
    %           - .Frames: List of frames to reconstruct frames
    %           - .Folder: Basefolder where raw hologams are saved
    %           - .Name: File Name prefix of ID numbers
    %       Log: General File Log containing frame specific information:
    %           - .Topography_Nr_: Topography Holograms to Frame assignment
    %           - .ID_: File Name Suffixes incl. data type ending
    %           - .Translation_x,y: Components of Translation vector to
    %              correct relative drift between raw holo and topo holo
    %              Shift = [Translation_y Translation_x]
    %           - .Helicity: X-ray helicity of hologram
    %       Topo: Set of Topography holograms to eliminate topography
    %       scattering from raw holograms and calculate magnetization
    %       difference holograms
    %       mask_bs: Beamstop masks. First is pure binary mask for
    %       calculation of dynamic factor to match raw holo and topo holo
    %       intentsity. Second is mask with smooth edges used to prevent
    %       edge effects in Patterson map
    %       ROI: Region of interest in Patterson map
    %       Phase: phase to shift domain contrast of complex reconstruction 
    %       into real part
    %
    %Input functions: prop_back.m
    %
    %Output: 'Reconstruct' struct storing all relevant Reconstruction data
    %
    %%%

    %% Reconstruction
    
    %Predefine Diff Holo Array
    Diff_Array = zeros(size(Topo(1).Holo));
    
    %Init Topo Index relevant for topo updating
    Check_Topo = 0;
    
    %Vary Frames of Modes 
    for k = [Reconstruct.Frames] %Varies Images in Cluster
        %Get IDs for Dataset
        Frame = k;
        Topo_Index = Log.Topography_Nr_(Frame);
        
        %Load Raw hologram
        temp = Log.ID_(Frame);
        FileName = fullfile(Reconstruct.Folder,[Reconstruct.Name temp{1}]);
        fprintf(1, 'Now reading: %s\n', FileName);
        fid = fopen(FileName,'r');
        temp_Holo = fread(fid,size(Topo(1).Holo),'double');
        fclose(fid);    
        
        %Apply temp beamstop and translate holo
        temp_Holo = temp_Holo.*mask_bs(:,:,1);
        temp_Holo = imtranslate(temp_Holo,[Log.Translation_y_(Frame) Log.Translation_x_(Frame)]);
        
        %Update Topo if necessary
        if Topo_Index ~= Check_Topo
            temp_Topo = Topo(Topo_Index).Holo.*mask_bs(:,:,1);
            Check_Topo = Topo_Index;
        end
        
        %Calculate dynamic intensity scaling factor
        Dyn_Factor = sum(sum((temp_Holo.*temp_Topo)))/sum(sum((temp_Topo.*temp_Topo)));
        
        %Calculate Diff Holo
        if Log.Helicity_(Frame) == -1
            temp_Holo = -(temp_Holo - Dyn_Factor.*temp_Topo);
        elseif Log.Helicity_(Frame) == 1
            temp_Holo = +(temp_Holo - Dyn_Factor.*temp_Topo);
        end
        
        Diff_Array = Diff_Array + temp_Holo;
    end
    
    %Average
    Reconstruct.Diff_Array = Diff_Array/Reconstruct.Range;
    
    %FFT to patterson map with numerical wave propagation (focussing)
    Reconstruct.Diff_Array = Reconstruct.Diff_Array.*mask_bs(:,:,2);
    Reconstruct.Diff_Array_Prop = prop_back(Reconstruct.Diff_Array,3.0e-6,33e-2,30e-6,778);
    Reconstruct.FFT_Diff_Array = fftshift(ifft2(ifftshift(Reconstruct.Diff_Array_Prop)));
    
    %Phase Shift and reconstruction
    temp_Reco = rot90(real(Reconstruct.FFT_Diff_Array(ROI(1,:),ROI(2,:))*exp(-1i*Phase*pi/180)),-1);
    temp_Reco = fliplr(temp_Reco);
    Reconstruct.Reco = temp_Reco;