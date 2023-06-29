
holoBase = 'C:\Users\evana\Documents\Thesis\Provided\SMALL_SAMPLE\Raw_Holograms';
holoName = 'Raw_Hologram_01901.bin';
holoPath = fullfile(holoBase, holoName);
fid = fopen(holoPath,'r');
temp_Array = zeros(960,972);
temp_Array(:,:) = fread(fid,[960,972],'double');

contour(temp_Array)