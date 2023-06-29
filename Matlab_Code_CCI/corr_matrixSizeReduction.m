matrixPath = 'C:\Users\evana\Documents\Thesis\Provided\SMALL_SAMPLE\Clustering_Analysis\'; 
matrixName = 'Correlation_Map.bin';

outname = 'Correlation_Map_Mini.bin';

corrMatrixPath = fullfile(matrixPath, matrixName);
fid = fopen(FileName,'r');
Correlation = fread(fid,[28800,28800],'double');
fclose(fid);

%Correlation = single(Correlation);
disp('Loading correlation metric finished!')

%MiniCorrelation = Correlation

%fileID = fopen(outname,'w');


