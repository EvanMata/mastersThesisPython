%arr = rand(3,3,'double');
%arr2 = rand(3,3,'double');
%mask1 = [[0,1,1],[1,1,1],[0,1,1]];
%mask2 = [[0,1,1],[0,1,1],[1,1,1]];

arr = rand(9,1,'double');
arr2 = rand(9,1,'double');
mask1 = [0,0,1,1,1,1,1,1,1];
mask2 = [0,1,0,1,1,1,1,1,1];

CCI_Correlation_Function(arr, arr2, mask1, mask2);
%filepath = fullfile(pwd,'testFile.bin');
%fId = fopen(filepath,'w');
%fwrite(fId,arr,'float64');
%fclose(fId);