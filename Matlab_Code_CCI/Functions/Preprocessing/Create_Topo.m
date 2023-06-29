function [Topo] = Create_Topo(Sum_Array)

    %%%
    %Function to calculate the topography holograms based on the set of sum
    %holograms. The holograms are defined manually.
    
    %INPUT: Sum_Array: array of all sum holograms [960,972,144] calculated
    %       as the sum of averaged hologram stacks with positive and 
    %       negative helicity
    %OUTPUT: Array [960,972,144] of topography holograms, each pair of
    %stacks of positive and negative helicity holograms is assigned to 1
    %topo holo.
    
    %Requires: Avg_Slice
    %%%
    
Topo = zeros(size(Sum_Array));

%1
for m = 0:12
    Topo(:,:,m+1) = Avg_Slice(Sum_Array,0+1,2+1);  
end

% %2
% for m = 13:16 
%     Topo(:,:,m+1) = Avg_Slice(Sum_Array,10+1,12+1);  
% end
% 
% %3
% for m = 17:23 
%     Topo(:,:,m+1) = Avg_Slice(Sum_Array,16+1,17+1);  
% end
% 
% %4
% for m = 24:32 
%     Topo(:,:,m+1) = Avg_Slice(Sum_Array,22+1,24+1);  
% end
% 
% %5
% for m = 33:45 
%     Topo(:,:,m+1) = Avg_Slice(Sum_Array,31+1,33+1);  
% end
% 
% %6
% for m = 46:56 
%     Topo(:,:,m+1) = Avg_Slice(Sum_Array,44+1,46+1);  
% end
% 
% %7
% for m = 57:63 
%     Topo(:,:,m+1) = Avg_Slice(Sum_Array,55+1,57+1);  
% end
% 
% %8
% for m = 64:67 
%     Topo(:,:,m+1) = Avg_Slice(Sum_Array,62+1,64+1);  
% end
% 
% %9
% for m = 68:73 
%     Topo(:,:,m+1) = Avg_Slice(Sum_Array,67+1,69+1);  
% end
% 
% %10
% for m = 74:79 
%     Topo(:,:,m+1) = Avg_Slice(Sum_Array,71+1,73+1);  
% end
% 
% %11
% for m = 80:85 
%     Topo(:,:,m+1) = Avg_Slice(Sum_Array,79+1,80+1);  
% end
% 
% %12
% for m = 86:89 
%     Topo(:,:,m+1) = Avg_Slice(Sum_Array,83+1,85+1);  
% end
% 
% %13
% for m = 90:94 
%     Topo(:,:,m+1) = Avg_Slice(Sum_Array,89+1,89+1);  
% end
% 
% %14
% for m = 95:98 
%     Topo(:,:,m+1) = Avg_Slice(Sum_Array,94+1,94+1);  
% end
% 
% %15
% for m = 99:117 
%     Topo(:,:,m+1) = Avg_Slice(Sum_Array,97+1,98+1);  
% end
% 
% %16
% for m = 118:127 
%     Topo(:,:,m+1) = Avg_Slice(Sum_Array,115+1,117+1);  
% end
% 
% %17
% for m = 128:size(Sum_Array,3)-1 
%     Topo(:,:,m+1) = Avg_Slice(Sum_Array,125+1,127+1);  
% end   
%       
% %Additional corrections implemented after a review of correlation
% %matrix
Topo(:,:,5) = Topo(:,:,4);
% 
% %18
% for m = 11:15 
%     Topo(:,:,m+1) = Avg_Slice(Sum_Array,9+1,9+1);  
% end
% 
% %19
% for m = 16 
%     Topo(:,:,m+1) = Avg_Slice(Sum_Array,15,15);  
% end
% 
% %20
% Topo(:,:,29) = Avg_Slice(Sum_Array,26,28);
% 
% %21
% Topo(:,:,35) = Avg_Slice(Sum_Array,33,34);
% 
% %This Define Topo does exist
% for m = 46:48 
%     Topo(:,:,m+1) = Avg_Slice(Sum_Array,44+1,46+1);  
% end
% 
% %This Define Topo does exist
% for m = 57:63 
%     Topo(:,:,m+1) = Avg_Slice(Sum_Array,55+1,57+1);  
% end
% 
% %This Define Topo does exist
% for m = 64:67 
%     Topo(:,:,m+1) = Avg_Slice(Sum_Array,62+1,64+1);  
% end
% 
% %This Define Topo does exist
% for m = 74:79 
%     Topo(:,:,m+1) = Avg_Slice(Sum_Array,71+1,73+1);  
% end
% 
% %22
% for m = 80:85 
%     Topo(:,:,m) = Avg_Slice(Sum_Array,81,83);  
% end
% 
% %This Define Topo does exist
% for m = 90:94 
%     Topo(:,:,m) = Avg_Slice(Sum_Array,90,90);  
% end
% 
% %23
% for m = 99:101
%     Topo(:,:,m) = Avg_Slice(Sum_Array,102,102);  
% end
% 
% %24
% Topo(:,:,103) = Avg_Slice(Sum_Array,104,105);
% 
% %25
% for m = 120:123
%     Topo(:,:,m) = Avg_Slice(Sum_Array,124,125);  
% end
% 
% %26
% for m = 130:132
%     Topo(:,:,m) = Avg_Slice(Sum_Array,133,133);  
% end
