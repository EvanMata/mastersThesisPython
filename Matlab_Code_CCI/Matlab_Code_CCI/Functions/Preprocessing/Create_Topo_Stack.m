function [Topo] = Create_Topo(Sum_Array)

    %%%
    %Function to calculate the topography holograms based on the stack of 
    %holograms. The holograms are defined manually.
    
    %INPUT: Sum_Array: array of all sum holograms [960,972,144] calculated
    %       as the sum of averaged hologram stacks with positive and 
    %       negative helicity
    %OUTPUT: Array [960,972,144] of topography holograms, each pair of
    %stacks of positive and negative helicity holograms is assigned to 1
    %topo holo.
    
    %Requires: Avg_Slice
    %%%
    
Topo = Sum_Array;

%correct only frames where pattern changes during acquisition
Topo(:,:,5) = Topo(:,:,4);
for m = 11:15 
    Topo(:,:,m+1) = Avg_Slice(Sum_Array,9+1,9+1);  
end

for m = 16 
    Topo(:,:,m+1) = Avg_Slice(Sum_Array,15,15);  
end

Topo(:,:,29) = Avg_Slice(Sum_Array,26,28);
Topo(:,:,35) = Avg_Slice(Sum_Array,33,34);

for m = 46:48 
    Topo(:,:,m+1) = Avg_Slice(Sum_Array,44+1,46+1);  
end

for m = 57:63 
    Topo(:,:,m+1) = Avg_Slice(Sum_Array,55+1,57+1);  
end

for m = 64:67 
    Topo(:,:,m+1) = Avg_Slice(Sum_Array,62+1,64+1);  
end

for m = 74:79 
    Topo(:,:,m+1) = Avg_Slice(Sum_Array,71+1,73+1);  
end

for m = 80:85 
    Topo(:,:,m) = Avg_Slice(Sum_Array,81,83);  
end

for m = 90:94 
    Topo(:,:,m) = Avg_Slice(Sum_Array,90,90);  
end

for m = 99:101
    Topo(:,:,m) = Avg_Slice(Sum_Array,102,102);  
end

Topo(:,:,103) = Avg_Slice(Sum_Array,104,105);

for m = 120:123
    Topo(:,:,m) = Avg_Slice(Sum_Array,124,125);  
end

for m = 130:132
    Topo(:,:,m) = Avg_Slice(Sum_Array,133,133);  
end
