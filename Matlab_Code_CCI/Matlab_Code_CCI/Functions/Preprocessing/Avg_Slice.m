function [Avg_Matrix] = Avg_Slice(Matrix, NrStart, NrEnd)

    %%%
    %Function calculates average of a given range of slices in dim 3
    %Input: Matrix: Original Matrix [m x n x o],
    %       NrStart: First Slice of range for averaging
    %       NrEnd: Last Slice of range for averaging
    %Output; Avg_Image: Avg of Slice defined by range(NrStart,NrEnd)
    %%%
    
Avg_Matrix = mean(Matrix(:,:,NrStart:NrEnd),3);

