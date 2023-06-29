function [Flatten_Matrix,Flatten_Mask] = Flatten_Matrix(topo_holo)

    %%%
    %Function removes zeros which would lead to inf pixel values in
    %flattened reference filtered difference holos and mask to further
    %process corrected pixel, i.e., exclude them from analysis
    %
    %Input: topo_holo:  reference-filtered topo holo
    %       preprocessing_mask: additional masked areas, e.g, beamstop
    %
    %Output:Flatten_Matrix: Matrix which will further be used for
    %flattening
    %       Mask which stores adapted pixel        
    %%%

        Flatten_Matrix = topo_holo;
            Flatten_Mask  = (Flatten_Matrix > 0);                           %Exclude Values which give <= 0 -> Divide Problem -> Mask for Values which should be excluded from Correlation Calc
        Flatten_Matrix = Flatten_Matrix + (1-Flatten_Mask).*ones(size(Flatten_Mask));              %Add +1 to 0 Values, to solve Divide Problem 