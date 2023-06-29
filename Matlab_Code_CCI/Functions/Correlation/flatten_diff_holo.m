function [flat_holo] = flatten_diff_holo(diff_holo,topo_holo)

    %%%
    %Function flattens the difference holograms of FTH data to correct the 
    %high intentensity at small reciprocal vectors. Divides difference 
    %holograms by the topography hologram.
    %Input: diff_holo: FTH difference hologram
    %       topo_holo: FTH hologram repesenting scattering of topography
    %       mask
    %
    %Output: flat_holo: flattened hologram
    %%%
    
    flat_holo = diff_holo./sqrt(topo_holo);