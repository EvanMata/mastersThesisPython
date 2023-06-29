function [mask_bs] = create_beamstop(sz,centers,Radi,sigma)

    %%%
    %Function creates circle with smooted edges 
    %Input: sz: Targeted array size,
    %       centers: center coordinates of circle
    %       Radi: Radius of circle
    %       sigma: variance of the gaussian filter
    %
    %Input function: createCirclesMask.m;
    %
    %Output: mask_bs: Array of size sz 
    %%%

h = fspecial('gaussian', 4*sigma+1 , sigma);
  
mask_bs = createCirclesMask(sz,centers,Radi);
mask_bs = imfilter(double(mask_bs), h, 'symmetric', 'conv');
    

