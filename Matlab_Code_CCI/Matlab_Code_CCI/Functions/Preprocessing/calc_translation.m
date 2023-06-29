function [Trans_Vec] = calc_translation(Moving_Image,Static_Image)

    %%%
    %Function calculates relative shift beteween two images by Matlab image
    %registration algorithm
    %Input: Moving_Image:  Shifted Array [m x n] 
    %       Static_Image: Static reference Array [m x n]
    %
    %Input function: 'Matlab Image Processing Toolbox'
    %
    %Output: 2d-Translation vector between Moving_Image and Static_Image
    %%%
    
%Setup Registration Alogrithm
[optimizer, metric] = imregconfig('monomodal');

optimizer.GradientMagnitudeTolerance    = 0.5e-6;
optimizer.MaximumIterations             = 200;
optimizer.MaximumStepLength             = 0.05;
optimizer.MinimumStepLength             = 1e-6;
optimizer.RelaxationFactor              = 0.5;

reg_affine  = imregtform(Moving_Image,Static_Image,'translation',optimizer,metric);
Trans_Vec   = [reg_affine.T(3,1) reg_affine.T(3,2)];