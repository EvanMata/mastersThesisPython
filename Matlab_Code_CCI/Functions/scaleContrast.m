function [lower,upper] = scaleContrast(im,cutOff)
% scales contrast of image from CUTOFF(1) to CUTOFF(2) of IM
%   example: [lower, upper] = scaleContrast(IM,[5, 95]);
%   This scales the image to 5%-95% of the image value range
%
% ph, 30.05.2017

vec = sort(im(:));
lower = vec(round(cutOff(1)*length(vec)/100));
upper = vec(round(cutOff(2)*length(vec)/100));
if lower==upper
    lower = lower-1;
    upper = upper+1;
end
