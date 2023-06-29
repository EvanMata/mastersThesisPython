
function HOLO = prop_back(HOLO, prop_l, CCD_S_DIST,PX_SIZE, Energy)
% HOLO=Reconstruct(L).Diff_Array;
% prop_1 = 2.7e-6;
% CCD_S_DIST=33e-2;
% PX_SIZE = 30e-6;
% Energy = 778;
% fast backpropagation algortihm
% prop_l: distance to be propagated in m
% CCD_S_DIST: sample CCD dist in m
% PX_SIZE: pixelsize in m
% Energy: energy in eV 
%
% Jan G. based on Carsten Tiegs Igor code and Erik Gührs BP paper  

H_center_q=size(HOLO,1)/2;
H_center_p=size(HOLO,2)/2;
lambda=(1239.84172/Energy)*1e-9;

%q=(1:2*H_center_q)'*ones(1,2*H_center_q);
%p=((1:2*H_center_p)'*ones(1,2*H_center_p))';
[q,p]=meshgrid((1:2*H_center_q),(1:2*H_center_p));
pq_grid = (q-H_center_q).^2+(p-H_center_p).^2;

phase=(prop_l*2*pi/lambda)*sqrt(1-((PX_SIZE/CCD_S_DIST)^2)*pq_grid); 

HOLO=exp(1i*transpose(phase)).*HOLO;

