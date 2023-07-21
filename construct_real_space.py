"""
Code is copied primarily from CDI_for_CCI.ipynb but written as a file with a functional breakdown
"""

import sys, os
sys.path.append('./python_provided_code/')

import fth_reconstruction as fth

try:
    is_there_GPU=True
    import Phase_Retrieval as PhR
except:
    is_there_GPU=False
    print("ERROR!!!!!!!!!!!!!!!!!!!!! \n impossible to import Phase_Retrieval.py. no GPU")
    print("without a GPU, Phase_Retrieval_noGPU.py is imported instead. The code will work, but very slowly")
    import Phase_Retrieval_noGPU as PhR

import imageio
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import disk

def plot_amenities(what_to_plot,cbar_label, image, recon, colormap, colorbar_lim, mi, ma):
    fig, ax = plt.subplots(1,2, frameon = False, figsize = (12,5))
    cax=ax[0].imshow(recon, cmap = colormap, vmin=mi, vmax=ma)
    ax[1].scatter(np.real(image[supportmask_cropped[roi_cropped]==1].flatten()),np.imag(image[supportmask_cropped[roi_cropped]==1].flatten()),
                  c= recon[supportmask_cropped[roi_cropped]==1].flatten(),cmap=colormap,
                  vmin=mi, vmax=ma, s=2)
    cbar = plt.colorbar(cax)
    cbar.set_label(cbar_label)
    ax[0].set_axis_off()
    ax[0].annotate('mode:%02d\n%s'%(mode, what_to_plot), (.02, .85), xycoords = 'axes fraction', bbox = {'alpha': .5, 'ec': None, 'fc': 'w', 'lw': None})
    ax[1].set_facecolor('black')
    ax[1].set_xlabel("real part")
    ax[1].set_ylabel("imaginary part")
    lim_plot=1.1*np.maximum(np.abs(mi),np.abs(ma))
    plt.xlim(-lim_plot, lim_plot)
    plt.ylim(-lim_plot, lim_plot)
    
# camera-sample distance, energy and CCD pixel size. they are useful to focus the hologram
experimental_setup = {'ccd_dist': 18e-2, 'energy': 779.5, 'px_size' : 20e-6}

# center of the experimental hologram
center=np.array([485.43051269, 477.44748039])

# region of interest of our reconstruction
roi_array=np.array([343, 403, 490, 547])
roi = np.s_[roi_array[2]:roi_array[3], roi_array[0]:roi_array[1]]

# focusing parameters used for the reconstruction
prop_dist,phase,dx,dy=(1.49, 2.27441, 0.0, 0.0) 

# size of the beamstop used, in pixels (will be used make the algorithm ignore it)
bs_diam=58


if __name__ == "__main__":
    a=5