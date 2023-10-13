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
import jax.numpy as jnp
import matplotlib.pyplot as plt
from skimage.draw import circle

import pathlib
import openRawData as opn
import pathlib_variable_names as my_vars

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
    
############################
# Setup Constants and such #
############################

shape=(972,960)
folder = './Input_Holograms/'
folder_save = './processed/reconstructions/'
folder_masks= './processed/masks/'

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

# CROPPING by 82 pixels: all images are cropped at the borders by 82 pixels as there is no information at the edges.
crop=82


def proc_p(folder_masks, additional_path):
    return str(folder_masks.joinpath(additional_path))


def get_mask_pixel(folder_masks):
    """
    Gets the masks

    Inputs:
    --------
        folder_masks (str/folder) : Folder the paint masks are in.

    Returns:
    --------
        mask_pixel_raw () :
        supportmask_cropped () : 
    
    """

    # import mask for defective pixels- mask_pixel_raw is 1 where the pixels have to be ignored

    mask_pixel_raw = imageio.imread(proc_p(folder_masks, 'mask_pixel_paint.png'))[:,:,0]==255
    mask_pixel_raw = imageio.imread(proc_p(folder_masks, 'mask_pixel2_paint.png'))[:,:,0]==255
    mask_pixel_raw = imageio.imread(proc_p(folder_masks, 'mask_pixel3_paint.png'))[:,:,0]==255
    mask_pixel_raw = imageio.imread(proc_p(folder_masks, 'mask_pixel4_paint.png'))[:,:,0]==255

    mask_pixel_raw=np.minimum(mask_pixel_raw, np.ones(mask_pixel_raw.shape))

    # import supportmask for phase retrieval
    supportmask_cropped = imageio.imread(proc_p(folder_masks, 'supportmask_cropped.png'))[:,:,0]==255
    supportmask_cropped = np.minimum(supportmask_cropped, np.ones(supportmask_cropped.shape))
    # add a reference aperture at the center of the hologram
    radius=3
    yy,xx=circle(supportmask_cropped.shape[0]//2,supportmask_cropped.shape[0]//2,radius)
    supportmask_cropped[yy,xx]=1
    return supportmask_cropped, mask_pixel_raw

def make_pieces_no_p_retrieval(pos, neg, mask_pixel_raw, supportmask_cropped, 
                               roi_array, mode=1, crop=82, bs_diam=58,
                               experimental_setup = {}):
    """
    Attempts to use the same processing but without the phase retrieval algorithm

    Inputs:
    --------
        pos (jax/numpy array) : Positive Helicity Array of the MODE
        neg (jax/numpy array) : Negative Helicity Array of the MODE
        mask_pixel_raw (np array?) : The raw mask for the modes
        supportmask_cropped (np array?) : The support mask made into a disk
        roi_array (np array) : Denotes the rectangular regoin of interest
        mode (int) : Which mode this is - IF 2 or more, currently Breaks - 
                     Should be used to initialize guesses.
        crop (int) : How many pixels to crop at the borders
        bs_diam (int) : Size of the beamstop used
    
    """

    # GET RID OF OFFSET (and renormalize images)
    pos2, neg2, _=fth.load_both_RB(pos,neg, crop=50, auto_factor=0.5)

    #make sure to return a quadratic image, otherwise the fft will distort the image
    size = mask_pixel_raw.shape
    if size[0]<size[1]:
        mask_pixel=mask_pixel_raw[ :,:-(size[1]-size[0])]
    elif size[0]>size[1]:
        mask_pixel=mask_pixel_raw[:-(size[0]-size[1]),:]
    else:
        mask_pixel=mask_pixel_raw.copy()

    #CENTERING
    pos2= fth.set_center(pos2, center)
    neg2= fth.set_center(neg2, center)
    mask_pixel= fth.set_center(mask_pixel, center)

    pos_cropped,neg_cropped, mask_pixel_cropped, roi_cropped_array=pos2.copy(), neg2.copy(), mask_pixel.copy(), roi_array.copy()
    roi_cropped=np.s_[roi_cropped_array[2]:roi_cropped_array[3], roi_cropped_array[0]: roi_cropped_array[1]]
    pos_cropped,neg_cropped, mask_pixel_cropped=pos2[crop:-crop,crop:-crop],neg2[crop:-crop,crop:-crop], mask_pixel[crop:-crop,crop:-crop]

    roi_array_cropped=roi_array*(pos_cropped.shape[0]/pos2.shape[0])
    roi_array_cropped=roi_array_cropped.astype(int)
    roi_cropped = np.s_[roi_array_cropped[2]:roi_array_cropped[3], roi_array_cropped[0]:roi_array_cropped[1]]


    #get rid of any remaining offset
    mask_rect=np.zeros(pos_cropped.shape)
    N=10
    yy,xx=circle(pos_cropped.shape[0]//2,pos_cropped.shape[0]//2,pos_cropped.shape[0]*(N-1)/N*0.5)
    mask_rect[yy,xx]=1
    mask_rect[mask_pixel_cropped==1]=1

    thr=1
    pos_cropped -= np.percentile(pos_cropped[mask_rect==0],thr)
    neg_cropped -= np.percentile(neg_cropped[mask_rect==0],thr)


    # define an artificial beamstop
    yy,xx = circle(pos_cropped.shape[0]//2, pos_cropped.shape[1]//2+2, bs_diam//2)

    # pixels to mask during phase retrieval
    bsmask_p=mask_pixel_cropped.copy()
    bsmask_p[pos_cropped<=0]=1
    bsmask_p[yy,xx]=1
    bsmask_n=mask_pixel_cropped.copy()
    bsmask_n[neg_cropped<=0]=1
    bsmask_n[yy,xx]=1


    # starting guess for phase retrieval for the first mode.
    Startimage=supportmask_cropped*(pos_cropped*0 + 1.3*1e6*np.exp(-1j*2.5)) 
    Startimage=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Startimage)))
    # normalized with respect to the current mode hologram intensity
    Startimage*=np.sqrt(np.sum(pos_cropped*(1-bsmask_p))/(60*1e9))

    # from the second mode onwards,we use the result of mode 1 as a starting guess, after normalizing it by the intensity of the mode to reconstruct
    if mode==2:
        #temp=retrieved_p.copy()
        a=5
        print("NOT IMPLEMENTED")
    if mode>=2:
        #Startimage=temp*np.sqrt(np.sum(pos_cropped*(1-bsmask_p))/np.sum(np.abs(temp)**2*(1-bsmask_p)))
        a=5
        print("NOT IMPLEMENTED")
    """
    ############################### start the phase retrieval process
    # fully coherent phase retrieval

    # positive helicity - 700 RAAR
    retrieved_p=PhR.PhaseRtrv(diffract=np.sqrt(np.maximum(pos_cropped,np.zeros(pos_cropped.shape))), mask=supportmask_cropped, mode='mine',
                    beta_zero=0.5, Nit=700, beta_mode='arctan',Phase=Startimage, bsmask=bsmask_p,average_img=30, Fourier_last=True)
    # positive helicity - 50 ER
    retrieved_p=PhR.PhaseRtrv(diffract=np.sqrt(np.maximum(pos_cropped,np.zeros(pos_cropped.shape))), mask=supportmask_cropped, mode='ER',
                    beta_zero=0.5, Nit=50, beta_mode='const',
                    Phase=retrieved_p, bsmask=bsmask_p,average_img=30, Fourier_last=True)
    # negative helicity - 50 ER
    retrieved_n=PhR.PhaseRtrv(diffract=np.sqrt(np.maximum(neg_cropped,np.zeros(neg_cropped.shape))), mask=supportmask_cropped, mode='ER',
                    beta_zero=0.5, Nit=50, beta_mode='const',
                    Phase=retrieved_p, bsmask=bsmask_n,average_img=30, Fourier_last=True)
    """

    """
    retrieved_p = jnp.fft.ifft(jnp.fft.ifftshift(pos_cropped)) 
    retrieved_n = jnp.fft.ifft(jnp.fft.ifftshift(neg_cropped))
    """
    #^ Just what I think should be the easiest realspace reconstructions

    #focus and reconstruct images into real space
    image_p = fth.reconstructCDI(fth.propagate(retrieved_p, prop_dist*1e-6, experimental_setup))[roi_cropped]
    image_n = fth.reconstructCDI(fth.propagate(retrieved_n, prop_dist*1e-6, experimental_setup))[roi_cropped]
    # let's consider only pixels inside the supportmask for plotting
    image_p[supportmask_cropped[roi_cropped]==0]=None
    image_n[supportmask_cropped[roi_cropped]==0]=None

    phase = -0.6
    recon = (image_p-image_n)/(image_p+image_n)* np.exp(1j*phase)
    recon = np.real(recon)
    recon[np.isnan(recon)] = 0
    recon = np.fliplr(np.flipud(recon))
    fig, ax = plt.subplots()
    vmi, vma = np.percentile(recon,[1,99])
    m = ax.imshow(recon, vmin = vmi, vmax = vma, cmap = 'gray')
    plt.savefig('test_img', bbox_inches='tight', transparent = False)
    #plt.show()




def construct_pieces(my_mode=' 1-1', avg=True):
    pos = opn.open_and_combine_pieces(my_mode, helicity=1, avg=avg)
    neg = opn.open_and_combine_pieces(my_mode, helicity=-1, avg=avg)
    return pos, neg


if __name__ == "__main__":
    folder_masks = my_vars.modeMasks
    pos, neg = construct_pieces(my_mode=' 1-1', avg=True)
    roi_array = roi_array=np.array([343, 403, 490, 547])
    experimental_setup = {'ccd_dist': 18e-2, 'energy': 779.5, 'px_size' : 20e-6}
    supportmask_cropped, mask_pixel_raw = get_mask_pixel(folder_masks)
    make_pieces_no_p_retrieval(pos, neg, mask_pixel_raw, supportmask_cropped, 
                               roi_array, mode=1, crop=82, bs_diam=58,
                               experimental_setup = experimental_setup)