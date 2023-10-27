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

import time
import imageio
import cupy as cp
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from skimage.draw import circle

import pathlib
import openRawData as opn
import pathlib_variable_names as my_vars

from numpy.core.umath_tests import inner1d

def plot_amenities(what_to_plot,cbar_label, image, recon, colormap, colorbar_lim, mi, ma):
    fig, ax = plt.subplots(1,2, frameon = False, figsize = (12,5))
    cax=ax[0].imshow(recon, cmap = colormap, vmin=mi, vmax=ma)
    ax[1].scatter(jnp.real(image[supportmask_cropped[roi_cropped]==1].flatten()),jnp.imag(image[supportmask_cropped[roi_cropped]==1].flatten()),
                  c= recon[supportmask_cropped[roi_cropped]==1].flatten(),cmap=colormap,
                  vmin=mi, vmax=ma, s=2)
    cbar = plt.colorbar(cax)
    cbar.set_label(cbar_label)
    ax[0].set_axis_off()
    ax[0].annotate('mode:%02d\n%s'%(mode, what_to_plot), (.02, .85), xycoords = 'axes fraction', bbox = {'alpha': .5, 'ec': None, 'fc': 'w', 'lw': None})
    ax[1].set_facecolor('black')
    ax[1].set_xlabel("real part")
    ax[1].set_ylabel("imaginary part")
    lim_plot=1.1*jnp.maximum(jnp.abs(mi),jnp.abs(ma))
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
center=jnp.array([485.43051269, 477.44748039])

# region of interest of our reconstruction
roi_array=jnp.array([343, 403, 490, 547])
roi = jnp.s_[roi_array[2]:roi_array[3], roi_array[0]:roi_array[1]]

# focusing parameters used for the reconstruction
prop_dist,phase,dx,dy=(1.49, 2.27441, 0.0, 0.0) 

# size of the beamstop used, in pixels (will be used make the algorithm ignore it)
bs_diam=58

# CROPPING by 82 pixels: all images are cropped at the borders by 82 pixels as there is no information at the edges.
crop=82


def approx_percentile(input_array, percentile_goals):
    if input_array.size > 100000:
        num_samples = int(input_array.size/100)
        positions = np.random.permutation(np.arange(input_array.size))
        init_indices = positions[:num_samples].flatten()
        sub_array = input_array.flatten()[init_indices]
        return jnp.percentile(sub_array, percentile_goals)
    else:
        return jnp.percentile(input_array, percentile_goals)


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

    mask_pixel_raw=jnp.minimum(mask_pixel_raw, jnp.ones(mask_pixel_raw.shape))

    # import supportmask for phase retrieval
    supportmask_cropped = imageio.imread(proc_p(folder_masks, 'supportmask_cropped.png'))[:,:,0]==255
    supportmask_cropped = jnp.minimum(supportmask_cropped, jnp.ones(supportmask_cropped.shape))
    # add a reference aperture at the center of the hologram
    radius=3
    yy,xx=circle(supportmask_cropped.shape[0]//2,supportmask_cropped.shape[0]//2,radius)
    #supportmask_cropped[yy,xx]=1
    supportmask_cropped = supportmask_cropped.at[yy,xx].set(1)
    return supportmask_cropped, mask_pixel_raw


def create_diff_piece(my_mode=' 1-1', mode_i_names=[], topo_i_names=[]):
    """
    Calculates the Difference Hologram for the given mode, which 
    we know will NOT be correct bc. the calculations for the calculated
    positive and negative pieces are not the exact same.

    Inputs:
    --------

    
    Returns:
    --------
        diff_piece (jnp.array) : Array of difference hologram for the given mode
    """
    #PLACEHOLDER


def create_calculated_pieces(my_mode=' 1-1', helicity=1, useAvg=False,
                             mode_i_names=[], topo_i_names=[]):
    """
    Runs what I understand to be the calculation for the positive & negative helicity 
    calculated mode pieces; however we know these do NOT line up with the actual results

    Math being performed:
    mode_calculated = 0
    for each hologram h
        grab its associated topography hologram topo_h (out of 144)
        alpha_h = tr(topo_h, holo_h) / tr(topo_h, topo_h)
        p_k_prime_h = 2*alpha_h*topo_h  -  holo_h 
        mode_calculated += p_k_prime_h
    if avg:
        mode_calculated / h
        
    Where p_k_prime_h can represent the positie or negative calculated helicity piece,
    depending on the input helicity. The topography hologram that corresponds to the 
    given hologram is provided by the log data

    Inputs
    --------
        my_mode (str) : The mode's data from the log is being used 
                        to generate the calc'd piece
        helicity (1|-1) : the helicity of the relevant pieces/determines which
                        holograms of the mode will be used
        useAvg (bool) : Determines whether to use the avg or sum of all the p_k_prime's
                        Documentation indicates sum is correct
        mode_i_names (lst of strs) : Names of Holograms in mode i to open, 
                        MUST BE INDEXED THE SAME AS topo_i_names
        topo_i_names (lst of strs) : Names of Topography Holographs corresponding to 
                        modes to be opened. MUST BE INDEXED THE SAME AS mode_i_names

    Returns
    --------
        calced_mode (numpy array) : Calculated mode piece, with the helicity 
                corresponding to the input helicity
    """

    all_arrs = []
    topo_name_to_trace = dict()
    topo_name_to_data = dict()
    for i in range(len(mode_i_names)):
        key = topo_i_names[i]
        holo_name = mode_i_names[i].strip('.bin')
        holoArr = opn.openBaseHolo(holo_name, pathtype='f', proced=False, mask=False)
        if key in topo_name_to_trace:
            tr = topo_name_to_trace[key]
            topoArr = topo_name_to_data[key]
        else:
            topoArr = opn.openTopo(topoNumber=key, pathtype='f')
            tr = jnp.sum(inner1d(topoArr, topoArr))
            topo_name_to_trace[key] = tr
            topo_name_to_data[key] = topoArr
        alpha_num = jnp.sum(inner1d(topoArr, holoArr))
        alpha = alpha_num / tr
        p_k_prime = 2*alpha*topoArr - holoArr #Same formula for both p_k_prime and n_k_prime
        all_arrs.append(p_k_prime)
    
    if useAvg:
        calced_mode = sum(all_arrs)/len(all_arrs)
    else:
        calced_mode = sum(all_arrs)
    return calced_mode


def pre_gen_d_calculated_pieces(my_mode=' 1-1', helicity=1, useAvg=False):
    """
    Same as create_calculated_pieces, but instead of calculating from scratch, 
    Uses the previous clustering job's data.
    """

    mode_i_names, topo_i_names = opn.grab_mode_items(my_mode, use_helicty=True, \
                                       helicity=helicity, and_topos=True, pathtype='f')
    calced_mode = create_calculated_pieces(my_mode, helicity, useAvg,
                             mode_i_names, topo_i_names)
    return calced_mode


def pre_gen_d_calculated_diff(my_mode=' 1-1'):
    #PLACEHOLDER
    mode_i_names, topo_i_names = opn.grab_mode_items(my_mode, use_helicty=False, \
                                       helicity=1, and_topos=True, pathtype='f')
    calced_diff = create_diff_piece(my_mode, mode_i_names, topo_i_names)
    return calced_diff


def visualize_mode(pos, neg, mask_pixel_raw, supportmask_cropped, 
                               roi_array, mode=1, crop=82, bs_diam=58,
                               experimental_setup = {}, use_PhRt=True,
                               save_my_fig=True):
    """
    Attempts to use the same processing pipeline as the paper, assuming 
    pos and neg are pre-calculated. Depending on what function is used with 
    this function, they may be different inputs than the pipeline would assume
    (eg not created with the calculated pieces and different holos, just the 
    basic sums).

    Inputs:
    --------
        pos (jax/numpy array) : Positive Helicity Array of the MODE
        neg (jax/numpy array) : Negative Helicity Array of the MODE
        mask_pixel_raw (jnp array?) : The raw mask for the modes
        supportmask_cropped (jnp array?) : The support mask made into a disk
        roi_array (jnp array) : Denotes the rectangular regoin of interest
        mode (int) : Which mode this is - IF 2 or more, currently Breaks - 
                     Should be used to initialize guesses.
        crop (int) : How many pixels to crop at the borders
        bs_diam (int) : Size of the beamstop used
    
    """

    ######################################
    # POS & NEG REPLACES: ORIGINAL BELOW #
    ######################################

    """
    #load pos and neg helicity images and their "calculated" opposite helicity counterparts. Sum all of them up to obtain a topography image
    topo = (np.fromfile(folder+folder_mode+"Pos_Holo_Original_Mode_%02d.bin"%mode) + np.fromfile(folder+folder_mode+"Neg_Holo_Original_Mode_%02d.bin"%mode) + np.fromfile(folder+folder_mode+"Pos_Holo_Calculated_Mode_%02d.bin"%mode) + np.fromfile(folder+folder_mode+"Neg_Holo_Calculated_Mode_%02d.bin"%mode) ).reshape(shape)
    
    #load the difference image
    diff = (np.fromfile(folder+folder_mode+"Diff_Holo_Mode_%02d.bin"%mode)).reshape(shape)

    #define the pos/neg helicity images by adding/subtracting the known difference.
    pos  = ( topo + diff ) / 2
    neg  = ( topo - diff ) / 2
    """


    s1 = time.time()
    # GET RID OF OFFSET (and renormalize images)
    pos2, neg2, _=fth.load_both_RB(pos,neg, crop=50, auto_factor=0.5)
    e1 = time.time()
    print("Offset Adjustment: ", e1 - s1)

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
    e2 = time.time()
    print("Centering: ", e2 - e1)

    pos_cropped,neg_cropped, mask_pixel_cropped, roi_cropped_array=pos2.copy(), neg2.copy(), mask_pixel.copy(), roi_array.copy()
    roi_cropped=jnp.s_[roi_cropped_array[2]:roi_cropped_array[3], roi_cropped_array[0]: roi_cropped_array[1]]
    pos_cropped,neg_cropped, mask_pixel_cropped=pos2[crop:-crop,crop:-crop],neg2[crop:-crop,crop:-crop], mask_pixel[crop:-crop,crop:-crop]

    roi_array_cropped=roi_array*(pos_cropped.shape[0]/pos2.shape[0])
    roi_array_cropped=roi_array_cropped.astype(int)
    roi_cropped = jnp.s_[roi_array_cropped[2]:roi_array_cropped[3], roi_array_cropped[0]:roi_array_cropped[1]]
    e3 = time.time()
    print("Cropping (1): ", e3 - e2)

    #get rid of any remaining offset
    mask_rect=jnp.zeros(pos_cropped.shape)
    N=10
    yy,xx=circle(pos_cropped.shape[0]//2,pos_cropped.shape[0]//2,pos_cropped.shape[0]*(N-1)/N*0.5)
    #mask_rect[yy,xx]=1
    #mask_rect[mask_pixel_cropped==1]=1
    mask_rect = mask_rect.at[yy,xx].set(1)
    mask_rect = mask_rect.at[mask_pixel_cropped==1].set(1)

    thr=1
    e3p5 = time.time()
    pos_cropped -= approx_percentile(jnp.array(pos_cropped[mask_rect==0]),jnp.array(thr))
    neg_cropped -= approx_percentile(jnp.array(neg_cropped[mask_rect==0]),jnp.array(thr))
    e4 = time.time()
    print("Percentile Cropping: ", e4 - e3p5)

    # define an artificial beamstop
    yy, xx = circle(pos_cropped.shape[0]//2, pos_cropped.shape[1]//2+2, bs_diam//2)

    # pixels to mask during phase retrieval
    bsmask_p = mask_pixel_cropped.copy()
    #bsmask_p[pos_cropped<=0]=1
    #bsmask_p[yy,xx]=1
    bsmask_p = bsmask_p.at[pos_cropped<=0].set(1)
    bsmask_p = bsmask_p.at[yy,xx].set(1)
    bsmask_n = mask_pixel_cropped.copy()
    #bsmask_n[neg_cropped<=0]=1
    #bsmask_n[yy,xx]=1
    bsmask_n = bsmask_n.at[neg_cropped<=0].set(1)
    bsmask_n = bsmask_n.at[yy,xx].set(1)
    e5 = time.time()
    print("Beamstop Masking: ", e5 - e4)

    # starting guess for phase retrieval for the first mode.
    Startimage=supportmask_cropped*(pos_cropped*0 + 1.3*1e6*jnp.exp(-1j*2.5)) 
    Startimage=jnp.fft.fftshift(jnp.fft.ifft2(jnp.fft.ifftshift(Startimage)))
    # normalized with respect to the current mode hologram intensity
    Startimage*=jnp.sqrt(jnp.sum(pos_cropped*(1-bsmask_p))/(60*1e9))
    e6 = time.time()
    print("Starting Image Guess setup, multiple ffts: ", e6 - e5)

    # from the second mode onwards,we use the result of mode 1 as a starting guess, after normalizing it by the intensity of the mode to reconstruct
    if mode==2:
        #temp=retrieved_p.copy()
        print()
        print("NOT IMPLEMENTED")
        print()
    if mode>=2:
        #Startimage=temp*np.sqrt(np.sum(pos_cropped*(1-bsmask_p))/np.sum(np.abs(temp)**2*(1-bsmask_p)))
        print()
        print("NOT IMPLEMENTED")
        print()
    
    ############################### start the phase retrieval process
    # fully coherent phase retrieval

    if use_PhRt:
    # positive helicity - 700 RAAR
        retrieved_p=PhR.PhaseRtrv(diffract=jnp.sqrt(jnp.maximum(pos_cropped,jnp.zeros(pos_cropped.shape))), mask=supportmask_cropped, mode='mine',
                        beta_zero=0.5, Nit=700, beta_mode='arctan',Phase=Startimage, bsmask=bsmask_p,average_img=30, Fourier_last=True)
        # positive helicity - 50 ER
        retrieved_p=PhR.PhaseRtrv(diffract=jnp.sqrt(jnp.maximum(pos_cropped,jnp.zeros(pos_cropped.shape))), mask=supportmask_cropped, mode='ER',
                        beta_zero=0.5, Nit=50, beta_mode='const',
                        Phase=retrieved_p, bsmask=bsmask_p,average_img=30, Fourier_last=True)
        # negative helicity - 50 ER
        retrieved_n=PhR.PhaseRtrv(diffract=jnp.sqrt(jnp.maximum(neg_cropped,jnp.zeros(neg_cropped.shape))), mask=supportmask_cropped, mode='ER',
                        beta_zero=0.5, Nit=50, beta_mode='const',
                        Phase=retrieved_p, bsmask=bsmask_n,average_img=30, Fourier_last=True)

    else:
        retrieved_p = jnp.fft.ifft(jnp.fft.ifftshift(pos_cropped)) 
        retrieved_n = jnp.fft.ifft(jnp.fft.ifftshift(neg_cropped))
    #^ Just what I think should be the easiest realspace reconstructions
    e7 = time.time()
    print("Phase Retrieval: ", e7 - e6)


    #focus and reconstruct images into real space
    image_p = fth.reconstructCDI(fth.propagate(retrieved_p, prop_dist*1e-6, experimental_setup))[roi_cropped]
    image_n = fth.reconstructCDI(fth.propagate(retrieved_n, prop_dist*1e-6, experimental_setup))[roi_cropped]
    e8 = time.time()
    print("Reconstruct into real space: ", e8 - e7)


    ############################################
    # THE FOLLOWING LINES WILL DIVIDE BY NONE? #
    #     THOUGH NONE's ARE LATER SET TO 0     #
    ############################################
    #np.seterr(divide='ignore', invalid='ignore')
    
    # let's consider only pixels inside the supportmask for plotting
    #image_p[supportmask_cropped[roi_cropped]==0]=None
    #image_n[supportmask_cropped[roi_cropped]==0]=None
    image_p = image_p.at[supportmask_cropped[roi_cropped]==0].set(None)
    image_n = image_n.at[supportmask_cropped[roi_cropped]==0].set(None)

    phase = -0.6
    recon = (image_p-image_n)/(image_p+image_n)* jnp.exp(1j*phase)
    recon = jnp.real(recon)
    recon = recon.at[jnp.isnan(recon)].set(0)
    #recon[jnp.isnan(recon)] = 0
    #print()
    #print("RECONSTRUCTION: ", recon)
    recon = jnp.fliplr(jnp.flipud(recon))
    if save_my_fig:
        vmi, vma = approx_percentile(jnp.array(recon),jnp.array([1,99]))
        e9 = time.time()
        print("Finishing setup for vis: ", e9 - e8)

        fig, ax = plt.subplots()
        m = ax.imshow(recon, vmin = vmi, vmax = vma, cmap = 'gray')
        plt.savefig('test_img', bbox_inches='tight', transparent = False)
        e10 = time.time()
        print("Saving Fig: ", e10 - e9)
    return recon


def pre_gen_d_construct_pieces(my_mode=' 1-1', avg=True):
    """
    Return the calculated pos & neg mode piece 
    """
    pos = opn.pre_gen_d_open_and_combine_pieces(my_mode, helicity=1, avg=avg)
    neg = opn.pre_gen_d_open_and_combine_pieces(my_mode, helicity=-1, avg=avg)
    return pos, neg


def pre_gen_d_calc_pos_neg_using_diff(my_mode=' 1-1', avg=False):
    """
    Use the pre-calculated clustering to calculate my pieces for the visualization

    Inputs:
    --------
        my_mode (str) : name of mode in the log file.

    Returns:
    --------
        main_pos (jnp.array) : Hologram positive piece for feeding into visualize_mode, 
                            As the documentation describes how to calculate it 
                            (DOES NOT AGREE WITH ACTUAL PRE-GEN'D PIECE)
        main_pos (jnp.array) : Hologram negative piece for feeding into visualize_mode, 
                            As the documentation describes how to calculate it 
                            (DOES NOT AGREE WITH ACTUAL PRE-GEN'D PIECE)
    """
    pos, neg = pre_gen_d_construct_pieces(my_mode, avg)
    pos_calc = pre_gen_d_calculated_pieces(my_mode, helicity=1, useAvg=avg)
    neg_calc = pre_gen_d_calculated_pieces(my_mode, helicity=-1, useAvg=avg)
    diff = pre_gen_d_calculated_diff(my_mode)
    pos_negs_sum = sum([pos, neg, pos_calc, neg_calc])
    print(pos_negs_sum)
    main_pos = ( pos_negs_sum + diff ) / 2
    main_neg = ( pos_negs_sum - diff ) / 2
    return main_pos, main_neg


if __name__ == "__main__":
    folder_masks = my_vars.modeMasks
    pos, neg = pre_gen_d_construct_pieces(my_mode=' 1-1', avg=False)
    #pos, neg = pre_gen_d_calc_pos_neg_using_diff(my_mode=' 1-1', avg=False)
    roi_array = roi_array=jnp.array([343, 403, 490, 547])
    experimental_setup = {'ccd_dist': 18e-2, 'energy': 779.5, 'px_size' : 20e-6}
    supportmask_cropped, mask_pixel_raw = get_mask_pixel(folder_masks)
    s = time.time()
    visualize_mode(pos, neg, mask_pixel_raw, supportmask_cropped, 
                               roi_array, mode=1, crop=82, bs_diam=58,
                               experimental_setup = experimental_setup,
                               use_PhRt=True)
    e = time.time()
    print("Total Time Taken for Visualization: ", e - s)
    """
    TO DO: 
    - Write a func that calcs the diff pieces, then calc the relevant stuff to feed into 
    visualize_mode and see how the results differ
    - Check basic pos & neg visualization w. and w.o avg on (seem to have only done w. avg)
    """