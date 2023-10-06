"""
Python Dictionary for FTH reconstructions

2016/2019/2020
@authors:   MS: Michael Schneider (michaelschneider@mbi-berlin.de)
            KG: Kathinka Gerlinger (kathinka.gerlinger@mbi-berlin.de)
            FB: Felix Buettner (felix.buettner@helmholtz-berlin.de)
            RB: Riccardo Battistelli (riccardo.battistelli@helmholtz-berlin.de)
"""

import numpy as np
from scipy.ndimage.filters import gaussian_filter
import scipy.constants as cst
from skimage.draw import circle


###########################################################################################

#                               LOAD DATA                                                 #

###########################################################################################


def load_both_RB(pos, neg,crop=0, auto_factor=True):
    '''
    Load images for a double helicity reconstruction
    INPUT:  pos, neg: arrays, images of positive and negative helicity
            crop: decide whether you want to ignore the margin of the image when computing the factor
            auto_factor: optional, boolean, determine the factor by which neg is multiplied automatically, if FALSE: factor is set to 0.5 (defualt is False)
    OUTPUT: pos,neg with factor and offset correction, factor
    --------
    author: RB2021
    '''
    size = pos.shape
    if auto_factor:
        if crop==0:
            offset_pos = (np.mean(pos[:10,:10]) + np.mean(pos[-10:,:10]) + np.mean(pos[:10,-10:]) + np.mean(pos[-10:,-10:]))/4
            offset_neg = (np.mean(neg[:10,:10]) + np.mean(neg[-10:,:10]) + np.mean(neg[:10,-10:]) + np.mean(neg[-10:,-10:]))/4
        else:
            offset_pos = (np.mean(pos[crop:crop+10,crop:crop+10]) + np.mean(pos[-10-crop:-crop,crop:10+crop]) + np.mean(pos[crop:10+crop,-10-crop:-crop]) + np.mean(pos[-10-crop:-crop,-10-crop:-crop]))/4
            offset_neg = (np.mean(neg[crop:crop+10,crop:crop+10]) + np.mean(neg[-10-crop:-crop,crop:10+crop]) + np.mean(neg[crop:10+crop,-10-crop:-crop]) + np.mean(neg[-10-crop:-crop,-10-crop:-crop]))/4
            
        pos = pos - offset_pos
        neg = neg - offset_neg
        topo = pos+neg
        if crop==0:
            factor = np.sum(np.multiply(pos,topo))/np.sum(np.multiply(topo, topo))
        else:
            factor = np.sum(np.multiply(pos[crop:-crop,crop:-crop],topo[crop:-crop,crop:-crop]))/np.sum(np.multiply(topo[crop:-crop,crop:-crop], topo[crop:-crop,crop:-crop]))
    else:
        topo = pos + neg
        factor = 0.5

    #make sure to return a quadratic image, otherwise the fft will distort the image
    if size[0]<size[1]:
        return (pos[:, :size[0]], factor * topo[:, :size[0]], factor)
    elif size[0]>size[1]:
        return (pos[:size[1], :], factor * topo[:size[1], :], factor)
    else:
        return (pos, factor/(1-factor)*neg, factor)

###########################################################################################

#                               RECONSTRUCT                                                 #

###########################################################################################

def reconstructCDI(image):
    '''
    Reconstruct the image by fft. must be applied to retrieved images
    -------
    author: RB 2020
    '''
    return np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(image)))


###########################################################################################

#                               CENTERING                                                 #

###########################################################################################

def integer(n):
    '''return the rounded integer (if you cast a number as int, it will floor the number)'''
    return np.int(np.round(n))

def set_center(image, center):
    '''
    this centering routine shifts the image in a cyclical fashion
    INPUT:  image: array, difference hologram
            center: array, center coordinates [x, y]
    OUTPUT: centered hologram
    -------
    author: MS 2016, KG 2019
    '''
    xdim, ydim = image.shape
    xshift = integer(xdim / 2 - center[1])
    yshift = integer(ydim / 2 - center[0])
    image_shift = np.roll(image, yshift, axis=0)
    image_shift = np.roll(image_shift, xshift, axis=1)
    return image_shift


###########################################################################################

#                                 BEAM STOP MASK                                          #

###########################################################################################

def mask_beamstop(image, bs_size, sigma = 3, center = None):
    '''
    A smoothed circular region of the imput image is set to zero.
    INPUT:  image: array, the difference hologram
            bs_size: integer, diameter of the beamstop
            sigma: optional, float, the sigma of the applied gaussian filter (default is 3)
            center: optional, array, if the hologram is not centered, you can input the center coordinates for the beamstop mask. Default is None, so the center of the picture is taken.
    OUTPUT: hologram multiplied with the beamstop mask
    -------
    author: MS 2016, KG 2019
    '''

    #Save the center of the beamstop. If none is given, take the center of the image.
    if center is None:
        x0, y0 = [integer(c/2) for c in image.shape]
    else:
        x0, y0 = [integer(c) for c in center]

    #create the beamstop mask using scikit-image's circle function
    bs_mask = np.zeros(image.shape)
    yy, xx = circle(y0, x0, bs_size/2)
    bs_mask[yy, xx] = 1
    bs_mask = np.logical_not(bs_mask).astype(np.float64)
    #smooth the mask with a gaussion filter    
    bs_mask = gaussian_filter(bs_mask, sigma, mode='constant', cval=1)
    return image*bs_mask

###########################################################################################

#                                 PROPAGATION                                             #

###########################################################################################

def propagate(holo, prop_l, experimental_setup = {'ccd_dist': 18e-2, 'energy': 779.5, 'px_size' : 20e-6}, integer_wl_multiple=True):
    '''
    Parameters:
    ===========
    holo : array, hologram  to be propagated
    prop_l : float, propagation distance [m]
    experimental_setup : optional, dictionary, {CCD - sample distance [m] (default is 18e-2 [m]), photon energy [eV] (default is 779.5 [eV]), physical size of one pixel of the CCD [m] (default is 20e-6 [m])}
    integer_wl_mult : optional, boolean, if true, coerce propagation distance to nearest integermultiple of photon wavelength (default is True)
    
    Returns:
    ========
    holo : propagated hologram
    
    ========
    author: MS 2016
    '''
    wl = cst.h * cst.c / (experimental_setup['energy'] * cst.e)
    if integer_wl_multiple:
        prop_l = np.round(prop_l / wl) * wl

    l1, l2 = holo.shape
    q0, p0 = [s / 2 for s in holo.shape] # centre of the hologram
    q, p = np.mgrid[0:l1, 0:l2]  #grid over CCD pixel coordinates   
    pq_grid = (q - q0) ** 2 + (p - p0) ** 2 #grid over CCD pixel coordinates, (0,0) is the centre position
    dist_wl = 2 * prop_l * np.pi / wl
    phase = (dist_wl * np.sqrt(1 - (experimental_setup['px_size']/ experimental_setup['ccd_dist']) ** 2 * pq_grid))
    holo = np.exp(1j * phase) * holo

    #print ('Propagation distance: %.2fum' % (prop_l*1e6)) 
    return holo