import numpy as np


import openRawData as opn
import pathlib_variable_names as var_names

from scipy.optimize import minimize
from sklearn.cluster import SpectralClustering

def distance_from_pure_color(img, pure_colors):
    """
    Calculates the distance the img is from pure colors, eg for all pixels
    find the neared pure color, square that distance, sum over all pixels,
    norm by the number of pixels and the maximum possible distance and sqrt.
    Dist should be [0, 1]

    Args:
        img (_type_): _description_
        pure_colors (_type_): _description_

    Returns:
        _type_: _description_
    """
    img = np.array(img)
    pure_colors = sorted(pure_colors)
    color_midpoints = [(pure_colors[i+1] + pure_colors[i]) / 2 
                       for i in range(len(pure_colors) - 1)]
    previously_updated = np.zeros(img.shape)
    closest_colors = np.zeros(img.shape)
    for i in range(len(color_midpoints)):
        color_midpoint = color_midpoints[i]
        currently_updated = np.zeros(img.shape)
        currently_updated[img < color_midpoint] = 1
        new_pts = currently_updated - previously_updated
        closest_colors[new_pts > 0] = pure_colors[i]
        previously_updated = currently_updated
    closest_colors[previously_updated == 0] = pure_colors[i + 1]
    
    all_large_dists_1 = (color_midpoints - np.array(pure_colors[:-1]))**2
    all_large_dists_2 = (np.array(pure_colors[1:]) - color_midpoints)**2
    largest_possi_dist = max([max(all_large_dists_1), max(all_large_dists_2)])
    
    dist_pc = (closest_colors - img)**2
    dist_pc = np.sum(dist_pc) / (img.size * largest_possi_dist)
    normed_dist_pc = dist_pc 
    
    return normed_dist_pc, closest_colors, largest_possi_dist