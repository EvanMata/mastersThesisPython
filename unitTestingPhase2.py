import os
import sys
import jax

import numpy as np
import jax.numpy as jnp
import clustering as clu
import openRawData as opn
#import construct_real_space as con
import genRealisticData as genR
import pathlib_variable_names as var_names
import matplotlib.pyplot as plt

from jax import random
from pathlib import Path
from itertools import combinations

#sys.path.append('./python_provided_code/')
#from fth_reconstruction import reconstructCDI as my_fft


def t_dPC_1():
    """
    Test distance from pure color with trivial example 
    -- PASSED
    """
    test_arr = np.array([[1, 0], [0, 1]])
    pure_colors = [0, 1]
    normed_dist_pc, closest_colors, largest_possi_dist = \
        clu.distance_from_pure_color(test_arr, pure_colors)
    print(normed_dist_pc == 0)
    print(largest_possi_dist == .5**2)
    print(closest_colors)


def t_dPC_2():
    """
    Test distance from pure color with pts not exactly on colors, 
    and close to both possi pure colors
    -- PASSED
    """
    test_arr = np.array([[.4, .6], [.3, .5]])
    pure_colors = [0, 1]
    normed_dist_pc, closest_colors, largest_possi_dist = \
        clu.distance_from_pure_color(test_arr, pure_colors)
    print(normed_dist_pc - ((.4**2 + .4**2 + .3**2 + .5**2)/(4*.5**2))**0.5 < 0.01)
    print(largest_possi_dist == .5**2)
    print(closest_colors)
    print()


def t_dPC_3():
    """
    Test distance from pure color with pts not exactly on colors, 
    3 colors / multiple midpoints and an unsorted colors group
    -- PASSED
    """
    test_arr = np.array([[.4, .6], [.2, .5]])
    pure_colors = [0, 1, .5]
    normed_dist_pc, closest_colors, largest_possi_dist = \
        clu.distance_from_pure_color(test_arr, pure_colors)
    print(normed_dist_pc - ((.1**2 + .1**2 + .2**20)/(4*.25**2))**0.5 < 0.01)
    print(largest_possi_dist == .25**2)
    print(closest_colors)
    print()
    
    
def t_dPC_4():
    """
    Test distance from pure color with pts not exactly on colors, 
    3 colors / multiple midpoints where not all midpoints
    are equadistant from all colors and an unsorted colors group
    -- PASSED
    """
    test_arr = np.array([[.4, .6], [.2, .9]])
    pure_colors = [0, 1, .6]
    normed_dist_pc, closest_colors, largest_possi_dist = \
        clu.distance_from_pure_color(test_arr, pure_colors)
    print(normed_dist_pc - ((.2**2 + 0**2 + .2**2 + .1**2)/(4*.3**2))**0.5 < 0.01)
    print(largest_possi_dist == .3**2)
    print(closest_colors)
    print()


def t_dPC_5():
    """
    Test distance from pure color with pts not exactly on colors, 
    3 colors / multiple midpoints where not all midpoints
    are equadistant from all colors and largest possi distances 
    given non-uniform midpoints
    -- PASSED
    """
    test_arr = np.array([[.3, .3], [.3, .8]])
    pure_colors = [0, 1, .6]
    normed_dist_pc, closest_colors, largest_possi_dist = \
        clu.distance_from_pure_color(test_arr, pure_colors)
    print(normed_dist_pc - ((3*.3**2 + .2**2)/(4*.3**2))**0.5 < 0.01)
    print(largest_possi_dist == .3**2)
    print(closest_colors)
    print()
    print(normed_dist_pc)
    

def t_dPC_6():
    """
    Test distance from pure color with pts not exactly on colors, 
    3 colors / multiple midpoints where not all midpoints
    are equadistant from all colors and All largest possi distances 
    (requires all midpoints same distances)
    -- PASSED
    """
    test_arr = np.array([[.25, .75], [.25, .75]])
    pure_colors = [0, 1, .5]
    normed_dist_pc, closest_colors, largest_possi_dist = \
        clu.distance_from_pure_color(test_arr, pure_colors)
    print(normed_dist_pc - 1 < 0.01)
    print(largest_possi_dist == .25**2)
    print(closest_colors)
    print()
    
    
def t_dTV_1():
    test_arr = np.array([[0, 1], [1, 0]])
    print(clu.total_variation_norm(test_arr) == 1)


def t_mode_is_avg(using_helicity=True, print_it=True, my_mode=' 1-1'):
    """
    Check whether the sum of all the holos in mode 1-1 is actually 
    the positive holo of any of the modes

    3 folders w. 1 img per holo: base/raw holos, Diff Holos Flattened, Diff Holos Ref Filtered

    Can compare avged or summed vs mode item, can compare avg of raw, holo flatted, holo ref filtered 
    can compare vs pos holo og, neg holo og, pos holo calc, neg holo calc, or diff holo
    """
    if using_helicity:
        mode_0_names = opn.grab_mode_items(my_mode=my_mode, use_helicty=True)
    else:
        mode_0_names = opn.grab_mode_items(my_mode=my_mode, use_helicty=False)

    num_holos = 0
    base_arr = np.zeros((972, 960))
    raw_path = var_names.rawHoloNameS
    for holo_name in mode_0_names:
        holoNumber = holo_name.strip(".bin")
        holo_arr = opn.openBaseHolo(holoNumber, pathtype='f', proced=False, mask=False)
        base_arr += holo_arr
        num_holos += 1

    if print_it:
        differences = []
        for pos_holo in opn.yield_mode_pieces():
            sum_holo_diff = np.abs(np.sum(pos_holo - base_arr))
            print("DIFF: ", "{:e}".format(sum_holo_diff))
            differences.append(np.abs(sum_holo_diff))
        #print("MAX Val in constructed array: ", "{:e}".format(np.max(base_arr)))
        print("Minimum Difference: ", "{:e}".format(min(differences)))

    return base_arr


def visualize_fft(my_mode=' 1-1', using_helicity=True):
    avg_holo_arr = t_mode_is_avg(my_mode=my_mode, using_helicity=using_helicity, print_it=False)
    real_sp_maybe = my_fft(avg_holo_arr)
    opn.heatMapImg(real_sp_maybe)


def t1_clustering_caches():
    # Should be YES
    l1 = [0,0,1,1,]
    l2 = [1,1,0,1]
    nl1 = clu.clustering_to_cachable_labels(l1)
    nl2 = clu.clustering_to_cachable_labels(l2)

    for i in range(len(nl1)):
        if nl1[i] != nl2[i]:
            print("NOT the same clustering")
            return
    print("SAME clustering")


def t2_clustering_caches():
    # Should be NO, order of points matters
    l1 = [0,0,1,1,2]
    l2 = [1,1,0,2,2]
    nl1 = clu.clustering_to_cachable_labels(l1)
    nl2 = clu.clustering_to_cachable_labels(l2)

    for i in range(len(nl1)):
        if nl1[i] != nl2[i]:
            print("NOT the same clustering")
            return
    print("SAME clustering")


def t3_clustering_caches():
    # Should be YES, order of points matters
    l1 = [0,0,1,1,2,3,4]
    l2 = [1,1,0,0,2,4,3]
    nl1 = clu.clustering_to_cachable_labels(l1)
    nl2 = clu.clustering_to_cachable_labels(l2)

    for i in range(len(nl1)):
        if nl1[i] != nl2[i]:
            print("NOT the same clustering")
            return
    print("SAME clustering")


def t4_clustering_caches():
    # Should be NO, order of points matters
    l1 = [0,0,1,1,2,3,4,5,5,5]
    l2 = [5,1,5,1,0,0,2,4,3,5]
    nl1 = clu.clustering_to_cachable_labels(l1)
    nl2 = clu.clustering_to_cachable_labels(l2)

    for i in range(len(nl1)):
        if nl1[i] != nl2[i]:
            print("NOT the same clustering")
            return
    print("SAME clustering")


def t5_clustering_caches():
    # Should be NO, order of points matters
    l1 = [0,0,1,1,5,5,5,2,3,4]
    l2 = [1,1,0,0,5,5,5,2,4,3]
    nl1 = clu.clustering_to_cachable_labels(l1)
    nl2 = clu.clustering_to_cachable_labels(l2)

    for i in range(len(nl1)):
        if nl1[i] != nl2[i]:
            print("NOT the same clustering")
            return
    print("SAME clustering")


def t_reverse_str(arr):
    ar_s = jnp.array_str(arr)
    arr2 = clu.jnp_array_from_str(ar_s, arr.shape)
    print(jnp.array_equal(arr, arr2))


def t_vmaped_mat_construction(arr_of_imgs, gamma=0.5):
    def ez_mat_func(im1, im2, m_gamma):
        im = im1 + im2
        val = m_gamma + jnp.sum(im)
        return val
    print(type(arr_of_imgs))
    print(arr_of_imgs.shape)
    arr = clu.affinity_matrix(arr_of_imgs, gamma)
    print(arr)
    print()
    """
    arr2 = clu.affinity_matrix(arr_of_imgs, gamma, 
                                 pair_affinity_func=clu.calcPairAffinity, 
                                 pair_affinity_parallel_axes=(0, 0, None))
    print(arr2)
    print()
    arr3 = clu.affinity_matrix(arr_of_imgs, gamma, 
                                 pair_affinity_func=ez_mat_func, 
                                 pair_affinity_parallel_axes=(0, 0, None))
    print(arr3)
    """
    return arr


def gen_random_uni_arr(my_shape=(1000,)):
    key = random.PRNGKey(758493)  # Random seed is explicit in JAX
    return random.uniform(key, shape=my_shape)


def t_topo_holo(topo_num=1, pathtype='f'): #topo_num in [1,144]
    # THIS IS NOT HOW TOPOS ARE GENERATED.
    """
    topo_num is the name of the topography hologram you're trying to make.
    pathtype is where to load the holos from. Gotta be f otherwise incorrect holo_arr
    """
    differences = []
    holo_nums = list(range(topo_num,28800,50))
    holo_names = [str(n).zfill(5) for n in holo_nums]
    base_arr = np.zeros((972, 960))
    for holo_name in holo_names:
        holo_arr = opn.openBaseHolo(holo_name, pathtype, proced=False, mask=False)
        base_arr += holo_arr
    
    #Open each topo holo, see if base_arr is same as any of them.
    for i in range(1,144+1):
        topo_num = str(i).zfill(3)
        topo_arr = opn.openTopo(topo_num, pathtype)
        mag = np.abs(np.sum(topo_arr - base_arr))
        print("Topo %i diff: "%i, "{:e}".format(mag))
        differences.append(mag)

    print("Minimum Difference: ", "{:e}".format(min(differences)))


def t_generate_holo_calculated_mode(my_mode=' 1-1', helicity=1, useAvg=False):
    """
    Tests if the calculated piece actually lines up with any of the calculated pieces, 
    and gives me a relative sense of how much its different by for the minimum 
    relative to all of the pieces.

    Math being performed:
    for each hologram, grab its associated topography hologram
    Then calculate alpha = tr(topo, holo) / tr(topo, topo)    
    """

    pos_differences = []
    neg_differences = []
    calced_mode = con.pre_gen_d_calculated_pieces(my_mode, helicity, useAvg)

    pos_calced_pieces, neg_calced_pieces = opn.grab_calced_modes()
    for i in range(len(pos_calced_pieces)):
        piece = pos_calced_pieces[i]
        mag = np.abs(np.sum(piece - calced_mode))
        print("Pos Piece %i diff: "%i, "{:e}".format(mag))
        pos_differences.append(mag)

    for i in range(len(neg_calced_pieces)):
        piece = neg_calced_pieces[i]
        mag = np.abs(np.sum(piece - calced_mode))
        print("Neg Piece %i diff: "%i, "{:e}".format(mag))
        neg_differences.append(mag)

    print("Minimum Pos Difference: ", "{:e}".format(min(pos_differences)))
    print("Minimum Neg Difference: ", "{:e}".format(min(neg_differences)))


def t_orb(center_on_edge=False):
    array_shape = (100,100)
    base_canvas = jnp.zeros(array_shape)
    corner = jnp.array([100,100])
    diam = 10
    if center_on_edge:
        corner = corner - int(diam/2)
    orb_coords = genR.get_orb_pts(array_shape, corner, diam)
    xs_orb, ys_orb = zip(*orb_coords)
    base_canvas = base_canvas.at[xs_orb, ys_orb].set(1)
    plt.imshow(base_canvas, cmap='hot', interpolation='nearest')
    plt.show()

if __name__ == "__main__":
    #t_dPC_5()
    #t_dTV_1()
    #t_mode_is_avg(using_helicity=True, print_it=True)
    #visualize_fft()
    #print(clustering_to_cachable_labels([0,0,1,1]))
    #print(clustering_to_cachable_labels([1,1,0,0]))
    #t5_clustering_caches()
    #t_reverse_str(arr=gen_random_uni_arr(my_shape=(15,15,3)))
    #arr_of_imgs = jnp.array([[[1,0],[0,1]], [[0,2],[-1,0]], [[3,1],[1,3]], [[3,1],[1,3]], [[1,0],[0,1]]])
    #t_vmaped_mat_construction(arr_of_imgs, gamma=0.5)
    #t_topo_holo(topo_num=1, pathtype='f')
    #t_generate_holo_calculated_mode(my_mode=' 1-1', helicity=1, useAvg=True)
    #t_generate_holo_calculated_mode(my_mode=' 1-1', helicity=1, useAvg=False)
    t_orb(center_on_edge=True)