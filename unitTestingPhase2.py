import os
import sys

import numpy as np
import clustering as clu
import openRawData as opn
import pathlib_variable_names as var_names

from pathlib import Path

sys.path.append('./python_provided_code/')
from fth_reconstruction import reconstructCDI as my_fft


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
    Check whether the avg of all the holos in mode 1-1 is actually 
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
        holo_arr = opn.openBaseHolo(holoNumber, pathtype='s', proced=False, mask=False)
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

    avged_holo_arr = holo_arr / num_holos
    return avged_holo_arr


def visualize_fft(my_mode=' 1-1', using_helicity=True):
    avg_holo_arr = t_mode_is_avg(my_mode=my_mode, using_helicity=using_helicity, print_it=False)
    real_sp_maybe = my_fft(avg_holo_arr)
    opn.heatMapImg(real_sp_maybe)


if __name__ == "__main__":
    #t_dPC_5()
    #t_dTV_1()
    t_mode_is_avg(using_helicity=True, print_it=True)
    #visualize_fft()