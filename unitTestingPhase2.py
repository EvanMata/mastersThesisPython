import os
import sys
import jax

import numpy as np
import jax.numpy as jnp
import optimization as clu
import openRawData as opn
#import construct_real_space as con
import genRealisticData as genR
import pathlib_variable_names as var_names
import matplotlib.pyplot as plt

from jax import random
from pathlib import Path
from itertools import combinations, product

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
    # Should be NOT SAME
    l1 = [0,0,1,1,]
    l2 = [1,1,0,1]
    nl1 = clu.clustering_to_cachable_labels(l1, 2)
    nl2 = clu.clustering_to_cachable_labels(l2, 2)

    for i in range(len(nl1)):
        if nl1[i] != nl2[i]:
            print("NOT the same clustering")
            return
    print("SAME clustering")


def t2_clustering_caches():
    # Should be NOT SAME, order of points matters
    l1 = [0,0,1,1,2]
    l2 = [1,1,0,2,2]
    nl1 = clu.clustering_to_cachable_labels(l1, 3)
    nl2 = clu.clustering_to_cachable_labels(l2, 3)

    for i in range(len(nl1)):
        if nl1[i] != nl2[i]:
            print("NOT the same clustering")
            return
    print("SAME clustering")


def t3_clustering_caches():
    # Should be SAME, order of points matters
    l1 = [0,0,1,1,2,3,4]
    l2 = [1,1,0,0,2,4,3]
    nl1 = clu.clustering_to_cachable_labels(l1, 5)
    nl2 = clu.clustering_to_cachable_labels(l2, 5)

    for i in range(len(nl1)):
        if nl1[i] != nl2[i]:
            print("NOT the same clustering")
            return
    print("SAME clustering")


def t4_clustering_caches():
    # Should be NOT SAME, order of points matters
    l1 = [0,0,1,1,2,3,4,5,5,5]
    l2 = [5,1,5,1,0,0,2,4,3,5]
    nl1 = clu.clustering_to_cachable_labels(l1, 6)
    nl2 = clu.clustering_to_cachable_labels(l2, 6)

    for i in range(len(nl1)):
        if nl1[i] != nl2[i]:
            print("NOT the same clustering")
            return
    print("SAME clustering")


def t5_clustering_caches():
    # Should be YES?, order of points matters
    l1 = [0,0,1,1,5,5,5,2,3,4]
    l2 = [1,1,0,0,5,5,5,3,4,2]
    nl1 = clu.clustering_to_cachable_labels(l1, 6)
    nl2 = clu.clustering_to_cachable_labels(l2, 6)

    for i in range(len(nl1)):
        if nl1[i] != nl2[i]:
            print("NOT the same clustering")
            return
    print("SAME clustering")


def t6_clustering_caches():
    # Should be YES 
    l1 = [0,0,5,5,5,2,3,4]
    l2 = [1,1,5,5,5,3,4,2]
    nl1 = clu.clustering_to_cachable_labels(l1, 6)
    nl2 = clu.clustering_to_cachable_labels(l2, 6)

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


def t_vmapped_2():
    def f_w_o_arrs(a, b, vec, c=True):
        return not a, jnp.sum(vec)*b, not c
    v_f = jax.vmap(f_w_o_arrs, (None, 0, 0), 0)
    my_bs = jnp.array([1,1,2,3])
    my_vs = jnp.array([[1,1,1],[2,2,2],[1,1,1],[3,3,3]])
    m_v_f = v_f(True, my_bs, my_vs)
    print(m_v_f)


def t_vmapped_3():
    def split_f(a, which):
        if which:
            return 5*a
        else:
            return 10*a
    v_f = jax.vmap(split_f, (0, 0), 0)
    my_truths = jnp.array([True,True,False])
    my_as = jnp.array([1,2,2])
    m_v_f = v_f(my_as, my_truths)
    print(m_v_f)


def comp_vmap_3():
    def f1(a):
        return 5*a
    def f2(a):
        return 10*a
    my_as = jnp.array([1,2,2,3])
    m_v_f = jnp.zeros((3,))
    my_truths = jnp.array([True,True,False,False])
    f1_as = my_as[my_truths]
    f2_as = my_as[~my_truths]
    f1s = jax.vmap(f1, (0))(f1_as)
    f2s = jax.vmap(f2, (0))(f2_as)
    print(f1s)
    print(f2s)
    m_v_f = jnp.zeros((len(my_as), ))
    m_v_f = m_v_f.at[jnp.where(my_truths)].set(f1s)
    m_v_f = m_v_f.at[jnp.where(~my_truths)].set(f2s)
    print(m_v_f)


def comp_vmap_4():
    def f1(a):
        return 5*a, 1*a
    def f2(a):
        return 10*a, 20*a
    my_as = jnp.array([1,2,2,3])
    m_v_f = jnp.zeros((3,))
    my_truths = jnp.array([True,True,False,False])
    f1_as = my_as[my_truths]
    f2_as = my_as[~my_truths]
    f1s, f1sb = jax.vmap(f1, (0), (0,0))(f1_as)
    f2s, f2sb = jax.vmap(f2, (0), (0,0))(f2_as)
    m_v_f = jnp.zeros((len(my_as), ))
    m_v_f_b = jnp.zeros((len(my_as), ))
    m_v_f = m_v_f.at[jnp.where(my_truths)].set(f1s)
    m_v_f = m_v_f.at[jnp.where(~my_truths)].set(f2s)
    m_v_f_b = m_v_f_b.at[jnp.where(my_truths)].set(f1sb)
    m_v_f_b = m_v_f_b.at[jnp.where(~my_truths)].set(f2sb)
    print(m_v_f)
    print(m_v_f_b)


def t_vmap_region():
    x_mins = jnp.array([1,2,3])
    x_maxs = jnp.array([4,5,5])
    y_mins = jnp.array([1,1,1])
    y_maxs = jnp.array([1,2,2])
    def create_region(x_min, x_max, y_min, y_max):
        xs = jnp.arange(x_min, x_max, 1)
        ys = jnp.arange(y_min, y_max, 1)
        region = product(xs, ys)
        return region

    vmaped_region = jax.vmap(create_region, (0,0,0,0), 0)
    regions = vmaped_region(x_mins, x_maxs, y_mins, y_maxs)


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


def t_pdist_ok():
    c_key = jax.random.PRNGKey(0)
    n_key, adj_mat, G = genR.gen_graph(c_key, n_states=20, p=.1, self_loops=False)
    states = range(len(adj_mat))
    for state in states:
        nxt = genR.get_next_state(c_key, adj_mat, state)
        print(nxt)


def affininty_matrix_ex(n_arrays=10, img_size=5, key=jax.random.PRNGKey(0), gamma=jnp.array([0.5])):
    arr_of_imgs = jax.random.normal(jax.random.PRNGKey(0), (n_arrays, img_size, img_size))
    arr_of_indices = jnp.arange(n_arrays)
    inds_1, inds_2 = zip(*combinations(arr_of_indices, 2))
    v_cPA = jax.vmap(calcPairAffinity2, (0, 0, None, None), 0)
    affinities = v_cPA(jnp.array(inds_1), jnp.array(inds_2), arr_of_imgs, gamma)
    print()
    print(jax.make_jaxpr(v_cPA)(jnp.array(inds_1), jnp.array(inds_2), arr_of_imgs, gamma))
    
    affinities = affinities.reshape(-1)
    
    arr = jnp.zeros((n_arrays, n_arrays), dtype=jnp.float16)
    arr = arr.at[jnp.triu_indices(arr.shape[0], k=1)].set(affinities)
    arr = arr + arr.T
    arr = arr + jnp.identity(n_arrays, dtype=jnp.float16)
    
    return arr


def calcPairAffinity2(ind1, ind2, imgs, gamma):
    #Returns a jnp array of 1 float, jnp.sum adds all elements together
    image1, image2 = imgs[ind1], imgs[ind2]
    diff = jnp.sum(jnp.abs(image1 - image2))  
    normed_diff = diff / image1.size
    val = jnp.exp(-gamma*normed_diff)
    val = val.astype(jnp.float16)
    return val








if __name__ == "__main__":
    #t_dPC_5()
    #t_dTV_1()
    #t_mode_is_avg(using_helicity=True, print_it=True)
    #visualize_fft()
    #print(clustering_to_cachable_labels([0,0,1,1]))
    #print(clustering_to_cachable_labels([1,1,0,0]))
    #t6_clustering_caches()
    #t_reverse_str(arr=gen_random_uni_arr(my_shape=(15,15,3)))
    #arr_of_imgs = jnp.array([[[1,0],[0,1]], [[0,2],[-1,0]], [[3,1],[1,3]], [[3,1],[1,3]], [[1,0],[0,1]]])
    #t_vmaped_mat_construction(arr_of_imgs, gamma=0.5)
    #t_topo_holo(topo_num=1, pathtype='f')
    #t_generate_holo_calculated_mode(my_mode=' 1-1', helicity=1, useAvg=True)
    #t_generate_holo_calculated_mode(my_mode=' 1-1', helicity=1, useAvg=False)
    #t_orb(center_on_edge=True)
    #t_vmapped_2()
    #t_vmapped_3()
    #comp_vmap_4()
    #t_vmap_region()
    #t_pdist_ok()
    n_arrays=10
    img_size=5
    key=jax.random.PRNGKey(0)
    data=jax.random.normal(key, (n_arrays, img_size, img_size))
    print(affininty_matrix_ex(n_arrays=10000, img_size=100))
    #check_aff()