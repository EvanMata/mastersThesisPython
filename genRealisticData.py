import jax 
import time
import pickle
import numpy as np
import jax.numpy as jnp
import networkx as nx
import matplotlib.pyplot as plt

from itertools import product
from scipy.spatial.distance import cdist
from sinkhorn_knopp import sinkhorn_knopp as skp

import pathlib_variable_names as my_vars

MY_KEY = jax.random.PRNGKey(0)
DIAM = 5 #Diameter, in pixels, of region an orb can take 1 step in. =1 means unmoving

jnp.set_printoptions(precision=3)

####################
# Noise Generation #
####################


def gen_low_freq_noise_quarter(array_shape, c_key, l_bd=-1, u_bd=1, 
                       cutoff=0.05, r_cutoff=False, cutoff_bds=[0.02, 0.02]):
    """
    Generates a 2d low frequency noise setup, where it keeps a quarter circle of the 
    lowest frequencies.

    Inputs:
    --------
        array_shape (tup of 2 ints) : x and y size of the array to create
        c_key (jnp array) : The current key. Used by JAX to generate random numbers 
                    and ensure results reproducible
        l_bd (float) : infemum of generated data 
        u_bd (float) : Supremum of generated data 
        cutoff (float) : Should be in (0,1). Keep the bottom cutoff percentile of values
        r_cutoff (bool) : Whether cutoff should be always be the same value, or if it should be 
                        Chosen uniformly between cutoff - min(cutoff_bds), cutoff + max(cutoff_bds)
        cutoff_bfs (lst) : Only used if r_cutoff = True. Then cutoff is uniform on 
                            cutoff - min(cutoff_bds) & cutoff - max(cutoff_bds)

    Return:
    --------
        noise (jnp array) : Array with the noise
        n_key (jnp array) : new key for generating new random numbers
    """

    n_key, subkey = jax.random.split(c_key)
    rl_space = jax.random.uniform(key=subkey, shape=array_shape, minval=l_bd, maxval=u_bd)
    f_space = jnp.fft.fft2(rl_space)

    if r_cutoff:
        n_key, subsubkey = jax.random.split(n_key)
        min_cutoff = cutoff - min(cutoff_bds)
        max_cutoff = cutoff + max(cutoff_bds)
        cutoff_rad = float(jax.random.uniform(key=subsubkey, minval=min_cutoff, maxval=max_cutoff))
    else:
        cutoff_rad = cutoff

    x_cen = int(array_shape[0]/2)
    y_cen = int(array_shape[1]/2)
    tot_rad2 = array_shape[0]**2 + array_shape[1]**2
    potential_pairs = jnp.array(list(product(range(array_shape[0]), range(array_shape[1]))))
    elipse_vals = jnp.array([float((v[0]**2) + (v[1]**2))
                                for v in potential_pairs])
    valid_indices = potential_pairs[jnp.where(elipse_vals**0.5 <= cutoff_rad*array_shape[0])]
    inds_1, inds_2 = zip(*valid_indices)
    f_noise = jnp.zeros(array_shape)
    f_noise = f_noise.at[inds_1, inds_2].set(f_space[inds_1, inds_2])
    
    noise = jnp.real(jnp.fft.ifft2(f_noise))
    f_noise = jnp.real(f_noise)

    return noise, f_noise, n_key


def gen_low_freq_noise_rot(array_shape, c_key, l_bd=-1, u_bd=1, 
                       cutoff=0.05, r_cutoff=False, cutoff_bds=[0.02, 0.02]):
    """
    Generates a 2d low frequency noise setup, where it keeps a circle of the 
    lowest frequencies.

    Inputs:
    --------
        array_shape (tup of 2 ints) : x and y size of the array to create
        c_key (jnp array) : The current key. Used by JAX to generate random numbers 
                    and ensure results reproducible
        l_bd (float) : infemum of generated data 
        u_bd (float) : Supremum of generated data 
        cutoff (float) : Should be in (0,1). Keep the bottom cutoff percentile of values
        r_cutoff (bool) : Whether cutoff should be always be the same value, or if it should be 
                        Chosen uniformly between cutoff - min(cutoff_bds), cutoff + max(cutoff_bds)
        cutoff_bfs (lst) : Only used if r_cutoff = True. Then cutoff is uniform on 
                            cutoff - min(cutoff_bds) & cutoff - max(cutoff_bds)

    Return:
    --------
        noise (jnp array) : Array with the noise
        n_key (jnp array) : new key for generating new random numbers
    """

    n_key, subkey = jax.random.split(c_key)
    rl_space = jax.random.uniform(key=subkey, shape=array_shape, minval=l_bd, maxval=u_bd)
    f_space = jnp.fft.fft2(rl_space)

    if r_cutoff:
        n_key, subsubkey = jax.random.split(n_key)
        min_cutoff = cutoff - min(cutoff_bds)
        max_cutoff = cutoff + max(cutoff_bds)
        cutoff_rad = float(jax.random.uniform(key=subsubkey, minval=min_cutoff, maxval=max_cutoff))
    else:
        cutoff_rad = cutoff

    x_cen = int(array_shape[0]/2)
    y_cen = int(array_shape[1]/2)
    array_rad2 = (x_cen**2 + y_cen**2)
    potential_pairs = jnp.array(list(product(range(array_shape[0]), range(array_shape[1]))))
    relative_cen_x = cutoff_rad*array_shape[0]
    relative_cen_y = cutoff_rad*array_shape[1]
    #elipse_vals = jnp.array([float(((v[0] - x_cen)**2)/x_cen + ((v[1] - y_cen)**2)/y_cen)
    #                            for v in potential_pairs])
    radi2 = jnp.array([float(((v[0] - relative_cen_x)**2) + ((v[1] - relative_cen_y)**2))
                                for v in potential_pairs])
    valid_indices = potential_pairs[jnp.where(radi2**0.5 <= cutoff_rad*array_shape[0])]
    inds_1, inds_2 = zip(*valid_indices)
    f_noise = jnp.zeros(array_shape)
    f_noise = f_noise.at[inds_1, inds_2].set(f_space[inds_1, inds_2])
    
    noise = jnp.real(jnp.fft.ifft2(f_noise))
    f_noise = jnp.real(f_noise)

    return noise, f_noise, n_key


def gen_low_freq_noise(array_shape, c_key, l_bd=-1, u_bd=1, 
                       cutoff=0.05, r_cutoff=False, cutoff_bds=[0.02, 0.02]):
    """
    Generates a 2d low frequency noise setup, ideally rotationally symmetric (NOT WORKING)

    Inputs:
    --------
        array_shape (tup of 2 ints) : x and y size of the array to create
        c_key (jnp array) : The current key. Used by JAX to generate random numbers 
                    and ensure results reproducible
        l_bd (float) : infemum of generated data 
        u_bd (float) : Supremum of generated data 
        cutoff (float) : Should be in (0,1). Keep the bottom cutoff percentile of values
        r_cutoff (bool) : Whether cutoff should be always be the same value, or if it should be 
                        Chosen uniformly between cutoff - min(cutoff_bds), cutoff + max(cutoff_bds)
        cutoff_bfs (lst) : Only used if r_cutoff = True. Then cutoff is uniform on 
                            cutoff - min(cutoff_bds) & cutoff - max(cutoff_bds)

    Return:
    --------
        noise (jnp array) : Array with the noise
        n_key (jnp array) : new key for generating new random numbers
    """

    n_key, subkey = jax.random.split(c_key)
    rl_space = jax.random.uniform(key=subkey, shape=array_shape, minval=l_bd, maxval=u_bd)
    f_space = jnp.fft.fft2(rl_space)

    if r_cutoff:
        n_key, subsubkey = jax.random.split(n_key)
        min_cutoff = cutoff - min(cutoff_bds)
        max_cutoff = cutoff + max(cutoff_bds)
        cutoff_rad = float(jax.random.uniform(key=subsubkey, minval=min_cutoff, maxval=max_cutoff))
    else:
        cutoff_rad = cutoff

    x_cutoff = int(array_shape[0]*cutoff_rad)
    y_cutoff = int(array_shape[1]*cutoff_rad)

    f_space = f_space.at[x_cutoff:array_shape[0]-x_cutoff,:].set(0)
    f_space = f_space.at[:,y_cutoff:array_shape[1]-y_cutoff].set(0)
    
    noise = jnp.real(jnp.fft.ifft2(f_space))
    f_space = jnp.real(f_space)

    return noise, f_space, n_key


####################
# State Generation #
####################


def gen_states(c_key, n_states, array_shape, lb_orbs, ub_orbs, fix_avg=20, fix_stv=4, 
               fix_stv_stv=2, lb_size=6, ub_size=10, region_d=5):
    """
    Generates my states and transition probabilities for them. Currently assumes the 
    same number of orbs for each state, ub_orbs

    Inputs:
    --------
        c_key (jnp array) : Jax array/Current Key for generating random numbers
        n_states (int) : number of distinct states 
        array_shape (tup) : size of the canvas, x by y pixels
        lb_orbs (int) : lower bound on the number of orbs on the canvas as once
        ub_orbs (int) : upper bound on the number of orbs on the canvas as once
        fix_avg (int) : The expected average of the number of itterations a state stays fixed
        fix_stv (float) : The Standard Deviation of the expected number of itterations 
                         a state stays fixed
        fix_stv_stv (float) : Standard Deviation of the standard derivations of 
                              the expected number of itterations a state stays fixed
        lb_size (int) : Minimum diameter of an orb
        ub_size (int) : Maximum diameter of an orb
        region_d (int) : Diameter of a region for an orb to be centered in.

    Returns:
    --------
        n_key (jnp array) : new key for generating new random numbers
        states_i (dict) : Dict of State Number: Expected Image of State (array)
        states_c (dict) : Dict of State Number: Dict of Orb Numbers: tup of:
                          - valid region/destination locations for The given orb 
                          (eg where it can move to when in the state) 
                          - orb diam
        states_s (dict) : Dict of State Number: 
                            [Expected Number of itterations to stay in state, before 
                             identifying new state to transition to,
                             Stv of Number of itterations to stay in state]
    """
    n_key, subkey = jax.random.split(c_key)
    region_rad = int(region_d/2)
    x_width, y_height = array_shape[0], array_shape[1]
    valid_corners = jnp.array(list(product(range(x_width), range(y_height))))

    states_i = dict()
    states_c = dict()
    states_f = dict()

    # Setup the distributions of how long each state stays fixed
    n_key, subkey = jax.random.split(n_key)
    state_duration_avgs = fix_stv*(jax.random.normal(subkey, shape=(n_states,))) + fix_avg
    min_state_duration = max([0, fix_avg - fix_stv*3])
    state_duration_avgs = jnp.clip(state_duration_avgs, a_min=min_state_duration)

    n_key, subkey = jax.random.split(n_key)
    state_duration_variances = fix_stv_stv*(jax.random.normal(subkey, shape=(n_states,))) + fix_stv
    state_duration_variances = jnp.clip(state_duration_variances, a_min=0.1)
    
    n_orbs = ub_orbs #ADJUST THIS LATER
    n_key, subkey = jax.random.split(n_key)
    orb_diams = jax.random.randint(key=subkey, shape=(n_orbs,), 
                                    minval=lb_size, maxval=ub_size)
    
    for state in range(n_states):
        orb_num_to_corner = dict()
        visual = jnp.zeros(array_shape)

        for orb in range(len(orb_diams)):
            orb_diam = orb_diams[orb]
            orb_valid_corners = valid_corners.copy()
            orb_valid_corners = orb_valid_corners - int(orb_diam/2)
            n_key, subkey = jax.random.split(n_key)
            corner_index = jax.random.randint(key=subkey, shape=(1,), 
                                            minval=0, maxval=orb_valid_corners.shape[0])
            corner = orb_valid_corners[corner_index][0]
            region = get_region(array_shape, corner, region_d, orb_diam)
            orb_num_to_corner[orb] = (region, orb_diam) #Make region tuple?
            img_base_weight = 1/len(region)
            for region_c in region:
                orb_pts = get_orb_pts(array_shape, region_c, orb_diam)
                orb_xs, orb_ys = zip(*orb_pts)
                visual = visual.at[orb_xs, orb_ys].add(img_base_weight)

        visual = jnp.clip(visual, a_max=1)

        states_i[state] = visual
        states_c[state] = orb_num_to_corner
        state_dur_avg, state_dur_stv = state_duration_avgs[state], state_duration_variances[state]
        states_f[state] = (state_dur_avg, state_dur_stv)


    return n_key, states_i, states_c, states_f


def get_region(array_shape, corner, region_d, orb_diam, sq=True):
    """
    Given the corner of the region for a given orb, calculate the relevant region
    of points that are ok for the corner to go to. 
    Eg if square product(x_cor:x_cor + diam, y_cor:y_cor + diam) but within the grid.

    Inputs:
    --------
        array_shape (tup) : shape of the canvas 
        corner (jnp.array) : (x,y) pair indicating where the lowest x, lowest y valued 
                        corner a box around the orb would be
        region_d (int) : size of region's diameter 
        orb_diam (int) : Diameter of the orb who's region it will be.
        sq (bool) : If true, makes regions that are squares, 
                    If false NOT IMPLEMENTED, make plus or circle shapes regions
    Returns:
        region (jnp array) : jax array of [x,y] coords in region
    """
    #orb_rad = int(orb_diam/2)
    orb_rad = orb_diam/2
    orb_rad = orb_rad.astype(int)
    x_cor = corner[0]
    y_cor = corner[1]
    canvas_w = array_shape[0]
    canvas_h = array_shape[1]
    if sq:
        min_x = jnp.max(jnp.array([-orb_rad, x_cor]))
        max_x = jnp.min(jnp.array([x_cor + region_d, canvas_w - orb_rad])) + 1
        min_y = jnp.max(jnp.array([-orb_rad, y_cor]))
        max_y = jnp.min(jnp.array([y_cor + region_d, canvas_h - orb_rad])) + 1
        """
        if min_x != max_x:
            xs = range(min_x, max_x) #min=max possi currently; if on border
        else:
            xs = range(min_x, min_x+1)
        if min_y != max_y:
            ys = range(min_y, max_y)
        else:
            ys = range(min_y, min_y+1)
        xs = range(min_x, max_x)
        ys = range(min_y, max_y)
        """
        xs = jnp.arange(min_x, max_x, 1)
        ys = jnp.arange(min_y, max_y, 1)
        region = list(product(xs, ys))
    else:
        print("NOT IMPLEMENTED")

    region = jnp.array(region)
    return region


def get_orb_pts(array_shape, corner, orb_diam): 
    """
    Gets all indices the orb is currently filling

    Inputs:
    --------
        array_shape (tup) : shape of the canvas 
        corner (jnp.array) : (x,y) pair indicating where the lowest x, lowest y valued 
                        corner a box around the orb would be
        orb_diam (int) : The diameter of the orb in pixels
    
    Returns:
    --------
        current_coords (jnp array) : pairs of points where the orb occupies
    """
    orb_rad = int(orb_diam/2)
    center = corner + orb_rad
    x_cen = center[0]
    y_cen = center[1]
    canvas_w = array_shape[0]
    canvas_h = array_shape[1]

    min_x = max([0, x_cen - orb_rad - 1])
    max_x = min([x_cen + orb_rad + 1, canvas_w])
    min_y = max([0, y_cen - orb_rad - 1])
    max_y = min([y_cen + orb_rad + 1, canvas_h])
    xs = range(min_x, max_x)
    ys = range(min_y, max_y)
    potential_coords = jnp.array(list(product(xs, ys)))
    coords_dists = jnp.array([float(((v[0]-x_cen)**2 + (v[1]- y_cen)**2)**0.5) 
                              for v in potential_coords])
    current_coords = potential_coords[jnp.where(coords_dists <= orb_diam/2)]
    return current_coords


########################
# Graph Creation tools #
########################


def gen_graph(c_key, n_states, p, self_loops=True):
    """
    Generates a random graph that is fully connected. 
    Creates a random fully connected graph with n-1 vertices, and a random erdas 
    reini graph, then combines them (logical or) and sets all their weights randomly. 
    
    Inputs:
    --------
        c_key (jnp array) : Jax array/Current Key for generating random numbers
        n_states (int) : number of distinct states 
        p (float in (0,1)) : Probability of adding an edge for the erdas-reini construction
        self_loops (bool) : Whether to include the possibility of a state transitioning 
                            back to itself. 
    Returns:
    --------
        n_key (jnp array) : new key for generating new random numbers
        adj_mat (jnp array) : doubly stochastic matrix thats ~mostly symetric 
                                (up to ~3 orders of magnitude)
        G (networkx Graph) : A networkx graph object from the corresponding 
                             adjacency matrix.
    
    """
    adj_mat = jnp.zeros((n_states, n_states))
    n_key, subkey = jax.random.split(c_key)
    er_graph = nx.erdos_renyi_graph(n_states, p, seed=int(subkey[0]))
    er_graph = jnp.array(nx.to_numpy_array(er_graph))
    fully_connected, n_key = random_fully_connected(n_states, n_key)
    connected_indices = jnp.logical_or(fully_connected, er_graph)

    if self_loops:
        n_key, subkey = jax.random.split(n_key)
        diag_inds = jnp.arange(n_states)
        diag_vals = jax.random.uniform(subkey, shape=(n_states, ), maxval=0.25)    
    
    connected_indices_tri1 = jnp.triu(connected_indices, k=1)
    num_edges_tri1 = jnp.sum(connected_indices_tri1)

    n_key, subkey = jax.random.split(n_key)
    pre_adjusted_edge_weights = jax.random.uniform(subkey, shape=(num_edges_tri1, ))
    inds_for_wieghts = jnp.where(connected_indices_tri1 == 1)
    adj_mat = adj_mat.at[inds_for_wieghts].set(pre_adjusted_edge_weights)
    adj_mat = adj_mat + adj_mat.T
    if self_loops:
        adj_mat = adj_mat.at[diag_inds, diag_inds].set(diag_vals)

    sk = skp.SinkhornKnopp()
    adj_mat = sk.fit(adj_mat)
    adj_mat = jnp.array(adj_mat)
    G = nx.from_numpy_array(adj_mat)
    return n_key, adj_mat, G


def random_fully_connected(n_states, c_key):
    """
    Generates a random fully connected graph w. n-1 edges

    Inputs:
    --------
        n_states (int) : number of distinct states 
        c_key (jnp array) : The current key. Used by JAX to generate random numbers 
                    and ensure results reproducible
    Returns:
    --------
        fully_connected (jnp array) : Array of 1's and 0's, w. n-1 1's
        n_key (jnp array) : new key for generating new random numbers
    """
    fully_connected = jnp.zeros((n_states, n_states))
    i, j = jnp.indices((n_states, n_states))
    fully_connected = fully_connected.at[i==j-1].set(1) #Set superdiag = 1
    fully_connected, n_key = permute_matrix(fully_connected, c_key) #random permute
    return fully_connected, n_key


def gen_perm(n_states, c_key):
    """
    Generates a single random permutation matrix
    """
    n_key, subkey = jax.random.split(c_key)
    id = jnp.identity(n_states)
    perm = jax.random.permutation(subkey, id)
    return perm, n_key


def permute_matrix(matrix, c_key):
    """
    Permutates a matrix with a random permutation
    """
    n_states = matrix.shape[0]
    perm, n_key = gen_perm(n_states, c_key)
    permuted_m = jnp.matmul(perm.T, jnp.matmul(matrix, perm))
    return permuted_m, n_key


def vis_graph(G):
    edge_labels = dict([((u,v,), f"{d['weight']:.2f}") for u,v,d in G.edges(data=True)])
    pos = nx.spring_layout(G, seed=10)
    #pos = G.layout('dot')
    nx.draw_networkx(G, pos)
    nx.draw_networkx_edge_labels(G, pos, edge_labels, label_pos=0.5)
    #D = nx.drawing.nx_agraph.to_agraph(G)
    #D.layout('dot')
    #D.draw('Graph.eps')
    
    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def vis_graph2(adj_mat):
    from netgraph import Graph
    sources, targets = np.where(adj_mat)
    weights = adj_mat[sources, targets]
    #weights = [f"{d['weight']:.2f}" for w in weights]
    weights = [f"{w:.2f}" for w in weights]
    edges = list(zip(sources, targets))
    edge_labels = dict(zip(edges, weights))

    fig, ax = plt.subplots()
    Graph(edges, edge_labels=edge_labels, edge_label_position=0.66, arrows=True, ax=ax)
    plt.show()


##############
# Move 1 Orb #
##############


def prob_distro(destination, array_shape, corner, orb_diam, epsi, sq=True):
    """
    Calculates the probability of the given orb center transitioning to each adjacent space.
    Sets prob to go to a closer sq as epsi/num closer places 

    Inputs:
    -------
        epsi (float in [.5,1]) : Probability of moving towards the destination 
                                 (in aggregate)
        corner (jnp.array) : (x,y) pair indicating where the lowest x, lowest y valued 
                        corner a box around the orb would be
        orb_diam (int) : Diameter of the orb who's region it will be.
        array_shape (tup) : shape of the canvas 
        destination (jnp array) : states_c[state][orb] value, an array of tuples of 
                    corner locations that the orb is heading towards
        sq (bool) : If true, makes regions that are squares, 
                    If false NOT IMPLEMENTED, make plus or circle shapes regions
    
    Returns:
    --------
        valid_transitions (jnp array) : array of tuples indicating valid 
                                        corner's to transition to, 
        trans_probs (jnp array) : Array of normalize transition probabilies corresponding 
                                  to the valid transitions (trans to valid_transitions[i] 
                                                     has prob trans_probs[i])
    """

    #Needs Corner and Epsi and dest others pass to get region.

    diam = DIAM #Diameter of region where orb can travel to.
    
    
    valid_transitions = get_region(array_shape, corner - 2, diam, orb_diam, sq) #For corners, global
    trans_probs = jnp.zeros(valid_transitions.shape[0],)
    
    print(corner - 2, diam, orb_diam)
    print("Valid Transitions: ", valid_transitions)
    
    dists = cdist( valid_transitions, destination, metric='euclidean' )
    min_l2s = jnp.min(dists, axis=1)
    base_dist = jnp.min(cdist( [corner], destination, metric='euclidean' ))
    move_towards_indices = jnp.where(min_l2s < base_dist)
    move_away_indices = jnp.where(min_l2s >= base_dist)
    num_mv_towards = move_towards_indices[0].size
    normed_mv_towards_prob = float(epsi / num_mv_towards)
    normed_mv_away_prob = float((1-epsi) / (valid_transitions.size - num_mv_towards))
    trans_probs = trans_probs.at[move_towards_indices[0]].set(normed_mv_towards_prob)
    trans_probs = trans_probs.at[move_away_indices[0]].set(normed_mv_away_prob)

    return valid_transitions, trans_probs


def orb_step_toward_st(c_key, destination, array_shape, corner, orb_diam, epsi, sq=True):
    """
    Moves 1 orb 1 step closer to a state. NOT for when already in a state.

    Inputs:
    --------
        c_key (jnp array) : Jax array/Current Key for generating random numbers
        *args (lst) : Args for prob_distro, [epsi, corner, orb_diam, 
                                             array_shape, destination, sq] 
    Returns:
    --------
        n_key (jnp array) : new key for generating new random numbers
        trans_to (jnp array) : array of x,y value for new corner
    """
    n_key, subkey = jax.random.split(c_key)
    valid_transitions, trans_probs = prob_distro(destination, array_shape, 
                                                 corner, orb_diam, epsi, sq)
    trans_to = jax.random.choice(subkey, valid_transitions, p=trans_probs)
    return n_key, trans_to


def orb_step_arived_st(c_key, destination, array_shape, corner, orb_diam, sq=True):
    """
    Picks where the orb moves if it has already arrived at its destination
    Should update this if destinations get large, so that jumps aren't more than one step.

    Inputs:
    --------
        c_key (jnp array) : Jax array/Current Key for generating random numbers
        destination (jnp array) : states_c[state][orb][0] value, an array of tuples of 
                    corner locations that the orb is heading towards
        *args (lst) : args for get_region, eg [array_shape, corner, DIAM, orb_diam, sq]
    Returns:
    -------
        n_key (jnp array) : new key for generating new random numbers
        trans_to (jnp array) : (x,y) corner to transit to.
    """
    diam = DIAM
    valid_transitions = get_region(array_shape, corner, diam, orb_diam, sq)
    valid_trans = set([(loc[0].astype(int),loc[1].astype(int)) for loc in valid_transitions])
    dests = set([(loc[0].astype(int),loc[1].astype(int)) for loc in destination])
    valid_dests = list(dests.intersection(valid_trans)) #Anything within travel dist and in dest
    n_key, subkey = jax.random.split(c_key)
    trans_to_index = jax.random.randint(subkey, shape=(1,), minval=0, maxval=len(valid_dests))
    trans_to = destination[trans_to_index][0]
    return n_key, trans_to


def one_orb_one_st(c_key, corner, destination, array_shape, orb_diam, epsi, arrived_prev, 
                   sq=True, jitv=False):
    """
    Moves one orb one step

    Inputs:
    --------
        c_key (jnp array) : Jax array/Current Key for generating random numbers
        corner (jnp.array) : (x,y) pair indicating where the lowest x, lowest y valued 
                        corner a box around the orb would be
        destination (jnp array) : states_c[state][orb] value, an array of tuples of 
                    corner locations that the orb is heading towards
        array_shape (tup) : shape of the canvas 
        orb_diam (int) : Diameter of the orb who's region it will be.
        epsi (float in [.5,1]) : Probability of moving towards the destination 
                                 (in aggregate)
        arrived_prev (bool) : Whether the given orb's corner previously arrived 
                              at its destination
        sq (bool) : If true, makes regions that are squares, 
                    If false NOT IMPLEMENTED, make plus or circle shapes regions
    Returns:
    --------
        n_key (jnp array) : new key for generating new random numbers
        trans_to (jnp array) : (x,y) corner to transit to.
        arrived_to (bool) : Whether the orb arrived at its destination.
    """
    if jitv:
        dests = set([(loc[0].astype(int),loc[1].astype(int)) for loc in destination])
    else:
        dests = to_tups(destination)
    if arrived_prev:
        n_key, trans_to = orb_step_arived_st(c_key, destination, array_shape, 
                                             corner, orb_diam, sq)
    else:
        n_key, trans_to = orb_step_toward_st(c_key, destination, array_shape, 
                                             corner, orb_diam, epsi, sq)
    if jitv:
        trans_to_t = (trans_to[0].astype(int), trans_to[1].astype(int))
    else:
        trans_to_t = (int(trans_to[0]), int(trans_to[1]))
    arrived_to = trans_to_t in dests
    return n_key, trans_to, arrived_to


def all_orbs_one_st(c_key, corners, states_c, target_state, array_shape, epsi, arrived_prevs, 
                    sq=True):
    """
    Moves all orbs one step, vmapped for hopefully speed.
    INCOMPLETE - Would need to adjust get_region to be jit-able 
                    and all subsequent functions which rely on it to use this

    Inputs:
    --------
        c_key (jnp array) : Jax array/Current Key for generating random numbers
        corners (jnp arr) : jnp array of (x,y) pairs indicating where the 
                            lowest x, lowest y valued corner a box around the 
                            orb at index i would be
        states_c (dict) : Dict of State Number: Dict of Orb Numbers: tup of:
                          - valid region/destination locations for The given orb 
                          (eg where it can move to when in the state) 
                          - orb diam
        target_state (int) : State number that you're progressing towards, a key for states_c
        array_shape (tup) : shape of the canvas 
        epsi (float in [.5,1]) : Probability of an orb moving towards its destination 
                                 (in aggregate)
        arrived_prevs (jnp arr) : Array of whether each orb has arrived at its destination
        sq (bool) : If true, makes regions that are squares, 
                    If false NOT IMPLEMENTED, make plus or circle shapes regions
    Returns:
    --------
        n_key (jnp array) : new key for generating new random numbers
        out_corners (jnp array) : orb i's (x,y) corner it transited to.
        arrived_tos (jnp array) : Whether orb i reached its destination
    """   

    dests, diams = get_diameters_and_dests(states_c, target_state, jitv=True)

    arr_indices = (0, 0, 0, None, 0, None, 0) #array shape and epsi are const for all
    n_keys = jnp.array(jax.random.split(c_key, num=len(corners)))

    #Split into arrived and not-yet-arrived groups.
    arr_keys, tow_keys = n_keys[arrived_prevs], n_keys[~arrived_prevs]
    arr_dests, tow_dests = dests[arrived_prevs], dests[~arrived_prevs]
    arr_diams, tow_diams = diams[arrived_prevs], diams[~arrived_prevs]
    arr_corners, tow_corners = corners[arrived_prevs], corners[~arrived_prevs]
    
    arrived_indices = (0, 0, None, 0, 0, None)
    toward_indices = (0, 0, None, 0, 0, None, None)

    vmapped_arrived = jax.vmap(orb_step_arived_st, arrived_indices, (0,0))
    vmapped_toward = jax.vmap(orb_step_toward_st, toward_indices, (0,0))

    n_keys_arr, corners_arr, arrived_to_arr = vmapped_arrived(
        arr_keys, arr_dests, array_shape, arr_corners, arr_diams, sq)
    n_keys_tow, corners_tow, arrived_to_tow = vmapped_toward(
        tow_keys, tow_dests, array_shape, tow_corners, tow_diams, epsi, sq)

    arrived_tos = jnp.zeros((len(corners),))
    arrived_tos = arrived_tos.at[jnp.where(arrived_prevs)].set(arrived_to_arr)
    arrived_tos = arrived_tos.at[jnp.where(~arrived_prevs)].set(arrived_to_tow)
    out_corners = jnp.zeros(corners.shape)
    out_corners = out_corners.at[jnp.where(arrived_prevs)].set(corners_arr)
    out_corners = out_corners.at[jnp.where(~arrived_prevs)].set(corners_tow)

    n_key = jax.random.split(n_keys_arr[0])[0] #Weirdly, the inner set is 2 keys
    return n_key, out_corners, arrived_tos


def all_orbs_one_st_lazy(c_key, corners, states_c, target_state, 
                         array_shape, epsi, arrived_prevs, sq=True):
    """
    Moves all orbs one step, No Vmapping, so slow. 

    Inputs:
    --------
        c_key (jnp array) : Jax array/Current Key for generating random numbers
        corners (jnp arr) : jnp array of (x,y) pairs indicating where the 
                            lowest x, lowest y valued corner a box around the 
                            orb at index i would be
        states_c (dict) : Dict of State Number: Dict of Orb Numbers: tup of:
                          - valid region/destination locations for The given orb 
                          (eg where it can move to when in the state) 
                          - orb diam
        target_state (int) : State number that you're progressing towards, a key for states_c
        array_shape (tup) : shape of the canvas 
        epsi (float in [.5,1]) : Probability of an orb moving towards its destination 
                                 (in aggregate)
        arrived_prevs (jnp arr) : Array of whether each orb has arrived at its destination
        sq (bool) : If true, makes regions that are squares, 
                    If false NOT IMPLEMENTED, make plus or circle shapes regions
    Returns:
    --------
        n_key (jnp array) : new key for generating new random numbers
        out_corners (jnp array) : orb i's (x,y) corner it transited to.
        arrived_tos (jnp array) : Whether orb i reached its destination
    """
    out_corners = []
    arrived_tos = []
    n_key, subkey = jax.random.split(c_key)
    dests, diams = get_diameters_and_dests(states_c, target_state)
    for i in range(len(dests)):
        destination = dests[i]
        orb_diam = diams[i]
        corner = corners[i]
        print(corner, orb_diam)
        arrived_prev = arrived_prevs[i]
        subkey, trans_to, arrived_to = one_orb_one_st(subkey, corner, destination, 
                                        array_shape, orb_diam, epsi, arrived_prev, sq=True)
        out_corners.append(trans_to)
        arrived_tos.append(arrived_to)
        
    return n_key, out_corners, arrived_tos
        

def get_diameters_and_dests(states_c, target_state, jitv=False):
    """
    Gets the diameters and destinations of the target states, ordered by orb num
    """
    destinations = states_c[target_state]
    orb_info_l = sorted([(orb, dest) for orb, dest in destinations.items()])
    if jitv:
        dests = jnp.array([dest[1][0] for dest in orb_info_l])
        diams = jnp.array([dest[1][1] for dest in orb_info_l])
    else:
        dests = [dest[1][0] for dest in orb_info_l]
        diams = [dest[1][1] for dest in orb_info_l]
    return dests, diams


def to_tups(destination):
    return set([(int(loc[0]),int(loc[1])) for loc in destination])


#############################
# Visuals and Support Funcs #
#############################


def noise_visual():
    """
    Creates my visual comparing my rotation to regular noise items
    """
    xmin, xmax, ymin, ymax = 0, 20, 20, 0 
    array_shape = (100,100)
    noise_rot, f_noise_rot, n_key = gen_low_freq_noise_rot(array_shape, MY_KEY, l_bd=-1, u_bd=1, 
                       cutoff=0.025, r_cutoff=False, cutoff_bds=[0.02, 0.02])
    noise_quar, f_noise_quar, n_key = gen_low_freq_noise_quarter(array_shape, MY_KEY, l_bd=-1, u_bd=1, 
                       cutoff=0.05, r_cutoff=False, cutoff_bds=[0.02, 0.02])
    noise, f_noise, n_key = gen_low_freq_noise(array_shape, MY_KEY, l_bd=-1, u_bd=1, 
                       cutoff=0.05, r_cutoff=False, cutoff_bds=[0.02, 0.02])
    
    fig, axs = plt.subplots(3,2)
    axs[0, 0].imshow(noise_quar, cmap='hot', interpolation='nearest')
    axs[0, 0].set_title('Quarter Noise')
    axs[0, 1].imshow(f_noise_quar, cmap='hot', interpolation='nearest')
    axs[0, 1].set_title('Kept Fourier Space: Quarter Circle')
    axs[0, 1].set_xlim([xmin, xmax])
    axs[0, 1].set_ylim([ymin, ymax])
    axs[1, 0].imshow(noise_rot, cmap='hot', interpolation='nearest')
    axs[1, 0].set_title('Circle Noise')
    axs[1, 1].imshow(f_noise_rot, cmap='hot', interpolation='nearest')
    axs[1, 1].set_title('Kept Fourier Space: Circle')
    axs[1, 1].set_xlim([xmin, xmax])
    axs[1, 1].set_ylim([ymin, ymax])
    axs[2, 0].imshow(noise, cmap='hot', interpolation='nearest')
    axs[2, 0].set_title('Corners Noise')
    axs[2, 1].imshow(f_noise, cmap='hot', interpolation='nearest')
    axs[2, 1].set_title('Kept Fourier Space: Corners')
    axs[2, 1].set_xlim([xmin, xmax])
    axs[2, 1].set_ylim([ymin, ymax])
    plt.show()


def prob_distro_vis(epsi=.8):
    """
    Visualizes the basic prob distro of a state moving towards a destination.
    """
    array_shape = (20, 20)
    orb_diam = 8
    #corner = jnp.array([19-int(orb_diam/2),10]) #Side case
    #corner = jnp.array([19-int(orb_diam/2),19-int(orb_diam/2)])
    corner = jnp.array([10,10])

    destination = jnp.array(list(product(range(1,5),range(1,5))))
    dest_xs, dest_ys = zip(*destination)
    valid_transitions, trans_probs = prob_distro(\
        epsi, corner, orb_diam, array_shape, destination, sq=True)
    img = jnp.zeros(array_shape)
    img = img.at[dest_xs, dest_ys].set(.1)
    xs, ys = zip(*valid_transitions)
    img = img.at[xs, ys].set(trans_probs)
    img = jnp.ones(array_shape) - img
    plt.imshow(img, cmap='hot', interpolation='nearest')
    plt.show()


def orb_moving_visual(c_key, save_folder, n_steps=100):
    """
    Creates the visuals of a single orb moving to its destination.
    """
    #Inits
    val = 0 #Value for the orb and dest to show up
    n_key = c_key
    array_shape = (100,100)
    destination = jnp.array(list(product(range(3,7),range(3,7))))
    dest_xs, dest_ys = zip(*destination)
    corner = jnp.array([50,50])
    orb_diam = 8
    epsi = 0.75


    arrived_to = False #False for simplicity 
    arrived_prev = arrived_to
    trans_to = corner
    corners = [trans_to]
    for i in range(n_steps):
        canvas = jnp.ones(array_shape)
        n_key, trans_to, arrived_prev = one_orb_one_st(
                n_key, trans_to, destination, array_shape, 
                orb_diam, epsi, arrived_prev, sq=True)
        
        og_orb_coords = get_orb_pts(array_shape, corner, orb_diam)
        orb_coords = get_orb_pts(array_shape, trans_to, orb_diam)
        corners.append(trans_to)
        
        cor_og_xs, cor_og_ys = zip(*og_orb_coords)
        xs_orb, ys_orb = zip(*orb_coords)
        cor_xs, cor_ys = zip(*corners)

        canvas = canvas.at[cor_og_xs, cor_og_ys].set(.7) #Draw original orb
        canvas = canvas.at[xs_orb, ys_orb].set(val) #Draw current orb
        canvas = canvas.at[dest_xs, dest_ys].set(.1) #Draw Dest
        canvas = canvas.at[cor_xs, cor_ys].set(0.5) #Draw corners of all time
        canvas = canvas.at[trans_to[0], trans_to[1]].set(0.2) #Draw current corner

        plt.imshow(canvas, cmap='hot', interpolation='nearest')
        plt.axis([0, array_shape[0], 0, array_shape[1]])
        fname = save_folder%i
        plt.savefig(fname)


def visualize_states(c_key, states_folder=my_vars.stateImgsP, save=True, preload=True, 
                     n_states=3, array_shape = (120,120)):
    lb_orbs = 8
    ub_orbs = 8
    if preload:
        states_i, states_c, states_f, n_key = full_states_load(n_states=n_states)
    else:
        n_key, states_i, states_c, states_f = gen_states(
            c_key, n_states, array_shape, lb_orbs, ub_orbs, fix_avg=20, fix_stv=4, 
            fix_stv_stv=2, lb_size=15, ub_size=25, region_d=5)
    
    for state, expected_img in states_i.items():
        plt.imshow(expected_img, cmap='hot', interpolation='nearest')
        plt.axis([0, array_shape[0], 0, array_shape[1]])
        plt.colorbar()
        fname = states_folder%state
        if save:
            plt.savefig(fname)
        if not save:
            plt.show()
        plt.clf()


def vis_one_step(c_key, lazy=True):

    n_states = 5
    array_shape = (100,100)
    lb_orbs = 5
    ub_orbs = 5
    n_key, states_i, states_c, states_f = gen_states(
        c_key, n_states, array_shape, lb_orbs, ub_orbs, fix_avg=20, fix_stv=4, 
        fix_stv_stv=2, lb_size=15, ub_size=20, region_d=5)
    
    corners = jnp.array([[25,25], [25, 75], [50,50],[75,25],[75,75]])
    target_state = 3
    epsi = 0.8
    arrived_prevs = jnp.array([False, False, False, False, False])

    s = time.time()
    if lazy:
        n_key, n_corners, arrived_to = all_orbs_one_st_lazy(
            c_key, corners, states_c, target_state, array_shape, epsi, arrived_prevs, 
            sq=True)
    else:
        n_key, n_corners, arrived_to = all_orbs_one_st(
            c_key, corners, states_c, target_state, array_shape, epsi, arrived_prevs, 
            sq=True)
    e = time.time()
    print("Time to move all orbs 1 step: ", e - s)
    
    canvas = jnp.ones(array_shape)
    dests, diams = get_diameters_and_dests(states_c, target_state)
    for i in range(len(diams)):
        orb_diam = diams[i]
        corner = corners[i]
        trans_to = n_corners[i]
        og_orb_coords = get_orb_pts(array_shape, corner, orb_diam)
        orb_coords = get_orb_pts(array_shape, trans_to, orb_diam)
        cor_og_xs, cor_og_ys = zip(*og_orb_coords)
        xs_orb, ys_orb = zip(*orb_coords)

        canvas = canvas.at[cor_og_xs, cor_og_ys].set(.7) #Draw original orb
        canvas = canvas.at[xs_orb, ys_orb].set(0) #Draw current orb

    plt.imshow(canvas, cmap='hot', interpolation='nearest')
    plt.axis([0, array_shape[0], 0, array_shape[1]])
    #fname = save_folder%i
    #plt.savefig(fname)
    plt.show()
    

def vis_state_trans(c_key, img_save_folder, arr_save_folder, lazy=True):
    n_iter = 200
    n_states = 5
    array_shape = (120,120)
    lb_orbs = 12
    ub_orbs = 12
    ub_size = 25
    n_key, states_i, states_c, states_f = gen_states(
        c_key, n_states, array_shape, lb_orbs, ub_orbs, fix_avg=20, fix_stv=4, 
        fix_stv_stv=2, lb_size=15, ub_size=ub_size, region_d=5)

    target_state = 3
    epsi = 0.8
    arrived_prevs = jnp.zeros((ub_orbs, )) #Effectivly all false?
    simple_x_starts = jnp.arange(ub_size, array_shape[0] - ub_size, 1)
    simple_y_starts = jnp.arange(ub_size, array_shape[1] - ub_size, 1)
    corners = jnp.array(list(product(simple_x_starts, simple_y_starts)))
    n_key, subkey = jax.random.split(n_key)
    corner_indices = jax.random.randint(key=subkey, shape=(ub_orbs,), 
                                            minval=0, maxval=corners.shape[0])
    n_corners = corners[corner_indices]
    print(n_corners)
    dests, diams = get_diameters_and_dests(states_c, target_state)

    for j in range(n_iter):
        print("Step: ", j)
        if lazy:
            n_key, n_corners, arrived_prevs = all_orbs_one_st_lazy(
                n_key, corners, states_c, target_state, array_shape, epsi, arrived_prevs, 
                sq=True)
        else:
            n_key, n_corners, arrived_prevs = all_orbs_one_st(
                n_key, corners, states_c, target_state, array_shape, epsi, arrived_prevs, 
                sq=True)
            
        print(n_corners)
        print(arrived_prevs)
        print()
        
        canvas = jnp.ones(array_shape)
        for orb in range(len(diams)):
            orb_diam = diams[orb]
            trans_to = n_corners[orb]
            orb_coords = get_orb_pts(array_shape, trans_to, orb_diam)
            xs_orb, ys_orb = zip(*orb_coords)

            canvas = canvas.at[xs_orb, ys_orb].set(0) #Draw current orb

        arr_fname = arr_save_folder%j
        jnp.save(arr_fname, canvas)

        plt.imshow(canvas, cmap='hot', interpolation='nearest')
        plt.axis([0, array_shape[0], 0, array_shape[1]])
        plt_fname = img_save_folder%j
        plt.savefig(plt_fname)
        plt.clf()


def vis_state_trans2(n_states, st_st, st_end):
    c_key=MY_KEY
    img_save_folder=my_vars.orbsToStateP 
    arr_save_folder=my_vars.rawArraysP
    states_i, states_c, states_f, bleh_key = full_states_load(n_states)
    save_arrs = True
    save_figs = True
    array_shape = (120, 120)
    
    n_key, corners = simulate_corners(c_key, states_c, st_st)
    transition_to_state(n_key, states_c, st_st, st_end, save_arrs, save_figs, corners, 
                        img_save_folder, arr_save_folder, array_shape, epsi=0.7, lazy=True)
    

def simulate_corners(c_key, states_c, st):
    """
    Picks random corners from the given state's destination for 
    each orb.
    
    Inputs:
    --------
        c_key (jnp array) : Jax array/Current Key for generating random numbers
        states_c (dict) : Dict of State Number: Dict of Orb Numbers: tup of:
                          - valid region/destination locations for The given orb 
                          (eg where it can move to when in the state) 
                          - orb diam
        st (int) : The current state which corners are being generated for
                          
    Returns:
    --------
        n_key (jnp array) : new key for generating new random numbers
        corners (jnp arr) : jnp array of (x,y) pairs indicating where the 
                            lowest x, lowest y valued corner a box around the 
                            orb at index i would be
    """
    corners = []
    n_key = c_key
    dests, diams = get_diameters_and_dests(states_c, st)
    
    for dest in dests:
        num_dests = len(dest)
        n_key, subkey = jax.random.split(n_key)
        my_index = int(jax.random.randint(subkey, shape=(1,), minval=0, maxval=num_dests))
        corner = dest[my_index]
        corners.append(corner)
    
    corners = jnp.array(corners)
    print([(corners[i], diams[i]) for i in range(len(diams))])
    return n_key, corners


###################
# Full Simulation #
###################


def dup_dict(orig_d, key_modifier):
    """
    Takes a dictionary and returns a new dictionary where every key 
    is the key from the old dictionary + key_modifier
    """
    new_d = dict()
    for k, v in orig_d.items():
        new_d[k + key_modifier] = v
    return new_d


def full_states_save(c_key = MY_KEY, n_states = 30, array_shape = (120,120), 
                    lb_orbs=12, ub_orbs=12, fix_avg=20, fix_stv=4, 
                    fix_stv_stv=2, lb_size=15, ub_size=25, region_d=5,
                    save_loc = my_vars.generatedDataPath, in_parts=True,
                    part_step=3, load_prev=False, start_load=1):
    """
    Generates all the state information for the given number of states. 
    Does not transition between them, just saves the relevant information, 
    enabling me to say "I want to only use 5 states, lets simulate those".

    Inputs:
    --------
        in_parts (bool) : Whether to generate my total dict in parts, eg 
                        say 3 states at a time or all at once.
        part_step (int) : Size of steps to generate if in_parts is true
        load_prev (bool) : If True, then loads previous saved work and continues
                           based off of it.
        start_load (int) : If load_prev is true, then generates states from this number
                           onwards.
    """    
    
    if load_prev:
        states_i, states_c, states_f, n_key = full_states_load(n_states, save_loc)
    else:
        states_i = dict()
        states_c = dict() 
        states_f = dict()
        n_key = c_key
    
    states_i_path = str(save_loc.joinpath("%d_states_i.pickle"%n_states))
    states_c_path = str(save_loc.joinpath("%d_states_c.pickle"%n_states))
    states_f_path = str(save_loc.joinpath("%d_states_f.pickle"%n_states))
    n_key_path = str(save_loc.joinpath("%d_states_n_key.pickle"%n_states))

    for i in range(1, n_states+1):
        eval_criterion = (i%part_step==0 and load_prev and i >= start_load) \
                          or (i%part_step==0 and not load_prev)
        if eval_criterion:

            print(i)

            # Gen part_step states at a time. Then we'll combine them w/ previous results.
            n_key, states_i_step, states_c_step, states_f_step = gen_states(
                n_key, part_step, array_shape, lb_orbs, ub_orbs, fix_avg, fix_stv, 
                fix_stv_stv, lb_size, ub_size, region_d)

            states_i_step = dup_dict(states_i_step, key_modifier=i-part_step)
            states_c_step = dup_dict(states_c_step, key_modifier=i-part_step)
            states_f_step = dup_dict(states_f_step, key_modifier=i-part_step)

            states_i = {**states_i, **states_i_step}
            states_c = {**states_c, **states_c_step}
            states_f = {**states_f, **states_f_step}
            
            print(states_i)
            print()
            print(states_c)
            print()
            print(states_f)

            with open(states_i_path, 'wb') as handle:
                pickle.dump(states_i, handle)

            with open(states_c_path, 'wb') as handle:
                pickle.dump(states_c, handle)

            with open(states_f_path, 'wb') as handle:
                pickle.dump(states_f, handle)

            with open(n_key_path, 'wb') as handle:
                pickle.dump(n_key, handle)

            num_done = i

        # Capture the last little few if my steps didn't perfectly 
        # include all states
        elif i == n_states + 1:
            print("Final: ", i)
            num_needed = n_states +1  - num_done

            n_key, states_i_step, states_c_step, states_f_step = gen_states(
                n_key, num_needed, array_shape, lb_orbs, ub_orbs, fix_avg, fix_stv, 
                fix_stv_stv, lb_size, ub_size, region_d)
            
            states_i_step = dup_dict(states_i_step, key_modifier=num_done) #num_done + 1??
            states_c_step = dup_dict(states_c_step, key_modifier=num_done)
            states_f_step = dup_dict(states_f_step, key_modifier=num_done)

            states_i = {**states_i, **states_i_step}
            states_c = {**states_c, **states_c_step}
            states_f = {**states_f, **states_f_step}

            with open(states_i_path, 'wb') as handle:
                pickle.dump(states_i, handle)

            with open(states_c_path, 'wb') as handle:
                pickle.dump(states_c, handle)

            with open(states_f_path, 'wb') as handle:
                pickle.dump(states_f, handle)

            with open(n_key_path, 'wb') as handle:
                pickle.dump(n_key, handle)
            

def full_states_load(n_states, save_loc = my_vars.generatedDataPath):
    """
    Loads the saved dictionaries from full_states_save
    """

    states_i_path = str(save_loc.joinpath("%d_states_i.pickle"%n_states))
    states_c_path = str(save_loc.joinpath("%d_states_c.pickle"%n_states))
    states_f_path = str(save_loc.joinpath("%d_states_f.pickle"%n_states))
    n_key_path = str(save_loc.joinpath("%d_states_n_key.pickle"%n_states))

    with open(states_i_path, 'rb') as f:
        states_i = pickle.load(f)

    with open(states_c_path, 'rb') as f:
        states_c = pickle.load(f)

    with open(states_f_path, 'rb') as f:
        states_f = pickle.load(f)

    with open(n_key_path, 'rb') as f:
        n_key = pickle.load(f)

    return states_i, states_c, states_f, n_key


def transition_to_state(c_key, states_c, st_st, st_end, save_arrs, save_figs, corners, 
                        img_save_folder, arr_save_folder, array_shape, epsi=0.7, lazy=True):
    """
    Transitions between 2 states, potentially saving the info generated along the way.

    Args:
        c_key (_type_): _description_
        states_c (_type_) : Dict of State Number: Dict of Orb Numbers: tup of:
                          - valid region/destination locations for The given orb 
                          (eg where it can move to when in the state) 
                          - orb diam
        st_st (_type_): _description_
        st_end (_type_): _description_
        img_save_folder (_type_): _description_
        arr_save_folder (_type_): _description_
        lazy (bool): _description_. Defaults to True.
    """
    
    init_info = states_c[st_st]
    dests, diams = get_diameters_and_dests(states_c, st_end)
    dests_tups = [to_tups(dest) for dest in dests]
    corner_tups = [(int(corner[0]), int(corner[1])) for corner in corners]
    
    init_arrived_prevs = [corner_tups[orb] in dest for orb, dest in enumerate(dests_tups)]     
    
    j = 0
    n_key = c_key
    n_corners = corners
    arrived_prevs = init_arrived_prevs
    in_dest = all(arrived_prevs)
    
    print([(corners[i], diams[i]) for i in range(len(diams))])
    
    while not in_dest:
        j += 1
        print("Step: ", j)
        if lazy:
            n_key, n_corners, arrived_prevs = all_orbs_one_st_lazy(
                n_key, n_corners, states_c, st_end, array_shape, epsi, arrived_prevs, 
                sq=True)
        else:
            n_key, n_corners, arrived_prevs = all_orbs_one_st(
                n_key, n_corners, states_c, st_end, array_shape, epsi, arrived_prevs, 
                sq=True)
        
        if save_arrs | save_figs:
            canvas = jnp.ones(array_shape)
            for orb in range(len(diams)):
                orb_diam = diams[orb]
                trans_to = n_corners[orb]
                orb_coords = get_orb_pts(array_shape, trans_to, orb_diam)
                xs_orb, ys_orb = zip(*orb_coords)

                canvas = canvas.at[xs_orb, ys_orb].set(0) #Draw current orb

            if save_arrs:
                arr_fname = arr_save_folder%j
                jnp.save(arr_fname, canvas)

            if save_figs:
                plt.imshow(canvas, cmap='hot', interpolation='nearest')
                plt.axis([0, array_shape[0], 0, array_shape[1]])
                plt_fname = img_save_folder%j
                plt.savefig(plt_fname)
                plt.clf()
        
        in_dest = all(arrived_prevs)

    return n_key, n_corners


if __name__ == "__main__":
    array_shape = (100,100)
    #orb_moving_visual(c_key=MY_KEY, save_folder=my_vars.stateToDestP, n_steps=100)
    #visualize_states(c_key=MY_KEY, states_folder=my_vars.stateImgsP, save=False)
    #noise_visual()
    #prob_distro_vis(epsi=.8)
    #vis_one_step(c_key=MY_KEY)
    #vis_state_trans(c_key=MY_KEY, img_save_folder=my_vars.orbsToStateP, 
    #                   arr_save_folder=my_vars.rawArraysP)
    

    n_key, adj_mat, G = gen_graph(c_key=MY_KEY, n_states=20, p=0.1, self_loops=True)
    vis_graph(G)
    #vis_graph2(adj_mat)

    #visualize_states(c_key=MY_KEY, states_folder=my_vars.stateImgsP, save=True, 
    #                 preload=True, n_states=30, array_shape = (120,120))
    """
    states_i, states_c, states_f, n_key = full_states_load(n_states=30)
    for state, orb_d in states_c.items():
        destinations = states_c[state]
        orb_info_l = sorted([(orb, dest) for orb, dest in destinations.items()])
        orb_diams = [(orb, dest[1][1]) for orb, dest in enumerate(orb_info_l)]
        print(orb_diams)
    #print(states_f)
    """
    """
    full_states_save(c_key = MY_KEY, n_states = 30, array_shape = (120,120), 
                    lb_orbs=16, ub_orbs=16, fix_avg=20, fix_stv=4, 
                    fix_stv_stv=2, lb_size=15, ub_size=25, region_d=5,
                    save_loc = my_vars.generatedDataPath, in_parts=True,
                    part_step=2, load_prev=False, start_load=0)
    """
    #print(states_f)
    #vis_state_trans2(n_states=30, st_st=1, st_end=2)
    