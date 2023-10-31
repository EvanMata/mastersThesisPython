import jax 
import jax.numpy as jnp
import networkx as nx
import matplotlib.pyplot as plt

from itertools import product
from scipy.spatial.distance import cdist

MY_KEY = jax.random.PRNGKey(0)


####################
# Noise Generation #
####################


def gen_low_freq_noise_rot(array_shape, c_key, l_bd=-1, u_bd=1, 
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

    x_cen = int(array_shape[0]/2)
    y_cen = int(array_shape[1]/2)
    #potential_pairs = jnp.array(list(product(range(x_cen), range(y_cen))))
    potential_pairs = jnp.array(list(product(range(array_shape[0]), range(array_shape[1]))))
    elipse_vals = jnp.array([float(((v[0] - x_cen)**2)/x_cen + ((v[1] - y_cen)**2)/y_cen)
                                for v in potential_pairs])
    valid_indices = potential_pairs[jnp.where(elipse_vals > (1-cutoff_rad)*(x_cen*y_cen)**0.5)]
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
        states_c (dict) : Dict of State Number: Dict of Orb Numbers: valid region/destination
                          locations for The given orb (eg where it can move to when in the state) 
        states_s (dict) : Dict of State Number: 
                            [Expected Number of itterations to stay in state, before 
                             identifying new state to transition to,
                             Stv of Number of itterations to stay in state]
        trans (jnp array) : Jax array of transition probabilities between states
    """
    n_key, subkey = jax.random.split(c_key)
    region_rad = int(region_d/2)
    x_width, y_height = array_shape[0], array_shape[1]
    valid_corners = jnp.array(list(product(range(x_width), range(y_height)))) - region_rad

    trans = 5

    states_i = dict()
    states_c = dict()
    states_f = dict()

    # Setup the distributions of how long each state stays fixed
    n_key, subkey = jax.random.split(n_key)
    state_duration_avgs = fix_stv*(jnp.random.normal(subkey, shape=(n_states,))) + fix_avg
    min_state_duration = max([0, fix_avg - fix_stv*3])
    state_duration_avgs = jnp.clip(state_duration_avgs, a_min=min_state_duration)

    n_key, subkey = jax.random.split(n_key)
    state_duration_variances = fix_stv_stv*(jnp.random.normal(subkey, shape=(n_states,))) + fix_stv
    state_duration_variances = jnp.clip(state_duration_variances, a_min=0.1)
    
    for state in range(n_states):
        n_orbs = ub_orbs #ADJUST THIS LATER
        orb_num_to_corner = dict()
        visual = jnp.zeros(array_shape)
        n_key, subkey = jax.random.split(n_key)
        corner_indices = jnp.random.randint(key=subkey, shape=(n_orbs,), 
                                            minval=0, maxval=valid_corners.size)
        corners = valid_corners[corner_indices] #corners for each orb for the given state

        n_key, subkey = jax.random.split(n_key)
        orb_diams = jnp.random.randint(key=subkey, shape=(n_orbs,), 
                                       minval=lb_size, maxval=ub_size)

        for orb in range(len(corners)):
            orb_diam = orb_diams[orb]
            corner = corners[orb]
            region = get_region(array_shape, corner, region_d, orb_diam)
            orb_num_to_corner[orb] = region
            img_base_weight = 1/len(region)
            for region_c in region:
                orb_pts = get_orb_pts(array_shape, corner, orb_diam)
                orb_xs, orb_ys = zip(*orb_pts)
                visual = visual.at[orb_xs, orb_ys].add(img_base_weight)

        states_i[state] = visual
        states_c[state] = orb_num_to_corner
        state_dur_avg, state_dur_stv = state_duration_avgs[state], state_duration_variances[state]
        states_f[state] = (state_dur_avg, state_dur_stv)


    return n_key, states_i, states_c, states_f, trans


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
    orb_rad = int(orb_diam/2)
    x_cor = corner[0]
    y_cor = corner[1]
    canvas_w = array_shape[0]
    canvas_h = array_shape[1]
    if sq:
        min_x = max([-orb_rad, x_cor])
        max_x = min([x_cor + region_d, canvas_w - orb_rad])
        min_y = max([-orb_rad, y_cor])
        max_y = min([y_cor + region_d, canvas_h - orb_rad])
        xs = range(min_x, max_x)
        ys = range(min_y, max_y)
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
    
    """
    adj_mat = jnp.zeros((n_states, n_states))
    n_key, subkey = jax.random.split(c_key)
    er_graph = nx.erdos_renyi_graph(n_states, p, seed=int(subkey[0]))
    er_graph = jnp.array(nx.to_numpy_array(er_graph))
    fully_connected, n_key = random_fully_connected(n_states, n_key)
    connected_indices = jnp.logical_or(fully_connected, er_graph)
    if self_loops:
        id = jnp.identity(n_states)
        connected_indices = jnp.logical_or(connected_indices, id)
    connected_indices_tri1 = jnp.triu(connected_indices)
    num_edges_tri1 = jnp.sum(connected_indices)

    n_key, subkey = jax.random.split(n_key)
    pre_adjusted_edge_weights = jnp.random.normal(subkey, shape=(num_edges_tri1, ))
    inds_for_wieghts = jnp.where(connected_indices_tri1)
    adj_mat = adj_mat.at[inds_for_wieghts].set(pre_adjusted_edge_weights)

    #Now make it dobuley stochastic
    #softmax row 0
    #softmax row 1 from 1-n (eg not including 0) - multiply by (1-M[0,1]), so that 
    # it adds with row 0 value 1
    # repeat all the way down
    # Is top right much more likely to be larger values? Probabily?

    top_r = jnp.triu(adj_mat, k=1)
    adj_mat = 5

    #Dobule stochastic times doubly stochastic is doubly stochastic
    #So make my random fully connected graph, 
    #
    

    #add [k,k] + top_right + top_right.T

    #random_graph = jnp.clip(random_graph) #No, Softmax it instead to make it row stochastic


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
    id = jnp.identiy(n_states)
    perm = jax.random.shuffle(subkey, id)
    return perm, n_key


def permute_matrix(matrix, c_key):
    """
    Permutates a matrix with a random permutation
    """
    n_states = matrix.shape[0]
    perm, n_key = gen_perm(n_states, c_key)
    permuted_m = jnp.matmul(perm.T, jnp.matmul(matrix, perm))
    return permuted_m, n_key


##############
# Move 1 Orb #
##############

def prob_distro(epsi, corner, orb_diam, array_shape, destination, sq=True):
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

    diam = 5 #Diameter of region where orb can travel to.
    
    valid_transitions = get_region(array_shape, corner - 2, diam, orb_diam, sq) #For corners, global
    trans_probs = jnp.zeros(valid_transitions.shape[0],)
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


def orb_step_toward_st(c_key, *args):
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
    valid_transitions, trans_probs = prob_distro(*args)
    trans_to = jax.random.choice(subkey, valid_transitions, p=trans_probs)
    return n_key, trans_to


def orb_step_arived_st(c_key, destination, *args):
    """
    Picks where the orb moves if it has already arrived at its destination
    Should update this if destinations get large, so that jumps aren't more than one step.

    Inputs:
    --------
        c_key (jnp array) : Jax array/Current Key for generating random numbers
        destination (jnp array) : states_c[state][orb] value, an array of tuples of 
                    corner locations that the orb is heading towards
        *args (lst) : args for get_region, eg [array_shape, corner, 1, orb_diam, sq]
    Returns:
    -------
        n_key (jnp array) : new key for generating new random numbers
        trans_to (jnp array) : (x,y) corner to transit to.
    """
    valid_transitions = get_region(*args)
    valid_trans = set([(int(loc[0]),int(loc[1])) for loc in valid_transitions])
    dests = set([(int(loc[0]),int(loc[1])) for loc in destination])
    valid_dests = list(dests.intersection(valid_trans)) 
    n_key, subkey = jax.random.split(c_key)
    trans_to_index = jax.random.randint(subkey, shape=(1,), minval=0, maxval=len(valid_dests))
    trans_to = destination[trans_to_index][0]
    return n_key, trans_to


def one_orb_one_st(c_key, corner, destination, array_shape, orb_diam, epsi, arrived_prev, 
                   sq=True):
    """
    Moves one orb one state

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

    """
    corner_t = (int(corner[0]), int(corner[1]))
    dests = set([(int(loc[0]),int(loc[1])) for loc in destination])
    if arrived_prev:
        my_args = [array_shape, corner, 1, orb_diam, sq]
        n_key, trans_to = orb_step_arived_st(c_key, destination, *my_args)
    else:
        my_args = [epsi, corner, orb_diam, array_shape, destination, sq]
        n_key, trans_to = orb_step_toward_st(c_key, *my_args)
    trans_to_t = (int(trans_to[0]), int(trans_to[1]))
    arrived_to = trans_to_t in dests
    return n_key, trans_to, arrived_to


#############################
# Visuals and Support Funcs #
#############################


def noise_visual():
    """
    Creates my visual comparing my rotation to regular noise items
    """
    array_shape = (100,100)
    noise_rot, f_noise_rot, n_key = gen_low_freq_noise_rot(array_shape, MY_KEY, l_bd=-1, u_bd=1, 
                       cutoff=0.05, r_cutoff=False, cutoff_bds=[0.02, 0.02])
    noise, f_noise, n_key = gen_low_freq_noise(array_shape, MY_KEY, l_bd=-1, u_bd=1, 
                       cutoff=0.05, r_cutoff=False, cutoff_bds=[0.02, 0.02])
    
    fig, axs = plt.subplots(2,2)
    axs[0, 0].imshow(noise_rot, cmap='hot', interpolation='nearest')
    axs[0, 0].set_title('Rotational Noise')
    axs[0, 1].imshow(f_noise_rot, cmap='hot', interpolation='nearest')
    axs[0, 1].set_title('Kept Fourier Space: Rot')
    axs[1, 0].imshow(noise, cmap='hot', interpolation='nearest')
    axs[1, 0].set_title('Corners Noise')
    axs[1, 1].imshow(f_noise, cmap='hot', interpolation='nearest')
    axs[1, 1].set_title('Kept Fourier Space: Corners')
    plt.show()


def prob_distro_vis(epsi=.8):
    """
    Visualizes the basic prob distro of a state moving towards a destination.
    """
    array_shape = (20, 20)
    orb_diam = 8
    corner = jnp.array([19-int(orb_diam/2),10]) #Side case
    corner = jnp.array([19-int(orb_diam/2),19-int(orb_diam/2)])
    
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


def state_moving_visual(c_key, save_folder, n_steps=100):
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
    #for i in range(n_steps):
    for i in range(5):
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
        canvas = canvas.at[dest_xs, dest_ys].set(val) #Draw Dest
        canvas = canvas.at[cor_xs, cor_ys].set(0.5) #Draw corners of all time
        canvas = canvas.at[trans_to[0], trans_to[1]].set(0.2) #Draw current corner

        plt.imshow(canvas, cmap='hot', interpolation='nearest')
        plt.axis([0, array_shape[0], 0, array_shape[1]])
        base_fname = "state_moving_frame_%"%i
        fname = save_folder + base_fname
        plt.savefig()


if __name__ == "__main__":
    array_shape = (100,100)
    """
    noise, f_noise, n_key = gen_low_freq_noise(array_shape, MY_KEY, l_bd=-1, u_bd=1, 
                       cutoff=0.15, r_cutoff=False, cutoff_bds=[0.02, 0.02])
    plt.imshow(noise, cmap='hot', interpolation='nearest')
    plt.show()
    """
    #prob_distro_vis(epsi=0.8)
    state_moving_visual(c_key=MY_KEY, save_folder=5, n_steps=100)