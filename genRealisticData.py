import jax 
import jax.numpy as jnp
import networkx as nx
import matplotlib.pyplot as plt

from itertools import product

MY_KEY = jax.random.PRNGKey(0)

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


def gen_states(c_key, n_states, array_shape, lb_orbs, ub_orbs, st_avg=20, st_stv=4, 
               lb_size=6, ub_size=10, region_d=5):
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
        st_avg (int) : The expected average of the number of itterations a state stays fixed
        st_stv (float) : The Standard Deviation of the expected number of itterations 
                         a state stays fixed
        lb_size (int) : Minimum diameter of an orb
        ub_size (int) : Maximum diameter of an orb
        region_d (int) : Diameter of a region for an orb to be centered in.

    Returns:
    --------
        n_key (jnp array) : new key for generating new random numbers
        states_i (dict) : Dict of State Number: Expected Image of State (array)
        states_c (dict) : Dict of State Number: Dict of Orb Numbers: valid region locations
                          For The given orb (eg where it can move to when in the state) 
        states_s (dict) : Dict of State Number: Expected Number of itterations to stay in state
                          before identifying new state to transition to.
        trans (jnp array) : Jax array of transition probabilities between states
    """
    n_key, subkey = jax.random.split(c_key)
    region_rad = int(region_d/2)
    x_width, y_height = array_shape[0], array_shape[1]
    valid_corners = jnp.array(list(product(range(x_width), range(y_height)))) - region_rad

    trans = 5

    states_i = dict()
    states_c = dict()
    states_s = dict()


    
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


    return n_key, states_i, states_c, states_s, trans


def get_region(array_shape, corner, region_d, orb_diam, sq=True):
    """
    Given the corner of the region for a given orb, calculate the relevant region
    of points that are ok.
    
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





if __name__ == "__main__":
    array_shape = (100,100)
    noise, f_noise, n_key = gen_low_freq_noise(array_shape, MY_KEY, l_bd=-1, u_bd=1, 
                       cutoff=0.15, r_cutoff=False, cutoff_bds=[0.02, 0.02])
    plt.imshow(noise, cmap='hot', interpolation='nearest')
    plt.show()