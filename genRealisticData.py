import jax 
import jax.numpy as jnp

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




if __name__ == "__main__":
    array_shape = (100,100)
    noise, f_noise, n_key = gen_low_freq_noise(array_shape, MY_KEY, l_bd=-1, u_bd=1, 
                       cutoff=0.1, r_cutoff=False, cutoff_bds=[0.02, 0.02])
    plt.imshow(noise, cmap='hot', interpolation='nearest')
    plt.show()