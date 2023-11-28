
import os
import cv2
import jax
import time
import pickle
import psutil

import pandas as pd
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

import my_metric as my_metric
import optimization as opti
import pathlib_variable_names as my_vars

from pathlib import Path
from itertools import product
from functools import partial
from collections import defaultdict
from sklearn.metrics import adjusted_rand_score


def format_nice(lambdas_dict):
    """
    Turns the lambdas_dict output into 2 dicts,
    one for clusters and the other for lambdas associated w. the cluster
    
    Inputs:
    --------
        lambdas_dict (dict) : dict of cluster num: [(img lambda, img num), (), ...]
    Returns:
    --------
        clustering (lst) : List of [cluster j of pt i for i in all imgs]
        calc_lambdas (lst) : List of [Lambda j of pt i for i in all imgs]
        cluster_pts (dict) : dict of cluster number to lst pts in that cluster,
                             lst indexed the same as that in cluster_lambdas
        cluster_lambdas (dict) : dict of cluster number to lst of pt lambdas 
                                 for pts in that cluster. Lambda i corresponds 
                                 to pt i in cluster_pt's list for the same key/clus
    """
    
    cluster_pts = defaultdict(list)
    cluster_lambdas = defaultdict(list)
    clustering = []
    calc_lambdas = []
    for clus, pts_and_lambdas in lambdas_dict.items():
        pts_lambdas, pts_indexs = zip(*pts_and_lambdas)
        # cluster_pts[clus_l].append(pt) # Commented out to save memory
        # cluster_lambdas[clus_l].append(pt_lambda)
        clus_l = jnp.full(len(pts_indexs), clus)
        clustering.extend(zip(pts_indexs, clus_l))
        calc_lambdas.extend(zip(pts_indexs, pts_lambdas))
        
    clustering = sorted(clustering)
    clustering = [c[1] for c in clustering]
    calc_lambdas = sorted(calc_lambdas)
    calc_lambdas = [c[1] for c in calc_lambdas]
    
    return clustering, calc_lambdas, cluster_pts, cluster_lambdas


def find_useful_indices(data_name="my_data.pkl", thresh=0.8, follow_up=3, n_orbs=16, cap=10000,
                        load_folder=my_vars.picklesDataPath):
    '''
    Finds the indices of frames where I'm in a state, close to a state, or transitioning 
    between states. Also, get a list of the expected end states.
    
    Inputs:
    --------
        data_name (str) : path to the true data
        thresh (float in [0,1)) : Percent of orbs that need to have arrived to their 
                                  destinations for an index to be considered close to 
                                  arrived
        follow_up (int) : How many of the next frames/indices to include in the close indices 
                          after a leaving a state
        n_orbs (int) : Number of orbs in the correct clustering
        cap (int) : How many datapoints to use

    Returns:
    --------
        good_indices (list) : indices of frames where start state = end state, eg where 
                              the image was fully in the state
        close_indices (list) : indices of frames/images where either the orbs were recently
                                in a state, or more than thresh % of orbs had made it to the
                                subsequent state
        transitory_indices (list) : indices of frames/images in niether good nor close indicies, 
                                    eg indices where frames were transitioning between states
        end_states (list) : List of states where item i is what state frame i was heading towards
    '''
    
    data_path = str(load_folder.joinpath("my_data.pkl"))

    df = pd.read_pickle(data_path)
    df = df.head(cap)
    end_states = list(df['end state'])
    start_states = list(df['start state'])
    good_indices = [i for i in range(len(end_states)) if \
                    end_states[i] == start_states[i]]
    
    
    arr_col_names = []
    for i in range(n_orbs):
        col_title = "%d arrived"%i
        df[col_title] = df[col_title]*1 #Makes True -> 1, False -> 0
        arr_col_names.append(col_title)
        
    # Indices where more than thresh % of orbs, but not 100%, 
    # have made it to their destinations
    df['perc arrived'] = df[arr_col_names].mean(axis=1)
    close_indices = df[(df['perc arrived'] > thresh) & (df['perc arrived'] < 1)]['j'].tolist()
    
    # Include follow-up # of frames/indices of frames after leaving a state
    for i in range(1, len(good_indices)):
        prev = good_indices[i-1]
        curr = good_indices[i]
        if curr - prev > 1:
            top_index = min([prev + 1 + follow_up, len(end_states) - 1])
            follow_ups = list(range(prev + 1, top_index)) 
            close_indices.extend(follow_ups)
    
    # All not-close and not-arrived indices
    transitory_indices = set(df['j'].tolist()) - set(close_indices).union(set(good_indices))
    transitory_indices = sorted(list(transitory_indices))
    
    return good_indices, close_indices, transitory_indices, end_states


def preview_df(rows=[1,9999], data_name="my_data.pkl", load_folder=my_vars.picklesDataPath):
    """
    Shows the given rows of the df 
    """
    data_path = str(load_folder.joinpath(data_name))
    df = pd.read_pickle(data_path)

    sub_df = df[df['j'].isin(rows)]
    print(sub_df.head(len(rows)))


def counts_in_states(data_name="my_data.pkl", load_folder=my_vars.picklesDataPath):
    data_path = str(load_folder.joinpath(data_name))
    df = pd.read_pickle(data_path)
    sub_df = df[df['end state'] == df['start state']]
    print("Total Counts of states: ")
    print(df['end state'].value_counts())
    print("Counts of number of frames IN each state")
    print(sub_df['end state'].value_counts())


def eval_clustering(my_gamma = 1.0, n_cs = 17, cap=10000, print_it=True, 
                    with_noise=True, noise_type='repl', simple_avg=False, 
                    only_states=False,
                    data_arr_path=my_vars.rawArraysF,
                    save_folder=my_vars.picklesDataPath):
    """
    Calculates how many of the images were classified into the correct frames
    and some statistics on the lambdas of of my various groupings
    """
    s = time.time()
    names_and_data = load_data(data_arr_path, cap=cap, 
                               with_noise=with_noise, noise_type=noise_type)
    aff_mat = load_affinity_matrix()
    good_indices, approx_indices, transitory_indices, true_clusters = find_useful_indices(cap=cap)
    if only_states:
        names_and_data = pure_states_only(names_and_data, good_indices)
        aff_mat = pure_aff_only(aff_mat, good_indices)
    e = time.time()
    print("Took %d seconds to load %d data pts and affinity matrix"%((e-s), cap))
    metric_val, lambdas_dict = opti.get_info(my_gamma, names_and_data, n_cs, simple_avg, aff_mat)

    lmd_name = "lambdas_d_gamma_%f.pickle"%my_gamma
    met_name = "metric_gamma_%f.pickle"%my_gamma

    lmd_name = str(save_folder.joinpath(lmd_name))
    met_name = str(save_folder.joinpath(met_name))

    with open(lmd_name, 'wb') as handle:
        pickle.dump(lambdas_dict, handle)

    with open(met_name, 'wb') as handle:
        pickle.dump(metric_val, handle)

    calc_clusters, calc_lambdas, cluster_pts, cluster_lambdas = format_nice(lambdas_dict)
    
    t_clus_goods = [t for i, t in enumerate(true_clusters) if i in good_indices]
    if not only_states:
        c_clus_goods = [t for i, t in enumerate(calc_clusters) if i in good_indices]
    else:
        c_clus_goods = calc_clusters
    lambdas_good = jnp.array([l for i, l in enumerate(calc_lambdas) if i in good_indices])
    l_g_mean, l_g_std = jnp.mean(lambdas_good), jnp.std(lambdas_good)
    exact_score = adjusted_rand_score(t_clus_goods, c_clus_goods)

    if not only_states:
        t_clus_approx = [t for i, t in enumerate(true_clusters) if i in approx_indices]
        c_clus_approx = [t for i, t in enumerate(calc_clusters) if i in approx_indices]
    
        lambdas_approx = jnp.array([l for i, l in enumerate(calc_lambdas) if i in approx_indices])
        lambdas_transitory = jnp.array([l for i, l in enumerate(calc_lambdas) if i in transitory_indices])
        l_a_mean, l_a_std = jnp.mean(lambdas_approx), jnp.std(lambdas_approx)
        l_t_mean, l_t_std = jnp.mean(lambdas_transitory), jnp.std(lambdas_transitory)
        approx_score = adjusted_rand_score(t_clus_approx, c_clus_approx)

    for clus, info in lambdas_dict.items():
        vis_clus(my_gamma, clus, lambdas_dict, simple_avg, names_and_data, only_states,
             save_folder=my_vars.stateImgsP, array_shape=(120,120))

    if print_it:
        print("GAMMA: ")
        print(my_gamma)
        print("METRIC VALUE: ")
        print(metric_val)
        print("True Eval:")
        print(exact_score)
        if not only_states:
            print("Approx Eval: ")
            print(approx_score)
        print()
        print("Exacts mean %f and stv %f of lambdas"%(l_g_mean, l_g_std))
        if not only_states:
            print("Approx mean %f and stv %f of lambdas"%(l_a_mean, l_a_std))
            print("Transitory mean %f and stv %f of lambdas"%(l_t_mean, l_t_std))
        print()
    
    return 


def multi_eval_clus_compare(gammas=[0.001, 0.01, 0.05, 0.25, 1, 4, 20], 
                            n_cs=[17, 18, 19, 20, 21, 22], 
                            n_cs_os=[17, 18, 19, 20],
                            only_os=False, load_prev=False):
    """
    Compare multiple clusterings all at once.
    """
    data_arr_path=my_vars.rawArraysF
    save_folder=my_vars.picklesDataPath
    with_noise=True
    noise_type='repl'
    cap=10000

    if load_prev:
        df = load_comparisons()
    else:
        df = pd.DataFrame(columns=['Gamma', 'Num Clusters', 'Only States', 'Metric', 'Rand Score'])

    names_and_data = load_data(data_arr_path, cap=cap, 
                               with_noise=with_noise, noise_type=noise_type)
    aff_mat = load_affinity_matrix()
    good_indices, approx_indices, transitory_indices, true_clusters = find_useful_indices(cap=cap)
    
    names_and_data_os = pure_states_only(names_and_data, good_indices)
    aff_mat_os = pure_aff_only(aff_mat, good_indices)

    print()
    print()
    print("ONLY STATES:")
    print()
    print()

    i = 0
    for n_c in n_cs_os:
        for g in gammas:
            mask = (df['Gamma']==g) & (df['Num Clusters'] == n_c) & (df['Only States'] == True)
            if mask.sum() == 0:
                metric_val, lambdas_dict = opti.get_info(g, names_and_data_os, n_c, True, aff_mat_os)
                for clus, info in lambdas_dict.items():
                    vis_clus(g, clus, lambdas_dict, True, names_and_data, True, 
                        save_folder=my_vars.stateImgsP, array_shape=(120,120))
                    
                calc_clusters, calc_lambdas, cluster_pts, cluster_lambdas = format_nice(lambdas_dict)

                t_clus_goods = [t for i, t in enumerate(true_clusters) if i in good_indices]
                c_clus_goods = calc_clusters
                rand_s = adjusted_rand_score(t_clus_goods, c_clus_goods)

                df.loc[len(df.index)] = [g, n_c, True, metric_val, rand_s]
                df.to_pickle("Clustering_Comparisons3.pickle")
                i += 1
                print(df.head(i))

    if not only_os:
        print()
        print()
        print("INCLUDING TRANSITORY")
        print()
        print()

        for n_c in n_cs:
            for g in gammas:
                mask = (df['Gamma']==g) & (df['Num Clusters'] == n_c) & (df['Only States'] == False)
                if mask.sum() == 0:
                    metric_val, lambdas_dict = opti.get_info(g, names_and_data, n_c, True, aff_mat)
                    for clus, info in lambdas_dict.items():
                        vis_clus(g, clus, lambdas_dict, True, names_and_data, False, 
                            save_folder=my_vars.stateImgsP, array_shape=(120,120))
                        
                    calc_clusters, calc_lambdas, cluster_pts, cluster_lambdas = format_nice(lambdas_dict)

                    t_clus_goods = [t for i, t in enumerate(true_clusters) if i in good_indices]
                    c_clus_goods = [t for i, t in enumerate(calc_clusters) if i in good_indices]
                    rand_s = adjusted_rand_score(t_clus_goods, c_clus_goods)

                    df.loc[len(df.index)] = [g, n_c, False, metric_val, rand_s]
                    df.to_pickle("Clustering_Comparisons3.pickle")
                    i += 1
                    print(df.head(i))


def load_comparisons(f_name="Clustering_Comparions2.pickle", folder=my_vars.picklesDataPath):
    f = folder.joinpath(f_name)
    with open(f, 'rb') as my_f:
        df = pickle.load(my_f)
    return df


def pure_aff_only(aff_mat, good_indices):
    # Gets a principle submatrix including only good_indices rows & cols
    inds = jnp.array(good_indices)
    return aff_mat[jnp.ix_(inds,inds)]


def pure_states_only(names_and_data, good_indices):
    """
    Returns names and data, but only where its in a pure state and 
    not transitioning
    """
    return [nd for i, nd in enumerate(names_and_data) if i in good_indices]


def vis_and_report_clustering(names_and_data, my_gamma=50.0, simple_avg=True, 
                            load_folder=my_vars.picklesDataPath, print_it=True):
    """
    Same as vis_clus but loads in results and then calls it.
    """

    lmd_name = "lambdas_d_gamma_%f.pickle"%my_gamma
    met_name = "metric_gamma_%f.pickle"%my_gamma

    lmd_name = str(load_folder.joinpath(lmd_name))
    met_name = str(load_folder.joinpath(met_name))

    with open(lmd_name, 'rb') as handle:
        lambdas_dict = pickle.load(handle)

    with open(met_name, 'rb') as handle:
        metric_val = pickle.load(handle)

    calc_clusters, calc_lambdas, cluster_pts, cluster_lambdas = format_nice(lambdas_dict)
    good_indices, approx_indices, transitory_indices, true_clusters = find_useful_indices(cap=cap)
    
    t_clus_goods = [t for i, t in enumerate(true_clusters) if i in good_indices]
    c_clus_goods = [t for i, t in enumerate(calc_clusters) if i in good_indices]
    t_clus_approx = [t for i, t in enumerate(true_clusters) if i in approx_indices]
    c_clus_approx = [t for i, t in enumerate(calc_clusters) if i in approx_indices]
    
    lambdas_good = jnp.array([l for i, l in enumerate(calc_lambdas) if i in good_indices])
    lambdas_approx = jnp.array([l for i, l in enumerate(calc_lambdas) if i in approx_indices])
    lambdas_transitory = jnp.array([l for i, l in enumerate(calc_lambdas) if i in transitory_indices])
    l_g_mean, l_g_std = jnp.mean(lambdas_good), jnp.std(lambdas_good)
    l_a_mean, l_a_std = jnp.mean(lambdas_approx), jnp.std(lambdas_approx)
    l_t_mean, l_t_std = jnp.mean(lambdas_transitory), jnp.std(lambdas_transitory)
    
    exact_score = adjusted_rand_score(t_clus_goods, c_clus_goods)
    approx_score = adjusted_rand_score(t_clus_approx, c_clus_approx)
    if print_it:
        print("GAMMA: ")
        print(my_gamma)
        print("METRIC VALUE: ")
        print(metric_val)
        print("True Eval:")
        print(exact_score)
        print("Approx Eval: ")
        print(approx_score)
        print()
        print("Exacts mean %f and stv %f of lambdas"%(l_g_mean, l_g_std))
        print("Approx mean %f and stv %f of lambdas"%(l_a_mean, l_a_std))
        print("Transitory mean %f and stv %f of lambdas"%(l_t_mean, l_t_std))
        print()

    for clus, info in lambdas_dict.items():
        vis_clus(my_gamma, clus, lambdas_dict, simple_avg, names_and_data, 
             save_folder=my_vars.stateImgsP, array_shape=(120,120))


def load_data(data_arr_path=my_vars.rawArraysF, dtype=jnp.float16, cap=10000,
              with_noise=False, noise_path=my_vars.rawNoiseF, noise_type="mult",
              c_key=jax.random.PRNGKey(0), start_val=0):
    """
    Loads all my arrays into a list of [(f name, array), (), ...]
    names and data is lst of tuples
    """
    n_key = c_key
    for k in range(start_val):
        n_key, subkey = jax.random.split(c_key)
    names_and_data = []
    my_arrs = os.listdir(data_arr_path)
    if with_noise:
        my_noises = os.listdir(noise_path)
    i = start_val
    cond = (with_noise and noise_type.lower().strip() == 'mult') or (not with_noise)
    if cond:
        for j in range(start_val, len(my_arrs)):
            f = my_arrs[j]
            if str(f)[-4:] == ".npy" and i < cap:
                full_f = data_arr_path.joinpath( f )
                arr = jnp.load( full_f )
                arr = jnp.array(arr, dtype=dtype) #Reducing memory
                if with_noise:
                    n = my_noises[j]
                    full_n = noise_path.joinpath( n )
                    noise_arr = jnp.load( full_n )
                    noise_arr = jnp.array(noise_arr, dtype=dtype)
                    if noise_type.lower().strip() == 'mult':
                        arr = combine_signal_noise_mult(arr, noise_arr)

                names_and_data.append((f, arr))
                i += 1

    elif noise_type.lower().strip() == 'repl' and with_noise:
        for j in range(start_val, len(my_arrs)):
            f = my_arrs[j]
            if str(f)[-4:] == ".npy" and i < cap:
                repl_load_name = my_vars.replNoiseP%j + ".npy"
                arr = jnp.load(repl_load_name)

                names_and_data.append((f, arr))
                i += 1

    return names_and_data


#Can be deleted eventually
def lmd_dict_vs_all_clus(my_gamma=50.0, load_folder=my_vars.picklesDataPath, cap=10000):
    intended_inds = set(list(range(cap)))
    lmd_name = "lambdas_d_gamma_%f.pickle"%my_gamma

    lmd_name = str(load_folder.joinpath(lmd_name))

    with open(lmd_name, 'rb') as handle:
        lambdas_dict = pickle.load(handle)

    all_inds = []
    for clu, lmds_and_inds in lambdas_dict.items():
        lmds, inds = zip(*lmds_and_inds)
        inds = jnp.array(inds).reshape(-1)
        all_inds.extend(inds.tolist())

    print(intended_inds - set(all_inds))
    print(set(all_inds) - intended_inds)
    print(len(all_inds))


@jax.jit
def combine_signal_noise_mult(signal_arr, noise_arr):
    scaled_noise_arr = my_metric.scale01(noise_arr)
    signal_arr = signal_arr - 0.5
    composite_arr = signal_arr*scaled_noise_arr
    composite_arr = composite_arr + 0.5 #Back to [0,1] in theory
    return composite_arr


#@partial(jax.jit, static_argnames=['repl_perc'])
def combine_signal_noise_repl(c_key, signal_arr, noise_arr, repl_perc=0.25,
                              bottom=0.2, top=0.8):
    """
    Replaces repl_perc of the signal array with the noise array values randomly
    """
    n_key, subkey = jax.random.split(c_key)
    valid_indices = jnp.array(list(range(signal_arr.size))) #inds of indices
    n_samples = int(repl_perc*signal_arr.size)
    chosen_indices = jax.random.choice(subkey, valid_indices, 
                                       shape=(n_samples,), replace=False)
    xy_inds = jnp.array(list(product( \
        jnp.arange(signal_arr.shape[0]), jnp.arange(signal_arr.shape[1]))))
    
    xs_sub, ys_sub = zip(*xy_inds[chosen_indices])
    scaled_noise_arr = my_metric.scale01(noise_arr)
    #combined_arr = repl_arr(signal_arr, scaled_noise_arr, xs_sub, ys_sub)
    combined_arr = repl_arr2(signal_arr, scaled_noise_arr, xs_sub, ys_sub, bottom, top)
    return n_key, combined_arr


#@jax.jit
def repl_arr(signal_arr, scaled_noise_arr, xs_sub, ys_sub):
    scaled_noise_arr = scaled_noise_arr.at[xs_sub, ys_sub].set(signal_arr[xs_sub, ys_sub])
    return scaled_noise_arr


def repl_arr2(signal_arr, scaled_noise_arr, xs_sub, ys_sub, bottom, top):
    signal_arr, scaled_noise_arr, xs_sub, ys_sub = \
        np.array(signal_arr), np.array(scaled_noise_arr), xs_sub, ys_sub
    scaled_noise_arr[xs_sub, ys_sub] = signal_arr[xs_sub, ys_sub]
    
    scaled_noise_arr[scaled_noise_arr < bottom] = 0
    scaled_noise_arr[scaled_noise_arr > top] = 1

    scaled_noise_arr = cv2.blur(scaled_noise_arr.astype(float),(3,3),cv2.BORDER_REPLICATE) #Blur?

    scaled_noise_arr = jnp.array(scaled_noise_arr)
    return scaled_noise_arr


def load_affinity_matrix(gamma=jnp.array([1.0]), load_folder=my_vars.picklesDataPath):
    """
    Loads a preconstructed affinity matrix
    """
    digit3_gamma = '{0:.3f}'.format(float(gamma))
    digit3_gamma = digit3_gamma.replace('.', "_")
    affinity_mat_save_name = str(load_folder.joinpath(\
                                "Affinity_Mat_gamma_%s.npy"%digit3_gamma))
    affinity_mat = jnp.load(affinity_mat_save_name)
    return affinity_mat


def save_noise_arr(j=10000, data_arr_path=my_vars.rawArraysF, dtype=jnp.float16,
                   noise_path=my_vars.rawNoiseF, c_key=jax.random.PRNGKey(0)):
    """
    Create and save a noise array from the given j'th base array and noise array
    """
    n_key = c_key
    for k in range(j):
        n_key, subkey = jax.random.split(c_key)

    my_arrs = os.listdir(data_arr_path)
    my_noises = os.listdir(noise_path)
    f = my_arrs[j]
    if str(f)[-4:] == ".npy":
        full_f = data_arr_path.joinpath( f )
        arr = jnp.load( full_f )
        arr = jnp.array(arr, dtype=dtype)
        n = my_noises[j]
        full_n = noise_path.joinpath( n )
        noise_arr = jnp.load( full_n )

        n_key, combo_arr = combine_signal_noise_repl(n_key, arr, noise_arr, repl_perc=0.25,
                              bottom=0.2, top=0.8)
    
        save_name = my_vars.replNoiseP%j
        jnp.save(save_name, combo_arr)


def vis_sum_noise(noise_path=my_vars.rawNoiseF):
    """
    Creates a visual of the sum of my noise
    """
    my_noises = os.listdir(noise_path)
    noise_tot = jnp.zeros((120,120))
    for j in range(len(my_noises)):
        n = my_noises[j]
        full_n = noise_path.joinpath( n )
        noise_arr = jnp.load( full_n )
        noise_tot += noise_arr

    noise_tot = noise_tot / len(my_noises)
    print(jnp.min(noise_tot), jnp.max(noise_tot))
    plt.imshow(noise_tot, cmap='hot', interpolation='nearest')
    plt.savefig("Averaged Noise")


def vis_clus(gamma, clus, lambdas_dict, simple_avg, names_and_data, 
             only_states, save_folder=my_vars.stateImgsP, 
             array_shape=(120,120), is_best=False):
    """
    Saves an image of the cluster to the given save folder
    """
    digit3_gamma = '{0:.3f}'.format(float(gamma))
    digit3_gamma = digit3_gamma.replace('.', "-")
    new_dir = my_vars.stateImsF
    num_clus = sum([1 for k,v in lambdas_dict.items()])
    name_add = str(int(num_clus))+"clus_" + "gamma" + digit3_gamma
    if only_states:
        name_add += "_OS"
    if simple_avg:
        name_add += "_simple"
    if is_best:
        name_add += "_best"
    new_dir = new_dir.joinpath(name_add)
    Path(str(new_dir)).mkdir(parents=True, exist_ok=True)
    
    #save_name = my_vars.stateImgsP%(clus, digit3_gamma)
    save_name = str(new_dir.joinpath( "state_%d" ))%(clus)

    img = jnp.zeros(array_shape)
    lambdas_and_indices = lambdas_dict[clus]
    lambdas, indices = zip(*lambdas_and_indices)
    for i in range(len(lambdas)):
        l = lambdas[i]
        im_ind = indices[i]
        im = names_and_data[im_ind][1]
        img += l*im

    img = my_metric.clip_img(img)
    img = my_metric.scale01(img)

    plt.imshow(img, cmap='Greys_r', interpolation='nearest')
    plt.colorbar()
    plt.savefig(save_name)
    plt.clf()


def algo_p1(n_cs = 18, only_pure_states=True):
    """
    Runs part 1 of my optimization algorithm, 
    """
    cap = 10000
    with_noise = True
    noise_type = 'repl'
    data_arr_path = my_vars.rawArraysF
    names_and_data = load_data(data_arr_path, cap=cap, 
                               with_noise=with_noise, noise_type=noise_type)
    aff_mat = load_affinity_matrix()
    good_indices, approx_indices, transitory_indices, true_clusters = find_useful_indices(cap=cap)
    if only_pure_states:
        names_and_data = pure_states_only(names_and_data, good_indices)
        aff_mat = pure_aff_only(aff_mat, good_indices)
    opti.gamma_tuning_simple_avg(n_cs, names_and_data, aff_mat, only_pure_states)


def algo_p2(n_cs, gamma, only_states=True):
    cap = 10000
    with_noise = True
    noise_type = 'repl'
    data_arr_path = my_vars.rawArraysF
    save_folder = my_vars.picklesDataPath
    digit3_gamma = '{0:.3f}'.format(float(gamma))
    digit3_gamma = digit3_gamma.replace('.', "-")

    names_and_data = load_data(data_arr_path, cap=cap, 
                               with_noise=with_noise, noise_type=noise_type)
    aff_mat = load_affinity_matrix()
    good_indices, approx_indices, transitory_indices, true_clusters = find_useful_indices(cap=cap)
    if only_states:
        names_and_data = pure_states_only(names_and_data, good_indices)
        aff_mat = pure_aff_only(aff_mat, good_indices)
    metric_val, lambdas_dict = opti.get_info(gamma, \
            names_and_data, n_cs, simple_avg=False, premade_affinity_matrix=aff_mat)
    
    lmd_name = "best_lambdas_d_gamma_%s.pickle"%digit3_gamma
    met_name = "best_metric_gamma_%s.pickle"%digit3_gamma

    lmd_name = str(save_folder.joinpath(lmd_name))
    met_name = str(save_folder.joinpath(met_name))

    with open(lmd_name, 'wb') as handle:
        pickle.dump(lambdas_dict, handle)

    with open(met_name, 'wb') as handle:
        pickle.dump(metric_val, handle)

    for clus, info in lambdas_dict.items():
        vis_clus(gamma, clus, lambdas_dict, False, names_and_data, only_states,
             save_folder=my_vars.stateImgsP, array_shape=(120,120), is_best=True)



if __name__ == "__main__":
    #multi_eval_clus_compare()
    """
    s = time.time()
    cap=10000
    eval_clustering(cap=cap, simple_avg=True, with_noise=True, n_cs=17, 
                    only_states=True, noise_type='repl', my_gamma = 15.0)
    e = time.time()
    print("Time taken for cap = %d: "%cap, e - s)
    
    #save_noise_arr(j=10000, data_arr_path=my_vars.rawArraysF, dtype=jnp.float16,
    #               noise_path=my_vars.rawNoiseF, c_key=jax.random.PRNGKey(0))
    """
    #aff = load_affinity_matrix()
    #print(aff.shape)
    """
    cap=10000
    data_arr_path = my_vars.rawArraysF
    with_noise=True
    noise_type='repl'
    names_and_data = load_data(data_arr_path, cap=cap, 
                               with_noise=with_noise, noise_type=noise_type)
    vis_and_report_clustering(names_and_data, my_gamma=50.0, simple_avg=True, \
                            load_folder=my_vars.picklesDataPath, print_it=True)
    """
    #counts_in_states()
    #lmd_dict_vs_all_clus()
    #algo_p1(n_cs = 18)
    #algo_p2(n_cs=18, gamma=0.25, only_states=True)
    multi_eval_clus_compare(gammas=[0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,\
        0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09, \
        0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1,2,3,4,5,6,7,8,9,10], 
                            n_cs=[19, 20, 21, 22], 
                            n_cs_os=[],
                            only_os=False, load_prev=True)