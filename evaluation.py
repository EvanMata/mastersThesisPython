

import os

import pandas as pd
import jax.numpy as jnp
import matplotlib.pyplot as plt

import optimization as opti
import pathlib_variable_names as my_vars

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
        pts_lambdas, pts = zip(*pts_and_lambdas)
        # cluster_pts[clus_l].append(pt) # Commented out to save memory
        # cluster_lambdas[clus_l].append(pt_lambda)
        clus_l = jnp.full(len(pts), clus)
        clustering.extend(zip(pts, clus_l))
        calc_lambdas.extend(zip(pts, pts_lambdas))
        
    clustering = sorted(clustering)
    clustering = [c[1] for c in clustering]
    calc_lambdas = sorted(calc_lambdas)
    calc_lambdas = [c[1] for c in calc_lambdas]
    
    return clustering, calc_lambdas, cluster_pts, cluster_lambdas


def load_true(data_name="my_data.pkl", cap=10000):
    #Loads the technically correct clusters
    df = pd.read_pickle("my_data.pkl")
    true_clusters = list(df['end state'])[:cap]
    return true_clusters


def find_useful_indices(data_name="my_data.pkl", thresh=0.8, follow_up=3, n_orbs=16, cap=10000):
    '''
    Some images were transitioning between two states. Find 
    the indices where start and end were the same.
    
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
    '''
    
    df = pd.read_pickle("my_data.pkl")
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
    
    return good_indices, close_indices, transitory_indices


def eval_clustering(my_gamma = 0.2, n_cs = 17, cap=10000, print_it=True, data_arr_path=my_vars.rawArraysF):
    """
    Calculates how many of the images were classified into the correct frames
    and some statistics on the lambdas of of my various groupings
    """
    true_clusters = load_true(cap=cap)
    names_and_data = load_data(data_arr_path, cap=cap)
    metric_val, lambdas_dict = opti.get_info(my_gamma, names_and_data, n_cs)
    calc_clusters, calc_lambdas, cluster_pts, cluster_lambdas = format_nice(lambdas_dict)
    good_indices, approx_indices, transitory_indices = find_useful_indices()
    
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
        print("True Eval:")
        print(exact_score)
        print("Approx Eval: ")
        print(approx_score)
        print()
        print("Exacts mean %f and stv %d of lambdas"%(l_g_mean, l_g_std))
        print("Approx mean %f and stv %d of lambdas"%(l_a_mean, l_a_std))
        print("Transitory mean %f and stv %d of lambdas"%(l_t_mean, l_t_std))
        print()
    
    return 


def load_data(data_arr_path=my_vars.rawArraysF, dtype=jnp.float16, cap=5000):
    """
    Loads all my arrays into a list of [(f name, array), (), ...]
    """
    names_and_data = []
    my_files = os.listdir(data_arr_path)
    i = 0
    for f in my_files:
        if str(f)[-4:] == ".npy" and i < cap:
            full_f = data_arr_path.joinpath( f )
            arr = jnp.load( full_f )
            arr = jnp.array(arr, dtype=dtype) #Reducing memory
            names_and_data.append((f, arr))
            i += 1
        
    return names_and_data


if __name__ == "__main__":
    eval_clustering(cap=100)
    