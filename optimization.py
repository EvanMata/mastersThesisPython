
import os 
import jax
import time
import pickle
import psutil

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from sklearn.cluster import SpectralClustering

from jax.nn import softmax
from itertools import combinations
from functools import lru_cache, partial
from jax.scipy.optimize import minimize as minimize
from scipy.optimize import minimize as min2

import my_metric as my_metric
import openRawData as opn
import genSyntheticData as dg
import pathlib_variable_names as var_names

#################
# Global Params #
#################

KEY = jax.random.PRNGKey(23)
DEFAULT_METRIC = my_metric.closest_sq_dist_tv_mix


def convComb1Clust(convexComboParams, realSpaceImgs, metric=DEFAULT_METRIC):
    # Should use some args/kwargs stuff here.
    """
    Given a list of images and their corresponding lambdas representing some
    convex combination of them, evaluate the provided metric on them
    """
    convexComboParams = softmax(jnp.array(convexComboParams))
    convArrays = [convexComboParams[i]*realSpaceImgs[i] for i in range(len(realSpaceImgs))]
    imageToEval = sum(convArrays)
    clusterMetric = metric(imageToEval)
    return clusterMetric


#############
# Main Code #
#############


def basic_ex(my_gamma=0.1, use_new_data=True, n_cs=5, arraysPerCluster=3, solver="BFGS", 
             provided_data=None, simple_avg=False, premade_affinity_matrix=None):
    """
    Triest to run a basic clustering example on synthetic data.

    Inputs:
    --------
        my_gamma (float) : Value of gamma to use for the tunning.
        use_new_data (bool) :    False use pre-generated data, or
                                 True generate new data
        n_cs (int) :             Number of clusters to fit (and generate if using use_new_data=True)
        arraysPerCluster (int) : Number of fake data pts to generate in each cluster
        solver (str) :           What solver to use (only one works in jax at the moment - BFGS)
        provided_data (lst) :    A list of tuples of (cluster, data point)
        simple_avg (bool) : Whether to use convex combination or simple averaging.
        premade_affinity_matrix (jnp array|None) : If not none, a precomputed affinity matrix.
    Returns: 
    --------
        metric_val (float) : Value of my metric for the given clustering 
    """
    if use_new_data:
        names_and_data = dg.genClusters(numClusters=n_cs, arraysPerCluster=arraysPerCluster, n1=100, n2=100,
                                        noiseLv=0.25, display=False, saveClusters=False)
    else:
        names_and_data = provided_data

    names = [i[0] for i in names_and_data] #Unused
    data = [i[1] for i in names_and_data]

    data = jnp.array(data)
    images_tup = totuple(data)
    metric_val, lambdas_dict = const_gamma_clustering(gamma=my_gamma, images_tup=images_tup, 
                                                        n_clusters=n_cs, simple_avg=simple_avg,
                                                        premade_affinity_matrix=premade_affinity_matrix)
    
    #print("Gamma: ", my_gamma)
    #print("Metric Value: ", metric_val)
    #print("Lambdas: ")
    #print(lambdas_dict)
    #print()
    
    metric_val = float(metric_val)
    print(metric_val)

    return metric_val


def get_info(my_gamma, provided_data, n_cs, simple_avg=False, premade_affinity_matrix=None):
    """
    Same as basic_clustering, but returns my lambdas dict
    
    Inputs:
    --------
        my_gamma (float) : Value of gamma to use for the tunning.
        provided_data (lst) :    A list of tuples of (cluster, data point)
        n_cs (int) :             Number of clusters to fit
        simple_avg (bool) : Whether to use convex combination or simple averaging.
        premade_affinity_matrix (jnp array|None) : If not none, a precomputed affinity matrix.
    Returns: 
    --------
        metric_val (float) : Value of my metric for the given clustering 
        lamdbas_dict (dict) : dict of cluster_label: zip[(lambda_i, i of corresponding img), (), ...]
            Where img_i is in the given cluster
    """
    
    data = [i[1] for i in provided_data]

    data = jnp.array(data)
    # images_tup = totuple(data) # Takes a while, no longer cachine so irrelevant
    ## EDIT ## ADJUST TO KWARGS OR ARGS
    metric_val, lambdas_dict = const_gamma_clustering(gamma=my_gamma, images_tup=data, 
                                                        n_clusters=n_cs, simple_avg=simple_avg,
                                                        premade_affinity_matrix=premade_affinity_matrix)
    
    return metric_val, lambdas_dict


def gamma_tuning_ex(use_new_data=True, n_cs=5, arraysPerCluster=3, my_method="BFGS",
                    provided_data=None, premade_affinity_matrix=None):
    """
    Tunes the gamma used in my clustering.
    """
    
    fun = basic_ex
    init_guess = jnp.array([1.0])
    arguements = (use_new_data, n_cs, arraysPerCluster, my_method, \
                  provided_data, premade_affinity_matrix)
    s = time.time()
    minResults = min2(fun=fun, x0 = init_guess,
                            method=my_method, args=arguements)
    e = time.time()
    min_metric_value = minResults.fun
    gamma_value = minResults.x
    print("Optimal Alpha Value was: ", gamma_value)
    total_time = e - s
    print("Total Time: ", total_time)


def gamma_tuning_simple_avg(n_cs, provided_data, premade_affinity_matrix,
                            only_pure_states, save_items=False):
    """
    Using simple averaging, tunes my affinity matrix by gamma parameterization.
    """
    
    fun = basic_ex
    init_guess = jnp.array([1.0])
    arguements = (False, n_cs, 3, "BFGS", provided_data, False, premade_affinity_matrix)
    s = time.time()
    minResults = min2(fun=fun, x0 = init_guess,
                            method="BFGS", args=arguements, 
                            options={'return_all':True})
    e = time.time()
    min_metric_value = minResults.fun
    gamma_value = minResults.x
    print("Optimal Gamma Value was: ", gamma_value)
    total_time = e - s
    print("Total Time: ", total_time)

    save_folder = var_names.picklesDataPath
    pickle_save_name = str(int(n_cs))+"_optimized_gamma"
    if only_pure_states:
        pickle_save_name += "_OS"
    pickle_save_name += ".pickle"
    save_name = str(save_folder.joinpath(pickle_save_name))
    if save_items:
        with open(save_name, 'wb') as handle:
            pickle.dump(gamma_value, handle)

    pickle_save_name2 = "min_results_" + str(int(n_cs))
    if only_pure_states:
        pickle_save_name2 += "_OS"
    pickle_save_name2 += ".pickle"
    save_name2 = str(save_folder.joinpath(pickle_save_name2))
    if save_items:
        with open(save_name2, 'wb') as handle:
            pickle.dump(minResults, handle)


def const_gamma_clustering(gamma, images_tup, n_clusters, simple_avg=False, 
                           premade_affinity_matrix=None, print_timing=False):
    """
    Clustering given my gamma and params. This is what I want to minimize for 
    a given gamma.

    Inputs:
    --------
        gamma (float) : Gaussian Affinity Kernel param to test
        images_tup (tuple of tuples of tuples) : Set of images being clustered
        n_clusters (int) : number of clusters to test 
        simple_avg (bool) : Whether to use convex combination or simple averaging.
        premade_affinity_matrix (jnp array|None) : If not none, a precomputed affinity matrix.
    Returns:
    --------
        total_metric_value (float) :  The sum of my metric value across each cluster
                                      for the given gamma value
        lamdbas_dict (dict) : dict of cluster_label: zip[(lambda_i, i of corresponding img), (), ...]
            Where img_i is in the given cluster
    """
    images = jnp.array(images_tup)
    gamma = jnp.array([gamma])
    if premade_affinity_matrix is not None:
        s1 = time.time()
        affinities = premade_affinity_matrix**gamma # Original gamma was 1.0, so **gamma/1
        e1 = time.time()
    else:
        s1 = time.time()
        affinities = affinity_matrix3(images, gamma)
        e1 = time.time()
        if print_timing:
            print("Affinity Time: ", e1 - s1)
    clusters = performClustering(affinities, n_clusters)
    e2 = time.time()
    if print_timing:
        print("Clustering Time: ", e2 - e1)
    # clusters = clustering_to_cachable_labels(clusters, n_clusters) 
    # If caching doesn't work, don't use this
    clusters = tuple(clusters)
    e3 = time.time()
    if print_timing:
        print("Making Labels cachable: ", e3 - e2)
    e4 = time.time()
    total_metric_value, lambdas_dict = calc_clusters_lambdas_cached(clusters, images_tup,
                                                                    simple_avg)
    e5 = time.time()
    if print_timing:
        print("Performing Convex Minimization and Calc Metric: ", e5 - e4)
    return total_metric_value, lambdas_dict


def totuple(m):
    #makes m into a tuple of tuples of tuples all the way down
    if m.shape == ():
        return m.item()
    else:
        return tuple(map(totuple, m))


def calcPairAffinity(image1, image2, gamma): #Shouldn't I jit this? Or did that cause issues w. gamma?
    #Returns a jnp array of 1 float, jnp.sum adds all elements together
    diff = jnp.sum(jnp.abs(image1 - image2))  
    normed_diff = diff / image1.size
    val = jnp.exp(-gamma*normed_diff)
    val = val.astype(jnp.float16)
    return val


#@partial(jax.jit, static_argnames=['gamma'])
@jax.jit
def calcPairAffinity2(ind1, ind2, imgs, gamma):
    #Returns a jnp array of 1 float, jnp.sum adds all elements together
    image1, image2 = imgs[ind1], imgs[ind2]
    diff = jnp.sum(jnp.abs(image1 - image2))  
    normed_diff = diff / image1.size
    val = jnp.exp(-gamma*normed_diff)
    val = val.astype(jnp.float16)
    return val


def affinity_matrix2(arr_of_imgs, gamma=jnp.array([0.5]), \
                      pair_affinity_func=calcPairAffinity2, 
                      pair_affinity_parallel_axes=(0, 0, None, None),
                      batch_size=5000, print_progress=True):
    """
    Creates my affininty matrix, v-mapped.

    Inputs:
    --------
        arr_of_imgs (3d jnp array of 2d tuples) : jnp arr of imgs (a x b tuples)
        gamma (1d jnp array) : parameterizing the pair_affinity_func
        pair_affinity_func (func) : function which takes in 2 images, gamma, 
            and outputs the affinity between the two images
        pair_affinity_parallel_axes (tup) : see vmap for more info, the input axes 
            which are being parallelized over. 

    Returns: 
    --------
        arr (jnp array) : Array of pair affinities, item i,j is the affinity 
                          between imgs i and j
    """
    arr_of_imgs = jnp.array(arr_of_imgs)
    n_imgs = len(arr_of_imgs)
    arr_of_indices = jnp.arange(n_imgs)
    
    arr = jnp.zeros((n_imgs, n_imgs), dtype=jnp.float16)
    inds_1, inds_2 = zip(*combinations(arr_of_indices, 2))
    v_cPA = jax.vmap(pair_affinity_func, pair_affinity_parallel_axes, 0)
    
    all_affinities = []
    num_combos = len(inds_1)
    n_batches = int(num_combos / batch_size)
    if num_combos % batch_size != 0:
        n_batches += 1 #Round up
    
    time_taken = 0
    for i in range(n_batches):
        s = time.time()
        top = min([num_combos, (i+1)*batch_size])
        batch_inds = jnp.arange(i*batch_size, top)
        batch_inds_1, batch_inds_2 = jnp.array(inds_1)[batch_inds], jnp.array(inds_2)[batch_inds]
        affinities = v_cPA(jnp.array(batch_inds_1), jnp.array(batch_inds_2), arr_of_imgs, gamma)
        affinities = list(affinities)
        all_affinities.extend(affinities)
        e = time.time()
        # Chunk Affinities, and keep track of what chunk you were on.
        affinity_save_name = "Affinity_%f.pickle"%gamma
        with open(affinity_save_name, 'wb') as handle:
            pickle.dump((i, all_affinities), handle)
        if print_progress:
            batch_time_taken = e - s
            time_taken += batch_time_taken
            avg_batch_len = time_taken / (i+1)
            expected_duration = avg_batch_len*(n_batches + 1)
            print(batch_time_taken)
            print("Finished batch %d of %d"%(i+1, n_batches))
            print("Time taken for 1 batch of size %d: %d"%(batch_size, batch_time_taken))
            print("Estimated Total Time: %f"%expected_duration)
            print("Estimated Remaining Time: %f"%(expected_duration - time_taken))
        
        
    
    all_affinities = jnp.array(all_affinities)
    all_affinities = all_affinities.reshape(-1)
    arr = arr.at[jnp.triu_indices(arr.shape[0], k=1)].set(all_affinities)
    arr = arr + arr.T
    arr = arr + jnp.identity(n_imgs, dtype=jnp.float16)
    print()
    print("VMAP WORKED")
    print()
    return arr


def affinity_matrix3(arr_of_imgs, gamma=jnp.array([1.0]), \
                      pair_affinity_func=calcPairAffinity2, 
                      pair_affinity_parallel_axes=(0, 0, None, None),
                      batch_size=5000, print_progress=True, 
                      pickup=False, pickup_loc=49980000, 
                      save_folder=var_names.affinitiesPath,
                      save_items=False):
    """
    Creates my affininty matrix, v-mapped.

    Inputs:
    --------
        arr_of_imgs (3d jnp array of 2d tuples) : jnp arr of imgs (a x b tuples)
        gamma (1d jnp array) : parameterizing the pair_affinity_func
        pair_affinity_func (func) : function which takes in 2 images, gamma, 
            and outputs the affinity between the two images
        pair_affinity_parallel_axes (tup) : see vmap for more info, the input axes 
            which are being parallelized over. 

    Returns: 
    --------
        arr (jnp array) : Array of pair affinities, item i,j is the affinity 
                          between imgs i and j
    """
    digit3_gamma = '{0:.3f}'.format(float(gamma))
    digit3_gamma = digit3_gamma.replace('.', "-")
    affinity_mat_save_name = str(save_folder.joinpath(\
                                "Affinity_Mat_gamma_%s"%digit3_gamma))
    arr_of_imgs = jnp.array(arr_of_imgs)
    n_imgs = len(arr_of_imgs)
    arr_of_indices = jnp.arange(n_imgs)
    arr = jnp.zeros((n_imgs, n_imgs), dtype=jnp.float16)
    triu_inds = jnp.triu_indices(arr.shape[0], k=1)
    triu_1, triu_2 = triu_inds  
    inds_1, inds_2 = zip(*combinations(arr_of_indices, 2))

    n_combos = len(inds_1)
    all_affinities = []

    if pickup:
        to_do = range(pickup_loc, n_combos)
        past_batches = jnp.array(list(range(0, pickup_loc, batch_size))) + batch_size - 1
        past_batches = past_batches.tolist()
        for past_b in past_batches:
            affinities_load_name = str(save_folder.joinpath( \
                "Affinity_b_%d_gamma_%s.npy"%(past_b, digit3_gamma)))
            batch = jnp.load(affinities_load_name)
            batch = batch.tolist()
            all_affinities.extend(batch)
        
    else:
        to_do = range(n_combos) 

    last_batch = []
    for i in to_do:
        ind1, ind2 = inds_1[i], inds_2[i]
        aff = calcPairAffinity2(ind1, ind2, arr_of_imgs, gamma)
        all_affinities.append(aff)
        last_batch.append(aff)
        
        if i % batch_size == batch_size - 1:
            last_b_arr = jnp.array(last_batch)
            affinities_save_name = str(save_folder.joinpath( \
                                    "Affinity_b_%d_gamma_%s"%(i, digit3_gamma)))
            if save_items:
                jnp.save(affinities_save_name, last_b_arr)
            last_batch = []
            start = i - batch_size + 1
            batch_inds = (triu_1[start:i+1], triu_2[start:i+1])
            arr = arr.at[batch_inds].set(last_b_arr.reshape(-1))
            print("On Affinity pair %d of %d"%(i, n_combos))
    
    arr = arr + arr.T
    arr = arr + jnp.identity(n_imgs, dtype=jnp.float16)

    print("Affinites Matrix shape: ", arr.shape)
    if save_items:
        jnp.save(affinity_mat_save_name, arr)
    print()
    print("VMAP WORKED")
    print()
    return arr


def affinity_matrix(arr_of_imgs, gamma=jnp.array([0.5]), \
                      pair_affinity_func=calcPairAffinity, 
                      pair_affinity_parallel_axes=(0, 0, None)):
    """
    Creates my affininty matrix, v-mapped.

    Inputs:
    --------
        arr_of_imgs (3d lst of jnp array) : lst of imgs (a x b jnp arrays)
        gamma (1d jnp array) : parameterizing the pair_affinity_func
        pair_affinity_func (func) : function which takes in 2 images, gamma, 
            and outputs the affinity between the two images
        pair_affinity_parallel_axes (tup) : see vmap for more info, the input axes 
            which are being parallelized over. 

    Returns: 
    --------
        arr (jnp array) : Array of pair affinities, item i,j is the affinity 
                          between imgs i and j
    """
    arr_of_imgs = jnp.array(arr_of_imgs)
    n_imgs = len(arr_of_imgs)
    arr = jnp.zeros((n_imgs, n_imgs), dtype=jnp.float16)
    print("Bytes Size of Base Array: ", arr.nbytes)
    v_cPA = jax.vmap(pair_affinity_func, pair_affinity_parallel_axes, 0)
    imgs_1, imgs_2 = zip(*combinations(arr_of_imgs,2))
    print("Bytes Size of Imgs1 Array: ", jnp.array(imgs_1).nbytes)
    print("Size of Imgs1 Array: ", jnp.array(imgs_1).size)
    print("Usage Pre Affinity Matrix Fill In: ", usage())
    affinities = v_cPA(imgs_1, imgs_2, gamma)
    #affinities = v_cPA(jnp.array(imgs_1), jnp.array(imgs_2), gamma)
    affinities = affinities.reshape(-1)
    arr = arr.at[jnp.triu_indices(arr.shape[0], k=1)].set(affinities)
    arr = arr + arr.T
    arr = arr + jnp.identity(n_imgs, dtype=jnp.float16)
    return arr


def performClustering(affinities, n_clusters, print_it=False):
    """
    Spectral clustering with pre-computed affinity matrix.
    """
    if print_it:
        print()
        print("Affinities: ")
        print(affinities)
        print()
    clustering = SpectralClustering(n_clusters=n_clusters,
                                    affinity="precomputed",
                                    random_state=0).fit(affinities)
    clusters = clustering.labels_
    return clusters


def save_affinity_matrix(batch_size=5000, gamma=jnp.array([1.0]), n_imgs=10000,
                         batch_save_folder=var_names.affinitiesPath, 
                         affininty_folder=var_names.picklesDataPath):
    """
    Sets my affinity matrix in chunks w. my batches to hopefully preserve memory,
    then saves it.
    Unclear why affinity matrix takes up so much memory.

    Inputs:
    --------
        batch_size (int) : How many affinities are in 1 batch
        gamma (float) : The gamma value used to generate the affinity matrix
        n_imgs (int) : Number of imgs used for the simulation
        batch_save_folder (pathlib Path) : Path where batchs arrays were saved
        affinity_folder (pathlib Path) : Path to save the affinity matrix
    """
    
    digit3_gamma = '{0:.3f}'.format(float(gamma))
    digit3_gamma = digit3_gamma.replace('.', "_")
    affinity_mat_save_name = str(affininty_folder.joinpath(\
                                "Affinity_Mat_gamma_%s"%digit3_gamma))
    
    arr = jnp.zeros((n_imgs, n_imgs), dtype=jnp.float16)
    triu_inds = jnp.triu_indices(arr.shape[0], k=1)
    triu_1, triu_2 = triu_inds
    
    inds_1, inds_2 = zip(*combinations(jnp.arange(n_imgs), 2))
    n_combos = len(inds_1)
    past_batches = jnp.array(list(range(0, n_combos, batch_size))) + batch_size - 1
    past_batches = past_batches.tolist()
    for past_b in past_batches:
        affinities_load_name = str(batch_save_folder.joinpath( \
            "Affinity_b_%d_gamma_%s.npy"%(past_b, digit3_gamma)))
        batch = jnp.load(affinities_load_name).reshape(-1)
        print(batch)
        start = past_b - batch_size + 1
        batch_inds = (triu_1[start:past_b+1], triu_2[start:past_b+1])
        arr = arr.at[batch_inds].set(batch)
        print("Set Batch %d"%past_b)

    print("Affinity matrix mostly constructed")

    arr = arr + arr.T
    arr = arr + jnp.identity(n_imgs, dtype=jnp.float16)

    print(arr)

    print("Affinity matrix fully constructed")
    jnp.save(affinity_mat_save_name, arr)
    print("Affinity matrix saved")


##############################################################
# What is an elegant way to make the clusters name invarient #
##############################################################


#@jax.jit Maybe jitable but need to find workaround for .count
def clustering_to_cachable_labels(clusters, n_clusters):
    """
    Ensures that [0,0,1,1] & [1,1,0,0] or any other pair 
    of cluster labels that I recieve and are equivalent up to 
    the cluster labeling scheme all get mapped to the same thing.

    #Prob a better way to do this. Try to jit?

    Inputs:
    --------
        clusters (jnp array?) : 
        n_clusters (int) : number of clusters to use

    Returns:
    --------
        new_clusters (lst) : List of relabeled clusters.
    """
    
    clusters = list(clusters)
    counts = [clusters.count(x) for x in clusters]
    clus_to_occurances = dict(zip(clusters, counts))
    clus_occ_keys = clus_to_occurances.keys()
    #Sometimes, nothing is assigned to a given cluster.
    for i in range(n_clusters):
        if i not in clus_occ_keys:
            clus_to_occurances[i] = 0
    counts_list = sorted(list(set([count for clus, count in clus_to_occurances.items()])))
    counts_clusts = sorted([(count, clus) for clus, count in clus_to_occurances.items()])
    for count in counts_list:
        clusts_w_count = [cc[1] for cc in counts_clusts if cc[0] == count]
        if count == 0:
            for i in range(len(clusts_w_count)):
                clus = clusts_w_count[i]
                og_count = clus_to_occurances[clus]
                clus_to_occurances[clus] = og_count + .1*i
        else:
            initial_indices = sorted([(clusters.index(tied_label), tied_label) \
                            for tied_label in clusts_w_count])
            for i in range(len(initial_indices)):
                init_index, clus = initial_indices[i]
                og_count = clus_to_occurances[clus]
                clus_to_occurances[clus] = og_count + .1*i
    data = [(count, clus) for clus, count in clus_to_occurances.items()]
    sorted_data = sorted(data)
    clusts_mapping = dict([(t[1], i) for i, t in enumerate(sorted_data)])
    new_clusters = [clusts_mapping[clus] for clus in clusters]
    return new_clusters


#@lru_cache # Possibly incompatible with jax???
def calc_clusters_lambdas_cached(clusters, images_tup, simple_avg=False, print_it=False):
    """
    Performs the convex combo minimization to find the convex combo params 
    and total metric value for a given clustering.
    Does so in a memoized/cached manner to save on affinity paramaterization tuning
    time. 

    Inputs:
    -------
        clusters (tuple) : tuple where img at index i has value cluster (int)
            eg [1, 1, 0, 2] is items 1 & 2 in clus 1, 3 in clus 0 and 4 in clus 2. 
        images_tup (tuple) : tuples (all the way down) of data,  

    Returns:
    --------
        total_metric_value (jnp array) :  the sum of my metric value across each cluster
        lamdbas_dict (dict) : dict of cluster_label: zip[(lambda_i, i of corresponding img), (), ...]
            Where img_i is in the given cluster
    """
    if print_it:
        print("Simple Avg: ", simple_avg)
    images = jnp.array(images_tup)
    lambdas_dict = dict()
    total_metric_value = jnp.array([0])
    all_cluster_labels = set(clusters)
    clusters = jnp.array(clusters)
    # Maybe vmap-able? lambdas_dict may be annoying to construct
    for cluster_label in all_cluster_labels:
        relevant_image_indices = jnp.array([i for i, x in enumerate(clusters) \
                                            if x == cluster_label])
        if simple_avg:
            s = time.time()
            total_metric_value, lambdas_and_indices = min_simp_avg_metric_and_lambdas( \
                                    total_metric_value, relevant_image_indices, images)
            e = time.time()
            if print_it:
                print("Num pts in clus: ", len(relevant_image_indices))
                print("Clus Done in: ", e - s)
                print()
            
        else:
            s = time.time()
            total_metric_value, lambdas_and_indices = minimization_metric_and_lambdas( \
                                    total_metric_value, cluster_label, \
                                    relevant_image_indices, images)
            e = time.time()
            if print_it:
                print("Num pts in clus: ", len(relevant_image_indices))
                print("Clus Done in: ", e - s)
                print()
            
        lambdas_dict[cluster_label] = lambdas_and_indices
    return total_metric_value, lambdas_dict


def simple_avg_metric(relevant_images, metric=DEFAULT_METRIC):
    """
    Calculates my metric and lambdas if I'm only simple averaging my images. Eg
    lambdas are uniformly 1/len(relevant imgs)

    Inputs:
    --------
        relevant_images (jnp array) : Array of images within a cluster 
                                      which we're simple averaging.
        metric (func) : Used in eval_criterion to calculate a score for 
                        the convex combination created in eval_criterion.

    Returns:
    --------
        mini_d (dict) : dict of lambda's that are optimized and my objective
                        function value
    """
    mini_d = dict()
    # Set Lambdas as uniform.
    lambdas = jnp.full( len(relevant_images), 1/len(relevant_images) )
    mini_d['x'] = lambdas
    combo = jnp.array([lambdas[i]*relevant_images[i] for i in range(len(lambdas))])
    comboImg = jnp.sum(combo, axis=0) 
    metric_value = metric(comboImg)
    mini_d['fun'] = metric_value
    return mini_d
    

@jax.jit
def min_simp_avg_metric_and_lambdas(total_metric_value, relevant_image_indices, images):
    # Same as minimization_metric_and_lambdas but prevents if in jit'd func
    """
    Calculates my metric value accross the entire clustering 
    (for the given clustering / affinity alpha value)

    Inputs:
    --------
        total_metric_value (float) : metric value evaluated on the previous cluster(s)
        cluster_label_to_eval (int) : the label of the cluster to be evaluated
        relevant_image_indices (jax arr) : array of the indices of images in the given clus
        images (jnp arr) : Data, array of arrays where an img is a n x n array

    Returns:
    --------
        total_metric_value (float) : the input total metric value with the 
            current cluster's metric value added on w. the relevant weighting frac 
            of the total number of pts
        lambdas_and_indices (zip) : list of (item lambda, item index)
        temp_lambdas_dict (dict) : dict of cluster_label: [(lambda_i, img_i), (), ...]
    """
    temp_lambdas_dict = dict()
    relevant_images = jnp.array(images)[jnp.array(relevant_image_indices)]
    simpleCombo = simple_avg_metric(relevant_images)
    min_metric_value = simpleCombo['fun']
    clus_lambdas = simpleCombo['x']
    frac = len(relevant_images) / len(images)
    total_metric_value += min_metric_value*frac
    lambdas_and_indices = list(zip(clus_lambdas, relevant_image_indices))
    return total_metric_value, lambdas_and_indices


@partial(jax.jit, static_argnames=['eval_criterion', 'solver'])
def minimization_metric_and_lambdas(total_metric_value, cluster_label_to_eval, \
                                    relevant_image_indices, images, \
                                    eval_criterion = convComb1Clust, \
                                    solver='BFGS'):
    """
    Calculates my metric value accross the entire clustering 
    (for the given clustering / affinity alpha value)

    Inputs:
    --------
        total_metric_value (float) : metric value evaluated on the previous cluster(s)
        cluster_label_to_eval (int) : the label of the cluster to be evaluated
        relevant_image_indices (jax arr) : array of the indices of images in the given clus
        images (jnp arr) : Data, array of arrays where an img is a n x n array
        eval_criterion (func) : func which given your imgs & lambdas returns the metric val 
        solver (str) : The type of solver to use w. optimize. 
                        Only BFGS currently supported in jax

    Returns:
    --------
        total_metric_value (float) : the input total metric value with the 
            current cluster's metric value added on w. the relevant weighting frac 
            of the total number of pts
        lambdas_and_indices (list) : List of (item lambda, item index)
        temp_lambdas_dict (dict) : dict of cluster_label: [(lambda_i, img_i), (), ...]
    """
    temp_lambdas_dict = dict()
    relevant_images = jnp.array(images)[jnp.array(relevant_image_indices)]
    convexCombo = convexMinimization(relevant_images, eval_criterion, solver)
    min_metric_value = convexCombo['fun']
    clus_lambdas = convexCombo['x']
    frac = len(relevant_images) / len(images)
    total_metric_value += min_metric_value*frac
    lambdas_and_indices = list(zip(clus_lambdas, relevant_image_indices))
    return total_metric_value, lambdas_and_indices


@partial(jax.jit, static_argnames=['eval_criterion', 'solver', 'metric'])
def convexMinimization(params, eval_criterion=convComb1Clust, 
                        solver='SLSQP', metric=DEFAULT_METRIC):
    """
    clusterImages is the usual parameters we're using to form a ~convex combination,
    we're then evaluating the convex combo of the images via the eval_criterion,
        which must be better when its smaller.
    The eval_criterion is only fully defined when it has the params. 
    
    In this situation, instead of doing an explicit constrained convex minimization,
    we're doing an unconstrained minimization by using softmax to get rid 
    of our bounds and constraints (softmax will keep them each between 0-1 and ensure
    they sum to 1)
    
    Inputs:
    --------
        params (jnp array) : Array of images within a cluster which we're making 
                             A convex combination of.
        eval_critetion (func) : Function that I want to minimize 
        solver (str) : Solver supported by jax.scipy.minimize to use 
        metric (func) : Used in eval_criterion to calculate a score for 
                        the convex combination created in eval_criterion.

    Returns:
    --------
        mini_d (dict) : dict of lambda's that are optimized and my objective
                        function value
    """
    mini_d = dict()
    # Prior where everything's equally weighted
    initialLambdaGuesses = np.ones(len(params)) / len(params)
    arguments = [params, metric]
    arguments = tuple(arguments)

    finalLambdas = minimize(eval_criterion, initialLambdaGuesses,
                            method=solver, args=arguments)
    #My returned ~lambdas are the unconstrained items - but convex combo is not that
    constrained_convex_combo_params = softmax(finalLambdas.x)
    mini_d['x'] = constrained_convex_combo_params
    mini_d['fun'] = finalLambdas.fun
    return mini_d


###############
# Helper Func #
###############

def usage():
    process = psutil.Process(os.getpid())
    return process.memory_info()[0] / float(2 ** 20)


if __name__ == "__main__":
    # gamma_tuning_ex(use_new_data=True, n_cs=5, arraysPerCluster=3, my_method="BFGS")
    basic_ex(my_gamma=jnp.array([0.1]), use_new_data=True, n_cs=5, arraysPerCluster=3, solver="BFGS")
    #save_affinity_matrix()