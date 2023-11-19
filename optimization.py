import jax
import time
import pickle

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
             provided_data=None):
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
                                                        n_clusters=n_cs)
    
    print("Gamma: ", my_gamma)
    print("Metric Value: ", metric_val)
    print("Lambdas: ")
    print(lambdas_dict)
    print()
    
    return metric_val


def get_info(my_gamma, provided_data, n_cs):
    """
    Same as basic_clustering, but returns my lambdas dict
    
    Inputs:
    --------
        my_gamma (float) : Value of gamma to use for the tunning.
        provided_data (lst) :    A list of tuples of (cluster, data point)
        n_cs (int) :             Number of clusters to fit
    Returns: 
    --------
        metric_val (float) : Value of my metric for the given clustering 
        lamdbas_dict (dict) : dict of cluster num: [(img lambda, img num), (), ...]
    """
    
    data = [i[1] for i in provided_data]

    data = jnp.array(data)
    images_tup = totuple(data)
    metric_val, lambdas_dict = const_gamma_clustering(gamma=my_gamma, images_tup=images_tup, 
                                                        n_clusters=n_cs)
    
    return metric_val, lambdas_dict


def gamma_tuning_ex(use_new_data=True, n_cs=5, arraysPerCluster=3, my_method="BFGS"):
    """
    Tunes the gamma used in my clustering.
    """
    fun = basic_ex
    init_guess = jnp.array([1])
    arguements = (use_new_data, n_cs, arraysPerCluster, my_method)
    s = time.time()
    minResults = min2(fun=fun, x0 = init_guess,
                            method=my_method, args=arguements)
    e = time.time()
    min_metric_value = minResults.fun
    gamma_value = minResults.x
    print("Optimal Alpha Value was: ", gamma_value)
    total_time = e - s
    print("Total Time: ", total_time)
    """
    EDIT TO MAKE SURE USES ALPHA TO GET BEST.
    """


def const_gamma_clustering(gamma, images_tup, n_clusters, print_timing=False):
    """
    Clustering given my gamma and params. This is what I want to minimize for 
    a given gamma.

    Inputs:
    --------
        gamma (float) : Gaussian Affinity Kernel param to test
        images_tup (tuple of tuples of tuples) : Set of images being clustered
        n_clusters (int) : number of clusters to test 
    Returns:
    --------
        total_metric_value (float) :  The sum of my metric value across each cluster
                                      for the given gamma value
        lamdbas_dict (dict) : dict of cluster_label: [(lambda_i, img_i), (), ...]
                                Where img_i is in the given cluster
    """
    images = jnp.array(images_tup)
    gamma = jnp.array([gamma])
    s1 = time.time()
    affinities = affinity_matrix(images, gamma)
    e1 = time.time()
    if print_timing:
        print("Affinity Time: ", e1 - s1)
    clusters = performClustering(affinities, n_clusters)
    e2 = time.time()
    if print_timing:
        print("Clustering Time: ", e2 - e1)
    clusters = clustering_to_cachable_labels(clusters, n_clusters)
    clusters = tuple(clusters)
    e3 = time.time()
    if print_timing:
        print("Making Labels cachable: ", e3 - e2)
    e4 = time.time()
    total_metric_value, lambdas_dict = calc_clusters_lambdas_cached(clusters, images_tup)
    e5 = time.time()
    if print_timing:
        print("Performing Convex Minimization: ", e5 - e4)
    #return total_metric_value, lambdas_dict #Can't return more than a float for minimzation.
    return total_metric_value, lambdas_dict


def totuple(m):
    #makes m into a tuple of tuples of tuples all the way down
    if m.shape == ():
        return m.item()
    else:
        return tuple(map(totuple, m))


def calcPairAffinity(image1, image2, gamma):
    #Returns a jnp array of 1 float, jnp.sum adds all elements together
    diff = jnp.sum(jnp.abs(image1 - image2))  
    normed_diff = diff / image1.size
    val = jnp.exp(-gamma*normed_diff)
    val = jnp.array(val, dtype=jnp.float16)
    return 


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
    v_cPA = jax.vmap(pair_affinity_func, pair_affinity_parallel_axes, 0)
    imgs_1, imgs_2 = zip(*list(combinations(arr_of_imgs,2)))
    arr = jnp.zeros((n_imgs, n_imgs))
    affinities = v_cPA(jnp.array(imgs_1), jnp.array(imgs_2), gamma)
    affinities = affinities.reshape(-1)
    arr = arr.at[jnp.triu_indices(arr.shape[0], k=1)].set(affinities)
    arr = arr + arr.T
    arr = arr + jnp.identity(n_imgs)
    return arr


def performClustering(affinities, n_clusters):
    """
    Spectral clustering with pre-computed affinity matrix.
    """
    print()
    print("Affinities: ")
    print(affinities)
    print()
    clustering = SpectralClustering(n_clusters=n_clusters,
                                    affinity="precomputed",
                                    random_state=0).fit(affinities)
    clusters = clustering.labels_
    return clusters


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


@lru_cache
def calc_clusters_lambdas_cached(clusters, images_tup):
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
        lamdbas_dict (dict) : dict of cluster_label: [(lambda_i, img_i), (), ...]
            Where img_i is in the given cluster
    """
    images = jnp.array(images_tup)
    lambdas_dict = dict()
    total_metric_value = jnp.array([0])
    all_cluster_labels = set(clusters)
    clusters = jnp.array(clusters)
    # Maybe vmap-able? lambdas_dict may be annoying to construct
    for cluster_label in all_cluster_labels:
        relevant_image_indices = jnp.array([i for i, x in enumerate(clusters) \
                                            if x == cluster_label])
        total_metric_value, temp_lambdas_dict = minimization_metric_and_lambdas( \
                                    total_metric_value, cluster_label, \
                                    relevant_image_indices, images)
        lambdas_dict = {**lambdas_dict, **temp_lambdas_dict}
    return total_metric_value, lambdas_dict


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
    temp_lambdas_dict[cluster_label_to_eval] = lambdas_and_indices
    return total_metric_value, temp_lambdas_dict


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




if __name__ == "__main__":
    # gamma_tuning_ex(use_new_data=True, n_cs=5, arraysPerCluster=3, my_method="BFGS")
    basic_ex(my_gamma=0.1, use_new_data=True, n_cs=5, arraysPerCluster=3, solver="BFGS")
     