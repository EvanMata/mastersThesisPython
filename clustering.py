import jax
import time
import pickle

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from sklearn.cluster import SpectralClustering

#from scipy.special import softmax
from jax.nn import softmax
from functools import partial
from scipy.optimize import minimize
from jax.scipy.optimize import minimize as min2


import openRawData as opn
import genSyntheticData as dg
import pathlib_variable_names as var_names


def distance_from_pure_color(img, pure_colors):
    """
    Calculates the distance the img is from pure colors, eg for all pixels
    find the neared pure color, square that distance, sum over all pixels,
    norm by the number of pixels and the maximum possible distance and sqrt.
    Dist should be [0, 1]

    Args:
        img (array of floats): Image being analyzed, assumed to be n x n, 
            if there are multiple different color axes, analyze seperately 
            then combine.
        pure_colors (list of floats): the numbers which represent pure colors
            eg what the ideal pixel should be eg
            1=black/magnetic, 0=white/non-magnetic so [0, 1]

    Returns:
        normed_dist_pc (float): distance, in [0,1]
        closest_colors (array of floats): closest pure colors for all pixels
        largest_possi_dist (float): Normalization factor, 
    """
    img = np.array(img)
    pure_colors = sorted(pure_colors)
    color_midpoints = [(pure_colors[i+1] + pure_colors[i]) / 2 
                       for i in range(len(pure_colors) - 1)]
    previously_updated = np.zeros(img.shape)
    closest_colors = np.zeros(img.shape)
    for i in range(len(color_midpoints)):
        color_midpoint = color_midpoints[i]
        currently_updated = np.zeros(img.shape)
        currently_updated[img < color_midpoint] = 1
        new_pts = currently_updated - previously_updated
        closest_colors[new_pts > 0] = pure_colors[i]
        previously_updated = currently_updated
    closest_colors[previously_updated == 0] = pure_colors[i + 1]
    
    all_large_dists_1 = (color_midpoints - np.array(pure_colors[:-1]))**2
    all_large_dists_2 = (np.array(pure_colors[1:]) - color_midpoints)**2
    largest_possi_dist = max([max(all_large_dists_1), max(all_large_dists_2)])
    
    dist_pc = (closest_colors - img)**2
    dist_pc = np.sum(dist_pc) / (img.size * largest_possi_dist)
    normed_dist_pc = dist_pc 
    
    return normed_dist_pc, closest_colors, largest_possi_dist


def norm_01(img):
    return (img - np.min(img))/np.ptp(img)


def total_variation_norm(img):
    img = norm_01(img)
    x_diffs = np.abs(img[:,1:] - img[:,:-1])
    y_diffs = np.abs(img[1:,:] - img[:-1,:])
    print(x_diffs)
    print(y_diffs)
    tot = np.sum(x_diffs) + np.sum(y_diffs)
    #NORM FACTOR NOT CORRECT
    norm_factor = (img.shape[0] - 1)*(img.shape[1] - 1)
    tv_dist = tot / norm_factor
    return tv_dist


def affinityMatrix(images, gamma):
    """
    Generate the affinity matrix for the given images and
    the set value of gamma parameterizing the affinity function
    """
    affinities = np.zeros((len(images), len(images)))
    for i in range(len(images)):
        for j in range(i+1, len(images)):
            pairAffinity = calcPairAffinity(images[i], images[j], gamma)
            affinities[i, j] = pairAffinity
            affinities[j, i] = pairAffinity

        affinities[i, i] = 1
    return affinities


def calcPairAffinity(image1, image2, gamma):
    diff = np.abs(np.sum(image1 - image2))  # L1 Norm, no further normalization
    return np.exp(-gamma*diff)


def performClustering(affinities, n_clusters):
    """
    Spectral clustering with pre-computed affinity matrix.
    """
    clustering = SpectralClustering(n_clusters=n_clusters,
                                    affinity="precomputed",
                                    random_state=0).fit(affinities)
    clusters = clustering.labels_
    return clusters


def customMetric(image, lambdaVal=0.5):  # rename closest_sq_dist_tv_mix
    """
    A convex combination of total variation and square distance from the closest domain/pureColor
    sTV >> dPC most of the time in the trivial example tested
    """
    scaledImg = (image - jnp.min(image)) / (jnp.max(image) - jnp.min(image))
    sTV = scaledTotalVariation(scaledImg)
    dPC = distFromPureColor(scaledImg, pureColors=[0, 1])
    return lambdaVal*sTV + (1 - lambdaVal)*dPC


def convexComboMetricEval1Cluster(convexComboParams, realSpaceImgs, metric=customMetric):
    # Should use some args/kwargs stuff here.
    """
    Given a list of images and their corresponding lambdas representing some
    convex combination of them, evaluate the provided metric on them
    """
    convArrays = [convexComboParams[i]*realSpaceImgs[i] for i in range(len(realSpaceImgs))]
    imageToEval = sum(convArrays)
    clusterMetric = metric(imageToEval)
    return clusterMetric


def convComboMetricEval1ClusterUnconstrained(convexComboParams, realSpaceImgs, metric=customMetric):
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


def scaledTotalVariation(scaledImg): #May adjust to punish steps?
    # Calculate the scaled total variation ( scaled to 0-1 )
    diffX = jnp.abs(jnp.diff(scaledImg, axis=1))
    diffY = jnp.abs(jnp.diff(scaledImg, axis=0))
    scaledTV = (jnp.sum(diffX) / diffX.size) + (jnp.sum(diffY) / diffY.size)
    return scaledTV


def rounder(values):
    def f(x):
        idx = jnp.argmin(jnp.abs(values - x))
        return values[idx]
    return jnp.frompyfunc(f, 1, 1)


def distFromPureColor(image, pureColors=[0, 1], printIt=False):
    # There's got to be a more efficient way of doing this.
    """
    For each pixel, find its closest class (ie pureColor, ie magnetic=1, nonMagentic=0)
    then take the distance from that pure color squared. Finally, normalize by
    the worst possible value to be [0,1] range
    """
    closestColors = jnp.zeros(image.shape)
    pureColors = sorted(pureColors)
    colorMidpoints = [jnp.mean(jnp.array([pureColors[i], pureColors[i+1]])) for i in
                      range(len(pureColors) - 1)]
    previouslyUnseen = jnp.ones(image.shape)
    justSeen = jnp.ones(image.shape)

    if printIt:
        print("Color Midpoints: ", colorMidpoints)
        print("Initial Closest Colors: \n", closestColors)

    for i in range(len(colorMidpoints)):
        colorMidpoint = colorMidpoints[i]
        
        #previouslyUnseen[image < colorMidpoint] = 0
        #previouslyUnseen = previouslyUnseen.at[image < colorMidpoint].set(0)
        previouslyUnseen = jnp.where(image < colorMidpoint, 0, 1)
        
        #previouslyUnseen[image < colorMidpoint] = 0
        # closestColors[image < colorMidpoint] = pureColors[i]
        
        # Could try to get set of indices and subtract it from set of indices in justSeen
        # But just a where clause to get indices breaks too.
        
        newInfo = justSeen - previouslyUnseen #Just seen is 1's
        currentColorArr = jnp.ones(image.shape) * pureColors[i]
        closestColors = jnp.where(newInfo >= 1, currentColorArr, closestColors)

        if printIt:
            print("Current Midpoint: ", colorMidpoint)
            print("Previously Unseen: \n", previouslyUnseen)
            print("New Points: \n", newInfo)
            print("Updated Closest Colors: \n", closestColors)

        justSeen = jnp.copy(previouslyUnseen)
        if i == len(colorMidpoints) - 1:
            #closestColors = closestColors.at[image >= colorMidpoint].set(pureColors[i+1])
            closestColors = jnp.where(image >= colorMidpoint, pureColors[i+1], closestColors)
            #closestColors[image >= colorMidpoint] = pureColors[i+1]
            #previouslyUnseen[image >= colorMidpoint] = 0
            #previouslyUnseen = previouslyUnseen.at[image >= colorMidpoint].set(0)
            previouslyUnseen = jnp.where(image < colorMidpoint, 0, 1)
            # ^Not necessary, aside from sanity check

            if printIt:
                print("Final Updated Closest Colors: \n", closestColors)
                print("Final Unseen: \n", previouslyUnseen)
                print("Final New Points: \n", justSeen - previouslyUnseen)

    distFromClosests = jnp.abs(image - closestColors)**2
    distOverallReduced = jnp.sum(distFromClosests) / image.size

    biggestPossibleGapInfo = [pureColors[0]] + colorMidpoints + [pureColors[-1]]
    allGaps = jnp.array([biggestPossibleGapInfo[i+1] - biggestPossibleGapInfo[i] for i in
                      range(len(biggestPossibleGapInfo) - 1)])
    largestPossiGap = jnp.max(allGaps)
    normFactor = largestPossiGap**2
    distOverall = distOverallReduced / normFactor

    return distOverall

# JIT THIS - issue will be functional inputs.
# @jax.jit
@partial(jax.jit, static_argnames=['eval_criterion', 'solver'])
def convexMinimization(params, eval_criterion=convexComboMetricEval1Cluster, solver='SLSQP'):
    """
    clusterImages is the usual parameters we're using to form a convex combination,
    we're then evaluating the convex combo of the images via the eval_criterion,
        which must be better when its smaller.
    The eval_criterion is only fully defined when it has the params
    """
    # Prior where everything's equally weighted
    initialLambdaGuesses = np.ones(len(params)) / len(params)
    cons = {'type': 'eq', 'fun': con}  # Enforce sum of lambdas = 1
    bnds = [(0, 1)]*len(params)  # Each lambda is between 0 and 1
    arguments = params

    finalLambdas = minimize(eval_criterion, initialLambdaGuesses,
                            method='SLSQP', args=arguments,
                            bounds=bnds, constraints=cons)

    return finalLambdas

# JIT THIS - issue will be functional inputs.
@jax.jit
def convexMinimization2(params, eval_criterion=convComboMetricEval1ClusterUnconstrained, 
                        solver='SLSQP', metric=customMetric):
    """
    clusterImages is the usual parameters we're using to form a ~convex combination,
    we're then evaluating the convex combo of the images via the eval_criterion,
        which must be better when its smaller.
    The eval_criterion is only fully defined when it has the params. 
    
    In this situation, instead of doing an explicit constrained convex minimization,
    we're doing an unconstrained minimization by using softmax to get rid 
    of our bounds and constraints (softmax will keep them each between 0-1 and ensure
    they sum to 1)
    
    """
    mini_d = dict()
    # Prior where everything's equally weighted
    initialLambdaGuesses = np.ones(len(params)) / len(params)
    arguments = [params, metric]
    arguments = tuple(arguments)

    finalLambdas = min2(eval_criterion, initialLambdaGuesses,
                            method=solver, args=arguments)
    # #My returned ~lambdas are the unconstrained items - but convex combo is not that
    constrained_convex_combo_params = softmax(finalLambdas.x)
    mini_d['x'] = constrained_convex_combo_params
    mini_d['fun'] = finalLambdas.fun
    return mini_d

# Jit this?
def convexMinimization3(params, eval_criterion=convComboMetricEval1ClusterUnconstrained, 
                        solver='SLSQP'):
    """
    clusterImages is the usual parameters we're using to form a ~convex combination,
    we're then evaluating the convex combo of the images via the metric,
        which must be better when its smaller.
    The metric is only fully defined when it has the params. 
    
    In this situation, instead of doing a convex combination over all points, we 
    split our points into a spanning disjoint union of sets, convex combo each set of points
    then convex combo the ~centroids of each set
    
    """
    sub_centroids_list = []
    subsets_info = dict()
    num_subsets = int(np.sqrt(len(params)))
    partitioned_sets = partition_sets(params, num_subsets)
    for i in range(len(partitioned_sets)): #Vectirize this?
        partial_set = partitioned_sets[i]
        setInfo = convexMinimization2(partial_set, eval_criterion, solver)
        setLambdas = setInfo['x']
        subset_centroid = sum(np.array([lmd*partial_set[j] for j, lmd in enumerate(setLambdas)]))
        subsets_info[i] = (subset_centroid, setLambdas)
        sub_centroids_list.append(subset_centroid)
        
    # Prior where everything's equally weighted
    initialLambdaGuesses = np.ones(len(partitioned_sets)) / len(partitioned_sets)
    arguments = sub_centroids_list

    subcentroid_Lambdas = min2(eval_criterion, initialLambdaGuesses,
                            method=solver, args=arguments)
    subcentroid_Lambdas_d = dict()
    subcentroid_Lambdas_d['x'] = softmax(subcentroid_Lambdas.x)
    true_lambdas = []
    #Multiply out the subset centroid lambda by all the lambda's used to make up the subset centroid
    for i in range(len(subcentroid_Lambdas['x'])):
        centroid_lmd = subcentroid_Lambdas['x'][i]
        centroid_pieces_lmds = subsets_info[i][1]
        individual_lmds = [centroid_lmd*individual_lmd for individual_lmd in centroid_pieces_lmds]
        true_lambdas += individual_lmds
        
    true_lambdas = np.array(true_lambdas)
    true_metric_val = eval_criterion(true_lambdas, params) 
    #eval_criterion, ie default 'fun' IS NOT CORRECT
    #Its the metric value for the clustering of subclusters, not all the individual pieces.
    #So Eval the correct one and plug it in as true_metric_val
    all_info = {'x': true_lambdas, 'fun': true_metric_val} 
    return all_info


def partition_sets(a, n):
    # Split list a into n non-overlapping parts
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]


def con(lambdas): # Trivial convex combination constraint
    return np.sum(lambdas) - 1


def clustering_objective(gamma, images, n_clusters, img_names, \
        print_it=False, clustering_case='unconstrained', solver="SLSQP"):
    """
    Calculates the total metric value after convex minimization
    for a fixed gamma and number of clusters, and the convex combination
    params in terms of the items in the cluster
    """
    t_s = time.time() #total
    a_s = time.time() #affininty
    affinities = affinityMatrix(images, gamma)
    a_e = time.time()
    c_s = time.time() #clustering
    clusters = performClustering(affinities, n_clusters)
    c_e = time.time()
    if print_it:
        print(clusters)
    all_cluster_labels = set(clusters)
    total_metric_value = 0
    lambdas_dict = dict()
    o_s = time.time() #optimization
    for cluster_label in all_cluster_labels:
        relevant_image_indices = [i for i, x in enumerate(clusters) if x == cluster_label]
        relevant_images = np.array(images)[relevant_image_indices]
        relevant_clus_names = np.array(img_names)[relevant_image_indices]
        if clustering_case == 'constrained':
            eval_criterion = convexComboMetricEval1Cluster
            convexCombo = convexMinimization(relevant_images, eval_criterion, solver)
        elif clustering_case == 'unconstrained':
            eval_criterion = convComboMetricEval1ClusterUnconstrained
            convexCombo = convexMinimization2(relevant_images, eval_criterion, solver)
        elif clustering_case == 'unconstrained_stack':
            eval_criterion = convComboMetricEval1ClusterUnconstrained
            convexCombo = convexMinimization3(relevant_images, eval_criterion, solver)
        else:
            eval_criterion = convComboMetricEval1ClusterUnconstrained
            convexCombo = convexMinimization3(relevant_images, eval_criterion, solver)
        min_metric_value = convexCombo['fun']
        clus_lambdas = convexCombo['x']
        frac = len(relevant_images) / len(images)
        total_metric_value += min_metric_value*frac
        lambdas_and_indices = list(zip(clus_lambdas, relevant_image_indices))
        lambdas_dict[cluster_label] = lambdas_and_indices
    o_e = time.time()
    t_e = time.time()
    print("Total Time: %f, Affininty Time: %f, Clustering Time: %f, Optimization_time: %f"%
          (t_e - t_s, a_e - a_s, c_e - c_s, o_e - o_s))
    
    return total_metric_value, lambdas_dict


def basic_ex(use_new_data=False, n_cs=5, arraysPerCluster=3, save_centroids=False, 
             print_it=False, graph_it=True, display_clusts=True, 
             clustering_case='unconstrained_stack', only1=False,
             solver="SLSQP"):
    """
    Identifies and prints lambdas and cluster indices for each
    possible number of clusters.
    Creates graphs of metric over number of clusters,
    and number of approx 0 lambdas over number of clusters.

    Inputs:
        use_new_data : bool - False use pre-generated data, or
                        True generate new data
        n_cs : number of clusters to generate if use_new_data = True
        save_centroids : bool - whether to save images of each centroid. NOT IMPLEMENTED
        only1 : bool - if True doesn't test every possible num clusters, only the correct one
    """
    ys = []
    num_approx_zero_lamdbas = []
    if use_new_data:
        names_and_data = dg.genClusters(numClusters=n_cs, arraysPerCluster=arraysPerCluster, n1=100, n2=100,
                                        noiseLv=0.25, display=False, saveClusters=False)
        my_n = n_cs
    else:
        names_and_data = opn.openSyntheticData()
        my_n = 3
    data = [i[1] for i in names_and_data]
    names = [i[0] for i in names_and_data]
    if only1:
        metric_val, clus_info = clustering_objective(
            gamma=0.01, images=data, n_clusters=n_cs, solver=solver,
            img_names=names, clustering_case=clustering_case)
        
        num_approx_zero_lambdas_current_clus = extract_num_non_zero_lambdas(clus_info)
        ys.append(metric_val)
        xs = np.array(range(len(data))) + 1
    else:
        for num_clusters in range(len(data)):
            metric_val, clus_info = clustering_objective(
                gamma=0.01, images=data, n_clusters=num_clusters + 1, 
                solver=solver, img_names=names, clustering_case=clustering_case)
            num_approx_zero_lambdas_current_clus = extract_num_non_zero_lambdas(clus_info)
            centroids = create_centroids(clus_info, data)
            if display_clusts:
                if num_clusters + 1 == my_n:
                    for centroid_num, centroid in centroids.items():
                        opn.heatMapImg(centroid)

            ys.append(metric_val)
            num_approx_zero_lamdbas.append(num_approx_zero_lambdas_current_clus)
            if print_it:
                print("For %d clusters: "%(num_clusters+1))
                print(clus_info)

        if print_it:
            print("Metric Values: ")
            print(ys)

        xs = np.array(range(len(data))) + 1
        if graph_it:
            fig, axs = plt.subplots(2, sharex=True)

            axs[0].plot(xs, ys)
            axs[0].set_title("Metric Over Number of Clusters, n_cs=%d"%my_n)
            axs[0].set_ylabel("Metric (TV + PureColorDist), smaller=better")

            axs[1].plot(xs, num_approx_zero_lamdbas)
            axs[1].set_title("Number of approx 0 Components across all Clusters, n_cs=%d" % my_n)
            axs[1].set_xlabel("Number of Clusters")
            axs[1].set_ylabel("Number of approx. 0 components")

            plt.show()

    return xs, ys, num_approx_zero_lamdbas


def extract_num_non_zero_lambdas(clustering_info_dict):
    """
    Given a dictionary of clustering info, extract the number of approximately 0
    lambda values.
    """
    num_approx_zero = 0
    for clus, lambdas_and_item_numbers in clustering_info_dict.items():
        threshold = len(lambdas_and_item_numbers)
        threshold = 1 / (threshold*100)
        pts_below_thresh = [i[0] for i in lambdas_and_item_numbers if i[0] < threshold]
        num_approx_zero += len(pts_below_thresh)

    return num_approx_zero


def create_centroids(clustering_info_dict, data_list):
    """
    Creates the centroids of a given clustering
    MAKE SURE THIS WORKS FOR NON-SYNTHETIC DATA, SPECIFICALLY i[1] STILL INDEX.
    """
    centroids_dict = dict()
    for clus, lambdas_and_item_numbers in clustering_info_dict.items():
        #i[0] is the lambda value, i[i] is the index of the cluster
        data_items = [np.array(data_list[i[1]])*i[0] for i in lambdas_and_item_numbers]
        centroid = sum(data_items)
        centroids_dict[clus] = centroid
    return centroids_dict


def how_often_correct(samples=100):
    """
    Generates random data with a random number of clusters samples times,
    then finds the min of my metric for each sample and sees if its the
    actually the same as the number of clusters originally randomly used to gen the data.
    """
    num_cor = 0
    for i in range(samples):
        n_cs = np.random.randint(2, 15)
        random_data = dg.genClusters(numClusters=n_cs, arraysPerCluster=3, n1=100, n2=100,
                noiseLv=0.1, display=False, saveClusters=False)
        data = [i[1] for i in random_data]
        names = [i[0] for i in random_data]
        ys = []
        for num_clusters in range(len(data)):
            metric_val, clus_info = clustering_objective(
                gamma=0.01, images=data, n_clusters=num_clusters + 1, img_names=names)
            ys.append(metric_val)
        pred_n_cs = np.amin(ys)
        if pred_n_cs == n_cs:
            num_cor += 1

    print( num_cor / samples )


def generate_stats(n_cs=5, arraysPerCluster=5, num_samples=100):
    """
    Generate plots of how my metric changes over a sample as the number of clusters changes 
    from 1 to the maximum possible, and how along the same scale the number of approximately 
    0 lambda values change as well.

    Args:
        n_cs (int, optional): Number of clusters. Defaults to 5.
        arraysPerCluster (int, optional): Number of pts per cluster. Defaults to 5.
        num_samples (int, optional): Number of times to redo the experiment 
                                        to generate statistics. Defaults to 100.
    """
    all_metric_vals = []
    zero_lambdas_vals = []
    for i in range(num_samples):
        xs, metric_values, num_approx_zero_lamdbas = basic_ex(
                                                        use_new_data=True, 
                                                        n_cs=n_cs, 
                                                        arraysPerCluster = arraysPerCluster,
                                                        save_centroids=False, 
                                                        print_it=False, 
                                                        graph_it=False, 
                                                        display_clusts=False)
        all_metric_vals.append(metric_values)
        zero_lambdas_vals.append(num_approx_zero_lamdbas)

    metric_info = np.percentile(all_metric_vals, [25, 50, 75], axis=0)
    lambda_info = np.percentile(zero_lambdas_vals, [25, 50, 75], axis=0)

    fig, axs = plt.subplots(2, sharex=True)

    axs[0].plot(xs, metric_info[0])
    axs[0].plot(xs, metric_info[1])
    axs[0].plot(xs, metric_info[2])
    axs[0].set_title("Metric Over Number of Clusters, n_cs=%d, n_pts/clus=%d"%(n_cs, arraysPerCluster))
    axs[0].set_ylabel("Metric (TV + PureColorDist), smaller=better")

    axs[1].plot(xs, lambda_info[0])
    axs[1].plot(xs, lambda_info[1])
    axs[1].plot(xs, lambda_info[2])
    axs[1].set_title("Number of ~0 Components over all Clusters, n_cs=%d, n_pts/clus=%d"%(n_cs, arraysPerCluster))
    axs[1].set_xlabel("Number of Clusters")
    axs[1].set_ylabel("Number of approx. 0 components")

    plt.show()

#@partial(jit, static_argnums=[1,2,3,4,5,6])
def convex_combo_lots_of_pts(data, names, num_clus=2, np_pts_per_clus=100, 
                             display_clusts=False, solver="SLSQP", clustering_case='unconstrained'): #num_clus=2, np_pts_per_clus=100
    """
    How long does it take to do larger convex combinations?
    """
    
    '''
    names_and_data = dg.gen_fake_data(num_clus=num_clus, 
                                np_pts_per_clus=np_pts_per_clus, 
                                display_imgs=False, 
                                save_files=False)
    data = [i[1] for i in names_and_data]
    names = [i[0] for i in names_and_data]
    '''
    s = time.time()
    metric_val, clus_info = clustering_objective(
            gamma=0.01, images=data, n_clusters=num_clus, 
            img_names=names, solver=solver, clustering_case=clustering_case)
    e = time.time()
    msg = """Given %d clusters and %d points per cluster, 
            time taken (s) for convex combo: """%(num_clus, np_pts_per_clus)
    time_taken = e - s
    print(msg, time_taken)
    if display_clusts:
        centroids = create_centroids(clus_info, data)
        for centroid_num, centroid in centroids.items():
            opn.heatMapImg(centroid)
    return time_taken


def convex_combo_time_scaling(num_clus, num_pts_to_test=[5,10,20,50,100,200], 
                              clustering_case='unconstrained'):
    solvers = ["Nelder-Mead", "Powell", "CG", "BFGS", \
            "L-BFGS-B", "TNC", 'COBYLA', "SLSQP", "trust-constr"] 
    solvers = ['COBYLA', "SLSQP"]
    solvers = ["BFGS"]
    #"trust-krylov", "trust-exact", "Newton-CG", "dogleg", "trust-ncg"
    # ^ All Error out: Jacobian is required for Newton-CG method error
    
    data_set = []
    name_sets = []
    for num_pts in num_pts_to_test:
        names_and_data = dg.gen_fake_data(num_clus=num_clus, 
                                np_pts_per_clus=num_pts, 
                                display_imgs=False, 
                                save_files=False)
        data = [i[1] for i in names_and_data]
        names = [i[0] for i in names_and_data]
        data_set.append(data)
        name_sets.append(names)
    
    pts_dict = {}
    for solver in solvers:
        print(solver)
        current_times = []
        for i in range(len(num_pts_to_test)):
            np_pts_per_clus = num_pts_to_test[i]
            data = data_set[i]
            names = name_sets[i]
            tot_time_taken = convex_combo_lots_of_pts(\
                data, names, num_clus=num_clus, 
                np_pts_per_clus=np_pts_per_clus,
                solver=solver, display_clusts=False, 
                clustering_case=clustering_case)
            time_taken = tot_time_taken / num_clus
            current_times.append(time_taken)
        pts_dict[solver] = current_times
        plt.plot(num_pts_to_test, current_times, label=solver)
    plt.ylabel("Time Taken (s)")
    plt.xlabel("Number of Pts per cluster")
    plt.title("Time Scaling of Convex Combination")
    plt.legend()
    plt_name = clustering_case + "_time_taken_lambdas_opti"
    
    with open(plt_name+"_data.pickle", 'wb') as handle:
        pickle.dump(pts_dict, handle)
    
    plt_name += ".png"
    plt.savefig(plt_name)


def gen_plot_from_saved(pickle_name, num_pts_to_test):
    with open(pickle_name, 'rb') as handle:
        pts_dict = pickle.load(handle)
        
    print(pts_dict)
    for solver, current_times in pts_dict.items():
        #current_times = [c if c < 10000 else 10000 for c in current_times ]
        plt.plot(num_pts_to_test, current_times, label=solver)
    plt.ylabel("Time Taken (s)")
    plt.xlabel("Number of Pts per cluster")
    plt.title("Time Scaling of Convex Combination")
    plt.legend()
    plt.show()
        

def compare_dpcs(num_clus=2, np_pts_per_clus=2):
    """Shows that its super quick to calculate the distance from a pure color for either method 
    ~order .00009 for t1 and .00014 for t2; t1 better but not my main time waster.

    Args:
        num_clus (int, optional): _description_. Defaults to 2.
        np_pts_per_clus (int, optional): _description_. Defaults to 2.
    """
    names_and_data = dg.gen_fake_data(num_clus=num_clus, 
                                np_pts_per_clus=np_pts_per_clus, 
                                display_imgs=False, 
                                save_files=False)
    data = [i[1] for i in names_and_data]
    names = [i[0] for i in names_and_data]
    s1 = time.time()
    for img in data:
        normed_dist_pc, closest_colors, largest_possi_dist = distance_from_pure_color(img, pure_colors=[0, 1])
    e1 = time.time()
    s2 = time.time()
    for img in data:
        distOverall = distFromPureColor(img, pureColors=[0, 1], printIt=False)
    e2 = time.time()
    t1 = e1 - s1
    t2 = e2 - s2
    num_pts = num_clus*np_pts_per_clus
    print("Amount of time for t1: %f and t2: %f"%(t1, t2))
    print("Avg time per pt: %f for t1 and %f for t2"%(t1/num_pts, t2/num_pts))
     
     
def compare_tv(num_clus=2, np_pts_per_clus=2):
    """Shows that its quick to calculate the total variation for either method
    method 1 is significantly better (multiple orders of magnitude) than method 2
    but both are still quite quick

    Args:
        num_clus (int, optional): _description_. Defaults to 2.
        np_pts_per_clus (int, optional): _description_. Defaults to 2.
    """
    names_and_data = dg.gen_fake_data(num_clus=num_clus, 
                                np_pts_per_clus=np_pts_per_clus, 
                                display_imgs=False, 
                                save_files=False)
    data = [i[1] for i in names_and_data]
    names = [i[0] for i in names_and_data]
    s1 = time.time()
    for img in data:
        scaledTV = scaledTotalVariation(img)
    e1 = time.time()
    s2 = time.time()
    for img in data:
        tv_dist = total_variation_norm(img)
    e2 = time.time()
    t1 = e1 - s1
    t2 = e2 - s2
    num_pts = num_clus*np_pts_per_clus
    print("Amount of time for t1: %f and t2: %f"%(t1, t2))
    print("Avg time per pt: %f for t1 and %f for t2"%(t1/num_pts, t2/num_pts))   

        
if __name__ == "__main__":
    # affinityMatrix(images, gamma=0.1)
    #basic_ex(use_new_data=True, n_cs=2, arraysPerCluster=16, graph_it=False, display_clusts=False,
    #         clustering_case='unconstrained', only1=True, solver='BFGS')
    #basic_ex(use_new_data=True, n_cs=5)
    # how_often_correct(samples=15)
    # generate_stats(n_cs=2, arraysPerCluster=5, num_samples=40)
    #convex_combo_time_scaling(num_clus=2, num_pts_to_test=[750,1000,1500],
    #                          clustering_case='unconstrained_stack')
    convex_combo_time_scaling(num_clus=2, num_pts_to_test=[50,60,70,80,90,100],
                              clustering_case='constrained')
    #convex_combo_time_scaling(num_clus=2, num_pts_to_test=[50,60,70,80,90,100],
    #                          clustering_case='constrained')
    #pickle_name = 'constrained'+"_time_taken_lambdas_opti"+"_data.pickle"
    #gen_plot_from_saved(pickle_name, num_pts_to_test=[50,60,70,80,90,100])
    #convex_combo_time_scaling(num_clus=2, num_pts_to_test=[400,500,600,700,800,900,1000])
    #compare_tv(num_clus=2, np_pts_per_clus=1000)

