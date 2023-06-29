#  import jax.numpy as jnp

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from sklearn.cluster import SpectralClustering

import openRawData as opn
import dataGeneration as dg

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
    scaledImg = (image - np.min(image)) / (np.max(image) - np.min(image))
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


def scaledTotalVariation(image): #May adjust to punish steps?
    # Calculate the scaled total variation ( scaled to 0-1 )
    scaledImg = (image - np.min(image)) / (np.max(image) - np.min(image))
    diffX = np.abs(np.diff(scaledImg, axis=1))
    diffY = np.abs(np.diff(scaledImg, axis=0))
    scaledTV = (np.sum(diffX) / diffX.size) + (np.sum(diffY) / diffY.size)
    return scaledTV


def distFromPureColor(image, pureColors=[0, 1], printIt=False):
    # There's got to be a more efficient way of doing this.
    """
    For each pixel, find its closest class (ie pureColor, ie magnetic=1, nonMagentic=0)
    then take the distance from that pure color squared. Finally, normalize by
    the worst possible value to be [0,1] range
    """
    closestColors = np.zeros(image.shape)
    pureColors = sorted(pureColors)
    colorMidpoints = [np.mean([pureColors[i], pureColors[i+1]]) for i in
                      range(len(pureColors) - 1)]
    previouslyUnseen = np.ones(image.shape)
    justSeen = np.ones(image.shape)

    if printIt:
        print("Color Midpoints: ", colorMidpoints)
        print("Initial Closest Colors: \n", closestColors)

    for i in range(len(colorMidpoints)):
        colorMidpoint = colorMidpoints[i]
        previouslyUnseen[image < colorMidpoint] = 0
        # closestColors[image < colorMidpoint] = pureColors[i]
        newInfo = justSeen - previouslyUnseen
        currentColorArr = np.ones(image.shape) * pureColors[i]
        closestColors = np.where(newInfo >= 1, currentColorArr, closestColors)

        if printIt:
            print("Current Midpoint: ", colorMidpoint)
            print("Previously Unseen: \n", previouslyUnseen)
            print("New Points: \n", newInfo)
            print("Updated Closest Colors: \n", closestColors)

        justSeen = np.copy(previouslyUnseen)
        if i == len(colorMidpoints) - 1:
            closestColors[image >= colorMidpoint] = pureColors[i+1]
            previouslyUnseen[image >= colorMidpoint] = 0
            # ^Not necessary, aside from sanity check

            if printIt:
                print("Final Updated Closest Colors: \n", closestColors)
                print("Final Unseen: \n", previouslyUnseen)
                print("Final New Points: \n", justSeen - previouslyUnseen)

    distFromClosests = np.abs(image - closestColors)**2
    distOverallReduced = np.sum(distFromClosests) / image.size

    biggestPossibleGapInfo = [pureColors[0]] + colorMidpoints + [pureColors[-1]]
    allGaps = [biggestPossibleGapInfo[i+1] - biggestPossibleGapInfo[i] for i in
                      range(len(biggestPossibleGapInfo) - 1)]
    largestPossiGap = max(allGaps)
    normFactor = largestPossiGap**2
    distOverall = distOverallReduced / normFactor

    return distOverall


def convexMinimization(params, metric=convexComboMetricEval1Cluster):
    """
    clusterImages is the usual parameters we're using to form a convex combination,
    we're then evaluating the convex combo of the images via the metric,
        which must be better when its smaller.
    The metric is only fully defined when it has the params
    """
    # Prior where everything's equally weighted
    initialLambdaGuesses = np.ones(len(params)) / len(params)
    cons = {'type': 'eq', 'fun': con}  # Enforce sum of lambdas = 1
    bnds = [(0, 1)]*len(params)  # Each lambda is between 0 and 1
    arguments = params

    finalLambdas = minimize(metric, initialLambdaGuesses,
                            method='SLSQP', args=arguments,
                            bounds=bnds, constraints=cons)

    return finalLambdas


def con(lambdas): # Trivial convex combination constraint
    return np.sum(lambdas) - 1


def get_opt_convex_combo(images, alpha0=0.1, n_clusters0=2):
    """
    Finds the convex combination parameters which minimize my metric for the
    given alpha parameterizing my affinity matrix construction, and the given
    number of clusters, and returns the metric score for the combination

    Adjust: Need to have subsets while not hitting criterian
            Need to have differentiation/equiv w. respect to alpha
            Need to somehow change number of clusters
                (Could run on a given number of clusters, optimize for given
                 number of clusters, adjust, then do descent with respect to number
                 of clusters vs min value after alpha's been tuned)

    Q: How do find nice cluster guesses for n_clusters based on affinity matrix?
            How does one turn affinity matrix into spectral decomp? Just SVD and look
            for anomously large jumps in singular values?
                1) SVD affinities
                2) sort Singular Values
                3) Calc diff's between singular values (as a percentage increase or absolute increase?)
                    Eg: 1 -> 2 is 200% or +1 ?
                4) Find stv of diff's between singular values
                5) Choose the largest diff (or next largest difference if first has already been tried)
    Q: How to tune alpha in metric between TV & ClosestLabelDist^2 ?
    Q: Scaling of SVD with full image set a problem? SVD scales as (numImgs)^3 I think?
    Q: Is my metric really smooth in terms of gamma? Jax seems to want that. New gamma's should give
        totally new clusters, which will then have different metric values?
    """
    images = 5  # Load Images
    images = np.array(images)
    affinities = affinityMatrix(images, alpha0)
    clusters = performClustering(affinities, n_clusters0)
    all_cluster_labels = set(clusters)
    total_metric_value = 0
    for cluster_label in all_cluster_labels:
        relevant_image_indices = [i for i, x in enumerate(clusters) if x == cluster_label]
        relevant_images = images[relevant_image_indices]
        convexCombo = convexMinimization(relevant_images)
        min_metric_value = convexCombo['fun']
        total_metric_value += min_metric_value

def clustering_objective(gamma, images, metric, n_clusters, img_names, print_it=False):
    """
    Calculates the total metric value after convex minimization
    for a fixed gamma and number of clusters, and the convex combination
    params in terms of the items in the cluster
    """
    affinities = affinityMatrix(images, gamma)
    clusters = performClustering(affinities, n_clusters)
    if print_it:
        print(clusters)
    all_cluster_labels = set(clusters)
    total_metric_value = 0
    lambdas_dict = dict()
    for cluster_label in all_cluster_labels:
        relevant_image_indices = [i for i, x in enumerate(clusters) if x == cluster_label]
        relevant_images = np.array(images)[relevant_image_indices]
        relevant_clus_names = np.array(img_names)[relevant_image_indices]
        convexCombo = convexMinimization(relevant_images, metric)
        min_metric_value = convexCombo['fun']
        clus_lambdas = convexCombo['x']
        frac = len(relevant_images) / len(images)
        total_metric_value += min_metric_value*frac
        lambdas_and_indices = list(zip(clus_lambdas, relevant_image_indices))
        lambdas_dict[cluster_label] = lambdas_and_indices

    return total_metric_value, lambdas_dict


def basic_ex(use_new_data=False, n_cs=5, arraysPerCluster=3, 
             save_centroids=False, print_it=False, graph_it=True, display_clusts=True):
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
    for num_clusters in range(len(data)):
        metric, clus_info = clustering_objective(
            gamma=0.01, images=data, metric=convexComboMetricEval1Cluster,
            n_clusters=num_clusters + 1, img_names=names)
        num_approx_zero_lambdas_current_clus = extract_num_non_zero_lambdas(clus_info)
        centroids = create_centroids(clus_info, data)
        if display_clusts:
            if num_clusters + 1 == my_n:
                for centroid_num, centroid in centroids.items():
                    opn.heatMapImg(centroid)

        ys.append(metric)
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
            metric, clus_info = clustering_objective(
                gamma=0.01, images=data, metric=convexComboMetricEval1Cluster,
                n_clusters=num_clusters + 1, img_names=names)
            ys.append(metric)
        pred_n_cs = np.amin(ys)
        if pred_n_cs == n_cs:
            num_cor += 1

    print( num_cor / samples )


def single_cluster(n_clusters, imgs):
    total_metric_value, lambdas_dict = clustering_objective(gamma, imgs, metric, n_clusters)


def generate_stats(n_cs=5, arraysPerCluster=5, num_samples=100):
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

if __name__ == "__main__":
    # affinityMatrix(images, gamma=0.1)
    # basic_ex()
    #basic_ex(use_new_data=True, n_cs=5)
    # how_often_correct(samples=15)
    generate_stats(n_cs=10, arraysPerCluster=25, num_samples=40)