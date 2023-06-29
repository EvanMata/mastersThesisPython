import openRawData as opn
import pathlib_variable_names as my_vars

import numpy as np
import pandas as pd
import random as rm
import matplotlib.pyplot as plt

import ot
import pickle
import pathlib

from pathlib import Path
from itertools import product
from scipy.spatial import distance

def tinySampleMatrix(n=4):
    #Create a random sample nxn matrix
    sampMatrix = np.random.randint(10, size=n*n)
    sampMatrix.resize(n,n)
    print(sampMatrix)
    return sampMatrix

def subtractHolos(n1='00001', n2='05000'):
    # Take the difference between 2 holograms
    n1Mat = opn.openBaseHolo(n1)
    n2Mat = opn.openBaseHolo(n2)
    diff = n1Mat - n2Mat
    opn.heatMapImg(diff)

def grabMiniSample(nPull):
    sampleHoloNumbers = rm.sample(range(2000), nPull)
    # Pull out the weird NA bad histos we couldn't use
    sampleHoloNumbers = set(sampleHoloNumbers) - \
                        set([0, 615, 616, 633, 634, 638, 1739, 1740, 1741])
    sampleHoloNumbers = np.array(list(sampleHoloNumbers))
    return sampleHoloNumbers

def getManyHoloArrays(nPull=300, miniSample=True, useMasks=True, proced=True, flatten=True):
    '''
    Return an array of all the pixel values in nPull'd
    random histograms, miniSample=True if using Small Sample
    '''
    if miniSample:
        sampleHoloNumbers = grabMiniSample(nPull)
    else:
        sampleHoloNumbers = rm.sample(range(28800), nPull)
    allData = []
    for holoNumber in sampleHoloNumbers:
        holoString = str(holoNumber).zfill(5)
        holoMat = opn.openBaseHolo(holoString, proced)
        if useMasks:
            maskMat = opn.openBaseHolo(holoString, proced, mask=True)
            holoMat = holoMat * maskMat
        holoMatFlat = holoMat.flatten()
        allData.append(holoMatFlat)
    allData = np.array(allData)
    if flatten:
        allData.flatten()
    return allData

def getVariationOfMaximums(nPull, printIt=True):
    holoMatrices = getManyHoloArrays( \
        nPull, miniSample=True, useMasks=True, proced=True, flatten=False)
    maxis = [np.max(holoMatrix) for holoMatrix in holoMatrices]
    varOfMax = np.var(maxis)
    plt.hist(maxis, bins=15)
    plt.title("Distribution of Maximum value in masked, processed holograms")
    plt.show()
    print("Variance of Maximums:, %f" %varOfMax)
    return varOfMax

def avgHoloHist(nPull=300, specialBins=True, miniSample=True):
    allData = getManyHoloArrays(nPull, miniSample)
    allData = np.hstack(allData)
    #print(allData.shape)
    if specialBins:
        bins = [-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7]
    else:
        bins = 30
    plt.hist(allData, bins=bins, density=True)
    plt.title("Histogram of Masked, Processed, Hologram Weights")
    plt.show()

def findEpsi(perc = 5, nPull=300, miniSample=True):
    '''
    Using nPull samples (from small sample if miniSample is true)
    find the perc-th percentile of my non-zero values. Also what
    proportion of my holo values are 0's (~35%) across my nPull'd
    '''
    allData = getManyHoloArrays(nPull, miniSample)
    allData = np.hstack(allData)
    allData = np.abs(allData)
    nonZeros = np.array([i for i in allData if i != 0])
    epsiToUse = np.percentile(nonZeros, perc)
    percentageZeros = 1 - len(nonZeros) / len(allData)
    return epsiToUse, percentageZeros

def findComparisons(nCompare, adjMatrix, printIt=False):
    '''
    For each holo find nCompare indices of Holograms that are closest
    by correlation to the given holo
    '''
    nCompare = nCompare + 1 #Will always include index 0, 100% correlated w. self
    indices = np.argsort(adjMatrix, axis=0)[-nCompare:]
    itemsToCompare = []
    for i in range(len(adjMatrix)):
        row_i = adjMatrix[i]
        indicesClosestToRow = [ item[i] for item in indices ]
        corrDistances = [adjMatrix[i] for i in indicesClosestToRow]
        itemsToCompare.append(indicesClosestToRow)
        if printIt:
            print(indicesClosestToRow)
            print(corrDistances)
            print()
    return itemsToCompare

def TfindComparisons(n=4, nCompare=2):
    sampleMatrix = tinySampleMatrix(n)
    findComparisons(nCompare, sampleMatrix, printIt=True)

def ot2Holos(holo1, holo2, addConst=True, normalize=True):
    # Finds the OT Sinkhorn distance between the 2 given holos
    n, m = holo1.shape
    M = otEucliadianCost(n, m, normalize=True) #Cost normalization
    lambd = 2e-3

    holoA = preprocHoloForOT(holo1, addConst, normalize) #Normalize array
    holoB = preprocHoloForOT(holo2, addConst, normalize)

    Gs = ot.emd2(holoA, holoB, M)
    return Gs

def preprocHoloForOT(holo1, addConst, normalize=True):
    '''
    For OT need all positive values. Also need the holograms to be 1D sized
    properly, but can't adjust the holo itself so make a copy.
    '''
    holoA = holo1.copy()
    # Negative values mess up OT. Found min of a sample was ~ -6.3, so add 7 to all values
    # + const accross the board doesn't change OT dist
    epsi = 8
    if addConst:
        holoA += epsi
    if normalize:
        holoA = holoA / np.sum(holoA)

    holoA = holoA.reshape((holoA.size,))
    return holoA

def otEucliadianCost(n,m,normalize=True):
    '''
    Given n and m, calculate the euclian dist between
        [n1,m1], [n2,m2] for all n,m.
    Then flatten into a 2d array?
    '''
    possi_points = list(product(range(n), range(m)))
    edists = distance.cdist(possi_points, possi_points, 'euclidean')
    if normalize:
        edists = edists / np.amax(edists)
    return edists

#Straight from the interwebz, bc scikit-image doesn't want to install :/
def pooling(mat,ksize,method='max',pad=False):
    '''Non-overlapping pooling on 2D or 3D data.

    <mat>: ndarray, input array to pool.
    <ksize>: tuple of 2, kernel size in (ky, kx).
    <method>: str, 'max for max-pooling,
                   'mean' for mean-pooling.
    <pad>: bool, pad <mat> or not. If no pad, output has size
           n//f, n being <mat> size, f being kernel size.
           if pad, output has size ceil(n/f).

    Return <result>: pooled matrix.
    '''

    m, n = mat.shape[:2]
    ky,kx=ksize

    _ceil=lambda x,y: int(np.ceil(x/float(y)))

    if pad:
        ny=_ceil(m,ky)
        nx=_ceil(n,kx)
        size=(ny*ky, nx*kx)+mat.shape[2:]
        mat_pad=np.full(size,np.nan)
        mat_pad[:m,:n,...]=mat
    else:
        ny=m//ky
        nx=n//kx
        mat_pad=mat[:ny*ky, :nx*kx, ...]

    new_shape=(ny,ky,nx,kx)+mat.shape[2:]

    if method=='max':
        result=np.nanmax(mat_pad.reshape(new_shape),axis=(1,3))
    else:
        result=np.nanmean(mat_pad.reshape(new_shape),axis=(1,3))

    return result

def check_change(holoNumber='00001'):
    '''
    Check how much a hologram varies between each pixel pair
        (after flattening into 1d array)
    '''
    n1Mat = opn.openBaseHolo(holoNumber)
    n1Vec = n1Mat.flatten()
    print(n1Vec.shape)
    diffs = [n1Vec[i+1] - n1Vec[i] for i in range(len(n1Vec) - 1)]
    diffs = sorted(diffs)
    sortedDiffsHighEnd = diffs[-10000:] # ~10k/1M or the top 1% of differences
    plt.hist(sortedDiffsHighEnd)
    plt.show()

def holoHisto(poolSize=10, holoNumber='00001', customBins=False):
    holoMat = opn.openBaseHolo(holoNumber)
    maxPooledHolo = pooling(holoMat,ksize=(poolSize, poolSize),\
                            method='max',pad=True)
    vecHolo = maxPooledHolo.flatten()
    print(min(vecHolo))
    if customBins:
        plt.hist(vecHolo, density=True, bins=customBins)
    else:
        plt.hist(vecHolo, density=True, bins=150)
    plt.title("Pooled Hologram Histogram")
    plt.show()

def histoCorrDistances(downTo=1000):
    dists = opn.openAdjMatrix(downTo=downTo)
    dists = np.triu(dists).flatten()
    dists = [d for d in dists if d != 0]
    plt.hist(dists, density=True, bins=100)
    plt.show()

def calcCorrClusCentroids(excludeZeros=True, threshold=0.05):
    '''
    State 0 seems to be Broken ones for some reason. So exclude them.
    Use threshold to only include clusters with at least that percent
    of the total data points.

    Returns a dictionary:
        clusterFiles of clusterLabel : [cluster filenames]
        eg:   '1': ['00001.bin', '00002.bin',...,'abcde.bin']
              '2': ['abcdf.bin', ...,]
    '''
    df = opn.parseDataTable()
    #clusterLabels = df[' State:'].unique()
    if np.abs(threshold) < 1:
        threshold = int( threshold * df.shape[0] )

    stateCountsDf = pd.DataFrame( df[' State:'].value_counts() )
    validClusterLabels = stateCountsDf[stateCountsDf[' State:'] > \
                                       threshold].index.tolist()

    clusterFiles = dict()
    for clusLabel in validClusterLabels:
        if clusLabel == 0 and excludeZeros:
            continue
        else:
            miniDf = df.loc[df[' State:'] == clusLabel, ' FileName:']
        clusterFilenames = list(miniDf)
        clusterFiles[str(clusLabel)] = [c.strip() for c in clusterFilenames]

    return clusterFiles

def calcCentroids(clusterFiles, useMasks=False, saveCentroids=False, baryCenters=False, full=False):
    '''
    Calculates the Centroid of the given clusters by averaging each item
    in the cluster together.

    Returns
    --------
    centroidsDict : dict of str to np.array
                            clusterId to Array
    '''
    centroidsDict = dict()
    for clusterName, clusterFilenames in clusterFiles.items():
        clusterCentroid = np.zeros((43,43))
        clusterMatrices = []
        for clusterFilename in clusterFilenames:
            clusterFilename = clusterFilename.strip('.bin')
            holoMatrix = opn.openBaseHolo(clusterFilename, proced=True)
            if useMasks:
                holoMask = opn.openBaseHolo(clusterFilename, proced=True, mask=True)
                clusterMatrix = holoMatrix * holoMask

            clusterMatrices.append(clusterMatrix)

        if baryCenters:
            clusterCentroid = calcBaryCenter(clusterMatrices)
        else:
            clusterCentroid = calcCentroid(clusterMatrices)

        if saveCentroids:
            if baryCenters:
                nameCore = "barycenters_centroid_%d"
            else:
                nameCore = "correlation_centroid_%d"

            if full:
                clusterCentroidPath = pathlib.Path( my_vars.centroidsFolderF )
            else:
                clusterCentroidPath = pathlib.Path( my_vars.centroidsFolderS )
            clusterCentroidPath = str( clusterCentroidPath.joinpath(nameCore) )
            clusterCentroidPath = clusterCentroidPath% int(clusterName) + ".pkl"
            with open(clusterCentroidPath, 'wb') as f:
                pickle.dump(clusterCentroid, f)

        centroidsDict[clusterName] = clusterCentroid

    return centroidsDict

def calcBaryCenter(clusterMatrices):
    '''
    Calcs the Wasserstein Barycenter
    '''
    reg = 10e-4
    n, m = clusterMatrices[0].shape
    M = otEucliadianCost(n, m, normalize=True)
    clusterMatrices = [preprocHoloForOT(c, addConst=True, normalize=True) for c in clusterMatrices]
    distrosMatrix = np.vstack(clusterMatrices).T
    baryWass = ot.bregman.barycenter(distrosMatrix, M, reg)
    clusterCentroid = baryWass.reshape((n, m))

    '''
    clusSums = [np.sum(c) for c in clusterMatrices]
    sum1 = clusSums[0]
    sameVals = [c == sum1 for c in clusSums]
    for i in range(len(sameVals)):
        if not sameVals[i]:
            diff = np.abs(clusSums[i] - sum1)
            if diff > 0.0000005:
                print(diff)
    '''

    '''
    mins = [min(c) for c in clusterMatrices]
    for m in mins:
        if m < 0:
            print(m)
    '''

    return clusterCentroid

def calcCentroid(clusterMatrices):
    clusTot = np.sum(clusterMatrices) / len(clusterMatrices)
    return clusTot



def visClusCentroid(centroidsDict):
    for cluster, centroid in centroidsDict.items():
        if cluster == '1':
            centroid_1 = centroid
    opn.heatMapImg(centroid_1)

def compareHoloToCentroids(holoNumber, centroidsDict, holoArray=False, useMask=False, useOt=True):
    '''
    Calculates the optimal transport sinkhorn distance between the
    given cluster / array and each of the centroids in our centroids dict.
    UseMask if you want to compare the masked holo to the cluster centroid.

    Returns:
    --------
    otDists - lst of (clus num, float), length = number of eval'd clusters
                      item i[1] is otDist between the given holo
                        and cluster i[0]
    '''
    otDists = []
    if not holoArray:
        holoArray = opn.openBaseHolo(holoNumber, proced=True)
        if useMask:
            maskArray = opn.openBaseHolo(holoNumber, proced=True, mask=True)
            holoArray = holoArray * maskArray


    for cluster, centroidArray in centroidsDict.items():
        if useOt:
            holoDist = ot2Holos(holoArray, centroidArray, addConst=True)
        else:
            holoDist = calcCorrMatlabEquiv(holoArray, centroidArray, maskArray, maskArray)
        otDists.append( (cluster, round(holoDist, 8)) )

    return otDists

def calcAcc(useMasks=True, printProgress=True, wassBaryCens=False, useOt=True):
    '''
    Goes through each hologram in the classes with
    more than the relevant threshold samples (see calcCorrClusCentroids)
    and calculates which cluster its closest to by Optimal Transport
    distance (if useOt) or correlation distance.
    If its the cluster it's assigned to, then its accurate

    Gets an accuracy measure for each cluster.

    '''
    numerator = 0
    denominator = 0
    clusterAccs = dict()
    cPathsDict = calcCorrClusCentroids()
    cDict = calcCentroids(cPathsDict, saveCentroids=False, \
                            useMasks=useMasks, baryCenters=wassBaryCens)

    print("Cluster Centroids Computed \n")

    numFilesToGetThrough = 0
    for clus, fileNumbers in cPathsDict.items():
        numFilesToGetThrough += len(fileNumbers)
        clusterAccs[clus] = 0

    for clus, fileNumbers in cPathsDict.items():
        for fileNumber in fileNumbers:
            fileNum = fileNumber.strip(".bin")
            otDists = compareHoloToCentroids( \
                holoNumber=fileNum, centroidsDict=cDict, \
                holoArray=False, useMask=useMasks, useOt=useOt)
            closest = getBestCluster(otDists, useOt)
            if closest == clus:
                numerator += 1
                clusterAccs[clus] += 1
            denominator += 1
            percentFinished = denominator / numFilesToGetThrough
            if (denominator % 100 == 0) and printProgress:
                print( "Percentage Finished: %f"%percentFinished )
                tempAcc = numerator / denominator
                print( "Accuracy at ^ above percentage: %f"%tempAcc )
                print(  ) #OVER ALL CLASSES< BUT NOT BALANCED IN LOOP

        clusterAccs[clus] = clusterAccs[clus] / len(fileNumbers)
    print()
    print("FINAL ACCURACY: ") #Could keep track of acc by cluster.
    print( numerator / denominator )
    print()
    print("ACCURACY BY CLUSTER: ")
    print(clusterAccs)
    return clusterAccs

def getBestCluster(otDists, useOt=False):
    if useOt: #Lower OT Cost = Better
        tup = min(otDists, key=lambda t: t[1])
    else: #Higher Correlation = Better
        tup = max(otDists, key=lambda t: t[1])
    return tup[0]

def calcCorrMatlabEquiv(arr1, arr2, mask1, mask2):
    '''
    Calculates the correlation between 2 arrays the same way as the Matlab code does. 
    NOTE: When calculating the comparison with a cluster centroid, there is no centroid mask
    So our process isn't exactly equiv (they don't compare centroids to indivi frames, only
    between individi frames, each of which would have its own mask)
    '''
    mask = mask1.astype(int) | mask2.astype(int)
    arrA = arr1*mask
    arrB = arr2*mask
    normFactor1 = np.sum(arrA**2)
    normFactor2 = np.sum(arrB**2)
    arr = arrA*arrB/np.sqrt(normFactor1*normFactor2)
    corrMatlab = np.sum(arr)
    #print(corrMatlab)
    return corrMatlab



if __name__ == "__main__":
    cPathsDict = calcCorrClusCentroids()
    cDict = calcCentroids(cPathsDict, saveCentroids=True, useMasks=True, baryCenters=False)
    #corrs1 = compareHoloToCentroids(holoNumber='00005', centroidsDict=cDict, holoArray=False, useMask=True, useOt=False)
    #print(corrs1)
    #print( getBestCluster( corrs1 ) )

    #calcAcc(useMasks=True, useOt=True, wassBaryCens=True)