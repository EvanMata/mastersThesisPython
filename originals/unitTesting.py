
import ot

import numpy as np
import matplotlib.pyplot as plt

import clustering as clu
import openRawData as opn
import dataGeneration as dg
import variables_names as my_vars
import optimalTransportComparison as myOT

def trivialSimulate2DGaussian(n,m,mu1,mu2,showImg=True):
    dataPts = np.random.multivariate_normal((mu1, mu2), np.identity(2), 10000)
    customBins = [np.arange(-5,5,.1), np.arange(-5,5,.1)]
    bins, xedyes, yedges, img = plt.hist2d(dataPts[:,0], dataPts[:,1], bins=customBins, density=True)
    if showImg:
        plt.show()
    return bins

def checkOTDist(addConst=True):
    '''
    Check that the OT distance between holograms is sensible.
    distance from item to itself should be 0, dist to 3 should be farther / bigger than dist to 2
    '''
    h1 = trivialSimulate2DGaussian(n=43,m=43,mu1=0,mu2=0,showImg=False)
    h2 = trivialSimulate2DGaussian(n=43,m=43,mu1=.5,mu2=.5,showImg=False)
    h3 = trivialSimulate2DGaussian(n=43,m=43,mu1=5,mu2=5,showImg=False)
    d1 = myOT.ot2Holos(h1, h1, addConst)
    d2 = myOT.ot2Holos(h1, h2, addConst)
    d3 = myOT.ot2Holos(h1, h3, addConst)
    print(d1, d2, d3)

def tPreProcHoloForOT(addConst=True):
    '''
    Expect: [1,2,3,4,5,6,7,8,9,10,11,12] + 7 ->
    '''
    fakeArray = np.array( [[1,2,3],[4,5,6],[7,8,9],[10,11,12]] )
    procdHolo = myOT.preprocHoloForOT(fakeArray, addConst)
    expectedResult = np.array( range(12) ) + 1 #since 0'd
    if addConst:
        expectedResult = expectedResult + 7
    check = [procdHolo[i] == expectedResult[i] for i in range(12)]
    if all(check):
        print("Passed")
    else:
        print("Failed")

def checkCostMatrix():
    holonum = '00001'
    holo1 = opn.openBaseHolo(holonum, proced=True)
    n, m = holo1.shape
    M = myOT.otEucliadianCost(n, m, normalize=True)
    if M.trace() == 0:
        print("Passed: Trace = 0")
    farthestPts = []
    for row in range(m):
        farthestPt = M[n,int(m*(row+1)) - 1]
        farthestPts.append(farthestPt)

    print(farthestPts)
    print(M[1848,0])

def trivialOT0(l=4):
    '''
    Test no change, but everything costs
    '''
    v1, v2 = [.0001]*l, [.0001]*l
    v1[0] = 1
    v2 = v1
    costRow = [1]*l
    costs = [costRow for i in range(l)]
    print()
    print(costs)
    print()
    lambd = 2e-3
    Gs = ot.emd2(v1, v2, costs)
    print(v1)
    print(v2)
    print()
    print(Gs)
    return Gs

def trivialOT1(l=4):
    '''
    Test moving from 1 bin to another
    '''
    v1, v2 = [.0001]*l, [.0001]*l
    v1[0] = 1
    v2[-1] = 1
    costRow = [1]*l
    costs = [costRow for i in range(l)]
    print()
    print(costs)
    print()
    lambd = 2e-3
    Gs = ot.emd2(v1, v2, costs)
    print(v1)
    print(v2)
    print()
    print(Gs)
    return Gs

def trivialOT2(l=4):
    '''
    Test moving from v2 -> v1 instead of v1 -> v2;
    Result: Transpose (What if not n x n?) of output from v1 -> v2
    '''
    v1, v2 = [.0001]*l, [.0001]*l
    v1[0] = 1
    v2[-1] = 1
    costRow = [1]*l
    costs = [costRow for i in range(l)]
    print()
    print(costs)
    print()
    lambd = 2e-3
    Gs = ot.emd2(v2, v1, costs)
    print(v1)
    print(v2)
    print()
    print(Gs)
    return Gs

def trivialOT3(l=4):
    '''
    Test moving buckets a -> c, b -> d
    '''
    v1, v2 = [.0001]*l, [.0001]*l
    v1[0] = 1
    v1[1] = 1
    v2[-1] = 1
    v2[-2] = 1
    costRow = [1]*l
    costs = [costRow for i in range(l)]
    print()
    print(costs)
    print()
    lambd = 2e-3
    Gs = ot.emd2(v1, v2, costs)
    print(v1)
    print(v2)
    print()
    print(Gs)
    return Gs

def trivialOT4(l=4):
    '''
    Test moving multiple units of mass from a to b != a
    '''
    v1, v2 = [.0001]*l, [.0001]*l
    v1[0] = 2
    v2[-1] = 2
    costRow = [1]*l
    costs = [costRow for i in range(l)]
    print()
    print(costs)
    print()
    lambd = 2e-3
    Gs = ot.emd2(v1, v2, costs)
    print(v1)
    print(v2)
    print()
    print(Gs)
    return Gs

def trivialOT5(l=4):
    '''
    Test splitting into 2 buckets
    '''
    v1, v2 = [.0001]*l, [.0001]*l
    v1[0] = 2
    v2[-1] = 1.0001
    v2[-2] = 1
    costRow = [1]*l
    costs = [costRow for i in range(l)]
    print()
    print(costs)
    print()
    lambd = 2e-3
    Gs = ot.emd2(v1, v2, costs)
    print(v1)
    print(v2)
    print()
    print(Gs)
    return Gs

def basicOT1(l=4):
    '''
    What if No mass conservation? Mass Start Distro > Mass End Distro
    Cost of moving mass to output is cost of moving all mass out of bucket?
    THEN GET COST 0 AND NOT IN SIMPLEX ERROR
    '''
    v1, v2 = [.0001] * l, [.0001] * l
    v1[0] = 2
    v2[-1] = 1
    costRow = [1] * l
    costs = [costRow for i in range(l)]
    print()
    print(costs)
    print()
    lambd = 2e-3
    Gs = ot.emd2(v1, v2, costs)
    print(v1)
    print(v2)
    print()
    print(Gs)
    return Gs

def basicOT2(l=4):
    '''
    What if No mass conservation? Mass Start Distro < Mass End Distro
    Cost of moving mass to output is cost of moving all mass out of bucket?
    THEN GET COST 0 AND NOT IN SIMPLEX ERROR
    '''
    v1, v2 = [.0001] * l, [.0001] * l
    v1[0] = 1
    v2[-1] = 2
    costRow = [1] * l
    costs = [costRow for i in range(l)]
    print()
    print(costs)
    print()
    lambd = 2e-3
    Gs = ot.emd2(v1, v2, costs)
    print(v1)
    print(v2)
    print()
    print(Gs)
    return Gs

def basicOT3(l=4):
    '''
    What if No mass conservation? Mass Start Distro < Mass End Distro
    Cost of moving mass to output is cost of moving all mass out of bucket?
    ALSO SIMPLEX ERROR, COST 0
    '''
    v1, v2 = [.0001] * l, [.0001] * l
    v1[0] = 1
    v2 = [1]*l
    costRow = [1] * l
    costs = [costRow for i in range(l)]
    print()
    print(costs)
    print()
    lambd = 2e-3
    Gs = ot.emd2(v1, v2, costs)
    print(v1)
    print(v2)
    print()
    print(Gs)
    return Gs

def basicOT4(l=4):
    '''
    USE Diff Cost Matrix - no cost for moving to yourself, but cost for moving anywhere!
    '''
    v1, v2 = [.0001] * l, [.0001] * l
    v1[0] = 1
    v2 = v1
    costs = np.array([[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]])
    print()
    print(costs)
    print()
    lambd = 2e-3
    Gs = ot.emd2(v1, v2, costs)
    print(v1)
    print(v2)
    print()
    print(Gs)
    return Gs

def basicOT5(l=4):
    '''
    Make sure the value of where your mass is in a histo doesn't effect mapping back to yourself
    '''
    v1, v2 = [.0001] * l, [.0001] * l
    v1[1] = 1
    v2 = v1
    costs = np.array([[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]])
    print()
    print(costs)
    print()
    lambd = 2e-3
    Gs = ot.emd2(v1, v2, costs)
    print(v1)
    print(v2)
    print()
    print(Gs)
    return Gs

def basicOT6(l=4):
    v1, v2 = [.0001] * l, [.0001] * l
    v1[1] = 1
    v2 = v1
    costs = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    costs2 = np.array([[1,1,1,1], [1,1,1,1], [1,1,1,1], [1,1,1,1]])
    costs3 = np.array([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]])
    costs4 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]  )
    costs5 = np.array([[0.5, 1, 2, 0], [1, 2, .5, 0], [1, 1, 1, 1], [3, 1, 0, 0]])
    print()
    print(costs)
    print(costs2)
    print(costs3)
    print(costs4)
    print(costs5)
    print()
    lambd = 2e-3
    Gs = ot.emd2(v1, v2, costs, lambd)
    Gs2 = ot.emd2(v1, v2, costs2, lambd)
    Gs3 = ot.emd2(v1, v2, costs3, lambd)
    Gs4 = ot.emd2(v1, v2, costs4, lambd)
    Gs5 = ot.emd2(v1, v2, costs5, lambd)
    print(v1)
    print(v2)
    print()
    print(Gs)
    print()
    print(Gs2)
    print()
    print(Gs3)
    print()
    print(Gs4)
    print()
    print(Gs5)
    check = np.array_equiv(Gs, Gs2)
    #print(check)
    return Gs

def basicOT7(l=4):
    v1, v2 = [.0001] * l, [.0001] * l
    v1[1] = 1
    v2[-1] = 1
    costs = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    costs2 = np.array([[1,1,1,1], [1,1,1,1], [1,1,1,1], [1,1,1,1]])
    costs3 = np.array([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]])
    costs4 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]  )
    print("Cost Matrices: ")
    print()
    print(costs)
    print()
    print(costs2)
    print()
    print(costs3)
    print()
    print(costs4)
    print()
    lambd = 2e-3
    Gs = ot.emd2(v1, v2, costs, lambd) #emd2 works while sinkhorn doesn't?
    Gs2 = ot.emd2(v1, v2, costs2, lambd)
    Gs3 = ot.emd2(v1, v2, costs3, lambd)
    Gs4 = ot.emd2(v1, v2, costs4, lambd)
    print("V1: ", v1)
    print("V2: ", v2)
    print()
    print("Sinkhorn Outputs:")
    print()
    print(np.around(Gs, 3))
    print()
    print(np.around(Gs2, 3))
    print()
    print(np.around(Gs3, 3))
    print()
    print(np.around(Gs4, 3))
    return Gs

def mOT(l=4):
    v1, v2 = [.0001] * l, [.0001] * l
    v1[0] = 1
    v2 = v1
    l2 = int(np.sqrt(l))
    costs = myOT.otEucliadianCost(l2, l2, normalize=True)
    print()
    print(costs)
    print()
    lambd = 2e-3
    Gs = ot.sinkhorn(v1, v2, costs, lambd)
    print(v1)
    print(v2)
    print()
    print(Gs)
    print(Gs.trace())
    return Gs

def distFromClosests():
    colors = [0, 1, 0.5, .8]
    img = np.array([[0, 1], [0, 1], [0.4, 0.6]])
    print(img)
    print()
    print( clu.distFromPureColor(img, pureColors=colors, printIt=True) )

def convexCombo():
    params = np.array([1, 5])
    print(clu.convexMinimization(params, metric=trivialMetric))
    # should be .5, .5, since .5*1 + .5*5 = 3, which minimizes (y=3 - 3)^2
    # When offset is 3

def convexCombo2():
    params = np.array([1, 5])
    d = clu.convexMinimization(params, metric=trivialMetric)
    print(d)
    print(type(d))
    print(d['fun'])
    print(d['x'])
    print(type(d['x']))
    # should be 0, 1, when offset is 6. Turns out correct, constraints work!


def trivialMetric(lambdas, xs, offset=6):
    return (np.sum(lambdas*xs) - offset)**2

def t_angle_gen():
    dir = dg.randomAngle(not_vert_hori=False, degrees_threshold=10.0)
    print(dir)

def t_synth_data_loader():
    imgs = opn.openSyntheticData()
    for img in imgs:
        arr = img[1]
        name = img[0]
        print(name)
        opn.heatMapImg(arr)

def t_basic_clus():
    names_and_data = opn.openSyntheticData()
    data = [np.array(i[1]) for i in names_and_data]
    names = [i[0] for i in names_and_data]
    my_metric = clu.convexComboMetricEval1Cluster
    metric, lambdas_dict = clu.clustering_objective(
        gamma=0.01, images=data, metric=my_metric,
        n_clusters=3, img_names=names)
    print("Metric Value: ", metric)
    print()
    print(lambdas_dict)

'''
Things to check: 
- my TV is working correctly and always in [0,1], and lower = better
- Convex combo is working
- Affinity matrix is calculating properly, that higher = more similar 
- Metric is evaluating such that lower = better, 
- create_centroid indexing/function works for non-synth data
'''




if __name__ == '__main__':
    #trivialSimulate2DGaussian(n=43,m=43,mu1=0,mu2=0)
    #checkOTDist(addConst=True)
    #tPreProcHoloForOT()
    #checkCostMatrix()
    #trivialOT5(l=4)
    #basicOT7(l=4)
    #trivialOT1(l=4)
    #mOT(l=4)
    #distFromClosests()
    #convexCombo2()
    #t_angle_gen()
    t_synth_data_loader()
    #t_basic_clus()