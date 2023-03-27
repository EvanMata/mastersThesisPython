import numpy as np
import ot
import warnings
#import ott

def ot_basicTest(l=4):
    v1, v2 = [0.01] * l, [0.01] * l
    v1[1] = 1
    v2[-1] = 1
    '''
    costs = np.array([[0.01,0.01,0.01,0.01],
                      [0.01,0.01,0.01,0.01],
                      [0.01,0.01,0.01,0.01],
                      [0.01,0.01,0.01,0.01]])
    '''
    costs = np.array([[0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]] )

    costs2 = np.array([[1.0,1.0,1.0,1.0],
                       [1.0,1.0,1.0,1.0],
                       [1.0,1.0,1.0,1.0],
                       [1.0,1.0,1.0,1.0]])

    costs3 = np.array([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]])
    '''
    costs4 = np.array([[1,0.01,0.01,0.01],[0.01,1,0.01,0.01],[0.01,0.01,1,0.01],[0.01,0.01,0.01,1]]  )

    costs5 = np.array([[0.01,0.01,0.01,0.01], [0.01,0.01,0.01,0.01], [0.01,0.01,0.01,0.01], [0.01,0.01,0.01,0.01]])
    
    costs6 = np.array( [ \
             [1, .3, .3, 1], \
             [.98, 1, 4, 1], \
             [1, 2, 1, 2], \
             [1, 1, 0, 0] ])
    # Costs6 gets a divide by 0 error... while none of the others do? Why?
    '''
    print("Cost Matrices: ")
    print()
    print(costs)
    print()
    print(costs2)
    print()

    print(costs3)
    print()
    '''
    print(costs4)
    print()
    print(costs5)
    print()
    print(costs6)
    print()
    '''
    lambd1 = 10e-1
    lambd2 = 10e-1
    lambd3 = 10e-2
    Gs = ot.sinkhorn(v1, v2, costs, lambd1)

    Gs2 = ot.sinkhorn(v1, v2, costs2, lambd2)

    Gs3 = ot.sinkhorn(v1, v2, costs3, lambd3)
    #'''
    Gs4 = ot.sinkhorn(v1, v2, costs4, lambd)
    
    Gs5 = ot.sinkhorn(v1, v2, costs5, lambd)
    
    Gs6 = ot.sinkhorn(v1, v2, costs6, lambd)
    #'''
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
    '''
    print()
    print(np.around(Gs4, 3))
    
    print()
    print(np.around(Gs5, 3))
    
    print()
    print(np.around(Gs6, 3))
    '''

def baryCenterTest():
    reg = 10e-3
    a1 = [1, 0.01, 0.01, 0.01]
    a3 = [0.01, 0.01, 1, 0.01]
    #'''
    costMatrix = [[0.01, 1, 1, 1], \
                  [1, 0.01, 1, 1], \
                  [1, 1, 0.01, 1], \
                  [1, 1, 1, 0.01] ]
    costMatrix = np.array(costMatrix)
    dataPts = (a1, a3)
    a = np.vstack(dataPts).T

    print( ot.barycenter(a, costMatrix, reg) )
    print( ot.bregman.barycenter(a, costMatrix, reg))

    weights = [1 / len(dataPts)]*len(dataPts)
    print( ot.lp.barycenter(a, costMatrix, weights))

def baryCenExceptTest():
    '''
    Try to test for exceptions when calcing BaryCenters
    '''
    reg = 10e-3
    a1 = [1, 0.01, 0.01, 0.01]
    a3 = [0.01, 0.01, 2, 0.01]
    # '''
    costMatrix = [[0.01, 1, 1, 1], \
                  [1, 0.01, 1, 1], \
                  [1, 1, 0.01, 1], \
                  [1, 1, 1, 0.01]]
    costMatrix = np.array(costMatrix)
    dataPts = (a1, a3)
    a = np.vstack(dataPts).T
    warnings.filterwarnings("error")
    try:
        print(ot.bregman.barycenter(a, costMatrix, reg))
    except:
        print("caught")

if __name__ == "__main__":
    #ot_basicTest(l=4)
    baryCenExceptTest()
    # ott_basicTest(l=4)