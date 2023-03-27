
from ott.solvers.linear import sinkhorn

import ott

import jax.numpy as jnp

def q1():
    '''
    Looks at vector with itself

    Results: Many outputs, documentation unclear. Seems like 1st, 2nd output or 4th (of 5)
    outputs are what I want. -inf value means.. 0 cost? Or can't work w. 0 values / pmf?
    Index 4 seems to be the overall cost, f and g are ??? costs to move to the given index?
    '''

    x1 = jnp.array( [1,0,0,0] )
    c1 = jnp.array( [ [0,1,1,1], \
                      [1,0,1,1], \
                      [1,1,0,1], \
                      [1,1,1,0] ] )
    g1 = ott.geometry.geometry.Geometry(c1)
    sink = sinkhorn.solve(g1, a=x1, b=x1)
    #print(len(sink)) #5 outputs, looks like the 1st or 2nd result is what I want?
    print( sink )

def q2():
    '''
    Looks at vector moving all mass to a new bin.
    '''

    x1 = jnp.array( [1,0,0,0] )
    x2 = jnp.array( [0,1,0,0] )
    c1 = jnp.array( [ [0,1,1,1], \
                      [1,0,1,1], \
                      [1,1,0,1], \
                      [1,1,1,0] ] )
    g1 = ott.geometry.geometry.Geometry(c1)
    sink = sinkhorn.solve(g1, a=x1, b=x2)
    print(len(sink)) #5 outputs, looks like the first 2 are what I want?
    print( sink )

def q3():
    '''
    Looks at vector moving all mass to another new bin.
    '''

    x1 = jnp.array( [1,0,0,0] )
    x2 = jnp.array( [0,0,1,0] )
    c1 = jnp.array( [ [0,1,1,1], \
                      [1,0,1,1], \
                      [1,1,0,1], \
                      [1,1,1,0] ] )
    g1 = ott.geometry.geometry.Geometry(c1)
    sink = sinkhorn.solve(g1, a=x1, b=x2)
    print(len(sink)) #5 outputs, looks like the first 2 are what I want?
    print( sink )

def q4():
    '''
    Looks at vector moving all mass to a new bin from a diff starting bin
    '''

    x1 = jnp.array( [0,1,0,0] )
    x2 = jnp.array( [0,0,1,0] )
    c1 = jnp.array( [ [0,1,1,1], \
                      [1,0,1,1], \
                      [1,1,0,1], \
                      [1,1,1,0] ] )
    g1 = ott.geometry.geometry.Geometry(c1)
    sink = sinkhorn.solve(g1, a=x1, b=x2)
    print(len(sink)) #5 outputs, looks like the first 2 are what I want?
    print( sink )

def q5():
    '''
    Looks at splitting mass from input to output, half should have 0 cost, and half should have 1 cost

    Definitely seems like item 4, reg_ot_cost, is what I want.
    '''

    x1 = jnp.array( [0,1,0,0] )
    x2 = jnp.array( [0,0.5,0.5,0] )
    c1 = jnp.array( [ [0,1,1,1], \
                      [1,0,1,1], \
                      [1,1,0,1], \
                      [1,1,1,0] ] )
    g1 = ott.geometry.geometry.Geometry(c1)
    sink = sinkhorn.solve(g1, a=x1, b=x2)
    print(len(sink)) #5 outputs, looks like the first 2 are what I want?
    print( sink )

def q6():
    '''
    Looks at combining mass from input to output, half should have 0 cost, and half should have 1 cost
    '''

    x1 = jnp.array( [0,0.5,0.5,0] )
    x2 = jnp.array( [0,0,1,0] )
    c1 = jnp.array( [ [0,1,1,1], \
                      [1,0,1,1], \
                      [1,1,0,1], \
                      [1,1,1,0] ] )
    g1 = ott.geometry.geometry.Geometry(c1)
    sink = sinkhorn.solve(g1, a=x1, b=x2)
    print( sink )

def q7():
    '''
    Looks at what occurs if there's cost to move to yourself, but nothing in a bin.
    '''

    x1 = jnp.array( [0,0,1,0] )
    x2 = jnp.array( [0,0,1,0] )
    c1 = jnp.array( [ [1,1,1,1], \
                      [1,1,1,1], \
                      [1,1,0,1], \
                      [1,1,1,1] ] )
    g1 = ott.geometry.geometry.Geometry(c1)
    sink = sinkhorn.solve(g1, a=x1, b=x2)
    print( sink[3] )

def q8():
    '''
    Looks at what occurs if there's more mass in input but 0 cost for moving mass

    GETS SIGNIFICANT COST (unclear why)
    '''

    x1 = jnp.array( [0,0,2,0] )
    x2 = jnp.array( [0,0,1,0] )
    c1 = jnp.array( [ [1,1,1,1], \
                      [1,1,1,1], \
                      [1,1,0,1], \
                      [1,1,1,1] ] )
    g1 = ott.geometry.geometry.Geometry(c1)
    sink = sinkhorn.solve(g1, a=x1, b=x2)
    print( sink )

def q9():
    '''
    Looks at what occurs if there's more mass in output but 0 cost for moving mass

    GETS SIGNIFICANT COST (unclear why)
    '''

    x1 = jnp.array( [0,0,1,0] )
    x2 = jnp.array( [0,0,2,0] )
    c1 = jnp.array( [ [1,1,1,1], \
                      [1,1,1,1], \
                      [1,1,0,1], \
                      [1,1,1,1] ] )
    g1 = ott.geometry.geometry.Geometry(c1)
    sink = sinkhorn.solve(g1, a=x1, b=x2)
    print( sink )

def t1():
    '''
    Tests not trivial cost matrix with same total mass.

    Somehow more mass creeps into the problem, just a tiny bit though? ~1/1000
    Is it the sinkhorn error / precision?
    '''

    x1 = jnp.array([0, 1, 1, 0])
    x2 = jnp.array([0, 0, 0, 2])
    c1 = jnp.array([[0, 0, 0, 0], \
                    [0, 0, 0, .5], \
                    [0, 0, 0, 3], \
                    [0, 0, 0, 0]])
    g1 = ott.geometry.geometry.Geometry(c1)
    sink = sinkhorn.solve(g1, a=x1, b=x2)
    print(sink[3])
    print("Expected 3.5")

def t2():
    '''
    Tests not trivial cost matrix with same total mass.

    Somehow more mass creeps into the problem, just a small bit though? ~1/100
    Is it the sinkhorn error / precision?
    1/100 relative to the 3, but only ~1/5 relative to the .5
    '''

    x1 = jnp.array([0, 1, 1, 0])
    x2 = jnp.array([0, 0, 0, 2])
    c1 = jnp.array([[1, 1, 1, 1], \
                    [1, 1, 1, .5], \
                    [1, 1, 1, 3], \
                    [1, 1, 1, 1]])
    g1 = ott.geometry.geometry.Geometry(c1)
    sink = sinkhorn.solve(g1, a=x1, b=x2)
    print(sink[3])
    print("Expected 3.5")

def t3():
    '''
    Tests not trivial cost matrix with same total mass.

    Only ~1/40 error from sinkhorn.
    '''

    x1 = jnp.array([0, 1, 1, 0])
    x2 = jnp.array([0, 0, 0, 2])
    c1 = jnp.array([[1, 1, 1, 1], \
                    [1, 1, 1, .5], \
                    [1, 1, 1, .3], \
                    [1, 1, 1, 1]])
    g1 = ott.geometry.geometry.Geometry(c1)
    sink = sinkhorn.solve(g1, a=x1, b=x2)
    print(sink[3])
    print("Expected 0.8")

def t4():
    '''
    Tests not trivial cost matrix with same total mass.


    '''

    x1 = jnp.array([2, 1, 1, 0])
    x2 = jnp.array([0.5, 0.5, 1, 2])
    c1 = jnp.array([[0, 99, 99, 1], \
                    [99, 0, 99, 1], \
                    [99, 99, 0, 1], \
                    [99, 99, 99, 0]])
    g1 = ott.geometry.geometry.Geometry(c1)
    sink = sinkhorn.solve(g1, a=x1, b=x2)
    print(sink[3])
    print("Expected 2 = 1.5 + 0.5")
    print()
    print(sink)

def t5():
    '''
    Tests not trivial cost matrix with same total mass.

    '''

    x1 = jnp.array([2, 1, 1, 0])
    x2 = jnp.array([0.5, 0.5, 1, 2])
    c1 = jnp.array([[0, 99, 99, 99], \
                    [99, 0, 99, 99], \
                    [99, 99, 0, 99], \
                    [1, 1, 1, 0]])
    g1 = ott.geometry.geometry.Geometry(c1)
    sink = sinkhorn.solve(g1, a=x1, b=x2)
    print(sink[3])
    print("Expected 2 = 1.5 + 0.5")

if __name__ == '__main__':
    #q9()
    t4()
