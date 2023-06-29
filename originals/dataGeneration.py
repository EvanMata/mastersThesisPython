import math
import random
import pickle

import numpy as np

import openRawData as opn
import pathlib_variable_names as my_vars


def randomAngle(not_vert_hori=False, degrees_threshold=10.0):
    # Generate a random direction on a 2d sphere.
    d1 = np.array([1, 0])
    d2 = np.array([0, 1])
    if not_vert_hori:
        while angle_dist < degrees_threshold:
            direction = gen_random_angle()
            angle_dist = max([vec_vertical(direction), vec_horizontal(direction)])

    else:
        direction = gen_random_angle()

    return direction

def gen_random_angle():
    twoPts = np.random.normal(size=2)
    direction = twoPts / np.sqrt(np.sum(twoPts ** 2))
    return direction

def vec_to_slope(vec):
    #Given a normalized vector, assumed to be coming from origin, find its slope.
    return vec[1]/vec[0] #dy/dx


def genClusters(numClusters=3, arraysPerCluster=5, n1=100, n2=100,
                noiseLv=0.1, display=False, saveClusters=True,
                save_folder=my_vars.generatedDataPath):
    '''
    Generate some n1 by n2 images of hyperplanes
    '''
    outImgs = []

    for i in range(numClusters):
        low1, high1 = int(0.1*n1), int(0.9*n1)
        low2, high2 = int(0.1*n2), int(0.9*n2)
        center1 = np.random.randint(low1, high1)
        center2 = np.random.randint(low1, high2)
        slope = randomAngle()
        side = np.random.binomial(1, 0.5) #symBern() #Choose which side is which color
        for j in range(arraysPerCluster):
            centerNoise1, centerNoise2 = np.random.randint(-5, 5, size=2)
            x_center = center1 + centerNoise1
            y_center = center2 + centerNoise2
            newSlope = perturbDirection(slope, plusMinus=np.pi / 16)
            newSlope = vec_to_slope(newSlope)
            offset = y_center - newSlope*x_center
            arr = genArray(offset, newSlope, side, n1, n2, noiseLv)

            if display:
                opn.heatMapImg(arr)

            clus_name = "cluster_%d_item_%d.pickle" % (i, j)
            if saveClusters:
                my_filepath = str(save_folder.joinpath(clus_name))
                with open(my_filepath,'wb') as my_filename:
                    pickle.dump(arr, my_filename)

            outImgs.append((clus_name, arr))
    return outImgs

def genArray(offset, slope, side, n1, n2, noiseLv):
    baseArr = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            if j >= i*slope + offset:
                baseArr[i,j] = side
            else:
                baseArr[i, j] = 1 - side

    noise = np.random.normal(loc=0, scale=noiseLv, size=n1*n2)
    noise = noise.reshape((n1, n2))
    arr = baseArr + noise
    arr = np.clip(arr, 0, 1)
    return arr

def vec_horizontal(vec):
    measureFrom = np.array([0.0, 1.0])
    radiansAngle = np.arccos(np.clip(np.dot(vec, measureFrom), -1.0, 1.0))
    radiansAngle = math.copysign(radiansAngle, vec[0]*vec[1]) #Flips to negative in 1/2 cases.
    degreesAngle = np.abs(math.degrees(radiansAngle))
    return degreesAngle

def vec_vertical(vec):
    measureFrom = np.array([1.0, 0.0])
    radiansAngle = np.arccos(np.clip(np.dot(vec, measureFrom), -1.0, 1.0))
    radiansAngle = math.copysign(radiansAngle, vec[0]*vec[1]) #Flips to negative in 1/2 cases.
    degreesAngle = np.abs(math.degrees(radiansAngle))
    return degreesAngle

'''


def getAngleNoise():
    #Anglenoise is a random angle uniformly distr between 0 and +- ~10 degrees
    baseAngle = np.array([0.0, 1.0])
    angleNoise = randomAngle()
    angleNoise = np.arccos(np.clip(np.dot(angleNoise, base), -1.0, 1.0)) / 20
    posNeg = symBern()
    angleNoise *= angleNoise
    return angleNoise
'''

def perturbDirection(vec, plusMinus=np.pi/32):
    perterbuationAngle = np.random.uniform(-plusMinus, plusMinus)
    rotationMatrix = np.array([[np.cos(perterbuationAngle), -np.sin(perterbuationAngle)], \
                               [np.sin(perterbuationAngle), np.cos(perterbuationAngle)]])
    newVec = np.dot(rotationMatrix, vec)
    return newVec

def symBern(a=-1,b=1):
    return [a, b][np.random.binomial(1, 0.5)]

if __name__ == "__main__":
    genClusters(numClusters=5, arraysPerCluster=3, n1=100, n2=100,
                noiseLv=0.1, display=True, saveClusters=False)