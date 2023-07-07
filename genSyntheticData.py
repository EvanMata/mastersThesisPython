'''
Generates Synthetic Data for trivial clustering and metric examination
'''
import math
import random
import pickle

import numpy as np
import openRawData as opn
import pathlib_variable_names as my_vars

from pathlib import Path


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
    """
    Generates a synthetic set of clusters and points which belong in them, 
    all points are hyperplanes w. gaussian noise.

    Args:
        numClusters (int, optional): _description_. Defaults to 3.
        arraysPerCluster (int, optional): _description_. Defaults to 5.
        n1 (int, optional): _description_. Defaults to 100.
        n2 (int, optional): _description_. Defaults to 100.
        noiseLv (float, optional): _description_. Defaults to 0.1.
        display (bool, optional): _description_. Defaults to False.
        saveClusters (bool, optional): _description_. Defaults to True.
        save_folder (_type_, optional): _description_. Defaults to my_vars.generatedDataPath.

    Returns:
        outImgs (list of tups of clus name/str, img/np.array): The data and clusters it belongs too
    """
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


def perturbDirection(vec, plusMinus=np.pi/32):
    perterbuationAngle = np.random.uniform(-plusMinus, plusMinus)
    rotationMatrix = np.array([[np.cos(perterbuationAngle), -np.sin(perterbuationAngle)], \
                               [np.sin(perterbuationAngle), np.cos(perterbuationAngle)]])
    newVec = np.dot(rotationMatrix, vec)
    return newVec


def symBern(a=-1,b=1):
    return [a, b][np.random.binomial(1, 0.5)]


def gen_random_slope():
    """
    Generate a random slope by taking a pt uniform on the surface of a 2d sphere 
    and turning it into a slope
    """
    pt = np.random.standard_normal(size=2)
    normed_pt = pt / np.sqrt(pt[0]**2 + pt[1]**2)
    slope = normed_pt[1] / normed_pt[0] #y/x
    return slope

def gen_random_center(x_max=100, y_max=100, centering=0.1):
    """
    Generates a random line
    """
    x_offset = int(x_max*centering)
    y_offset = int(y_max*centering)
    center_x = np.random.randint(x_offset, x_max - x_offset + 1)
    center_y = np.random.randint(y_offset, y_max - y_offset + 1)
    return (center_x, center_y)

    
def gen_fake_data_pt(slope, center, side, x_max=100, y_max=100, noise=0.2):
    """
    Generates a synthetic data point (numpy array) given the relevant parameters.

    Args:
        slope (float): The slope the dividing line of the image should have
        center (tup of ints): x,y center point which the slope goes through
        side (int): 0|1 basically which side should be colored white and which black
        x_max (int, optional): How many pixels the image along the x axis. Defaults to 100.
        y_max (int, optional): How many pixels the image along the y axis. Defaults to 100.

    Returns:
        _type_: _description_
    """
    x_center = center[0]
    y_center = center[1]
    base_arr = np.zeros((x_max, y_max))
    for i in range(x_max):
        for j in range(y_max):
            if j > slope*i + y_center - slope*x_center:
                if side == 0:
                    base_arr[i,j] = side + np.abs(np.random.normal(scale=noise))
                else: #side == 1
                    base_arr[i,j] = side - np.abs(np.random.normal(scale=noise))
            else:
                if side == 0:
                    base_arr[i,j] = 1 - (side + np.abs(np.random.normal(scale=noise)))
                else: #side == 1
                    base_arr[i,j] = 1 - (side - np.abs(np.random.normal(scale=noise)))
                #base_arr[i,j] = 1 - side
    
    
    
    return base_arr
    
    
def preturb_slope(original_slope, max_degree_change=15):
    """
    Adjusts slope by at most +- max_degree_change

    Args:
        original_slope (_type_): _description_
        max_degree_change (_type_, optional): _description_. Defaults to np.pi/16.
    """
    plus_minus = [-1, 1][np.random.binomial(n=1,p=0.5)]
    slope_change_magnitude = np.random.uniform(-max_degree_change, max_degree_change)
    slope_change = plus_minus*slope_change_magnitude
    slope_degrees = np.rad2deg(np.arctan2(original_slope,1))
    new_slope_degrees = slope_degrees + slope_change
    new_slope_radians = np.deg2rad(new_slope_degrees)
    new_x = np.cos(new_slope_radians)
    new_y = np.sin(new_slope_radians)
    new_slope = new_y/new_x
    return new_slope
    
    
def preturb_center(old_center, max_offset=3):
    """
    preturbs the center point of a cluster uniformly by max_offset in x or y axis 

    Args:
        old_center (tuple of 2 ints): The original x,y center 
        max_offset (int, optional): How much to preturb the center by. Defaults to 3.

    Returns:
        new_center (tuple of 2 ints): x,y center after perturbation
    """
    x_center_og = old_center[0]
    y_center_og = old_center[1]
    preturb_x = np.random.randint(-max_offset, max_offset)
    preturb_y = np.random.randint(-max_offset, max_offset)
    new_center = (x_center_og + preturb_x, y_center_og + preturb_y)
    return new_center
    
    
def gen_fake_data(num_clus=3, np_pts_per_clus=3, display_imgs=False, save_files=False):
    """
    Remade synthetic data generator. Returns a dict of clus nums to arrays in that clus

    Args:
        num_clus (int): Number of clusters to generate. Defaults to 3.
        np_pts_per_clus (int): Number of pts per cluster. Defaults to 3.
        display_imgs (bool): Visualizes each(and every) synth img if true. Defaults to False.
        save_files (bool): Saves all generated synthetic data to my_vars.synthDataPath. 
                            Defaults to False.

    Returns:
        all_data (list of tups of (clus name, data pt)): cluster numbers (str) 
            and synthetic data points. Each synth dp is a numpy array.
    """
    all_data = []
    file_name = 'clus_%d_item_%d'
    for n in range(num_clus):
        clus_data = []
        clus_slope = gen_random_slope()
        clus_center = gen_random_center()
        side = np.random.binomial(n=1,p=0.5)
        for i in range(np_pts_per_clus):
            new_slope = preturb_slope(clus_slope)
            new_center = preturb_center(clus_center)
            my_pt = gen_fake_data_pt(new_slope, new_center, side)
            clus_data.append(my_pt)
            if display_imgs:
                opn.heatMapImg(my_pt)
            
            current_filename = file_name%(n, i)
            if save_files:
                current_filename += ".pickle"
                file_path = Path(my_vars.synthDataPath)
                full_file_path = file_path.joinpath(current_filename)
                with open(full_file_path, 'wb') as handle:
                    pickle.dump(my_pt, handle)
            tup = (current_filename, my_pt)
            all_data.append(tup)
    return all_data

    
if __name__ == "__main__":
    gen_fake_data(display_imgs=True, save_files=True)