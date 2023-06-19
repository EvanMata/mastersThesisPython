'''
Generates Synthetic Data for trivial clustering and metric examination
'''

import pickle

import numpy as np

import openRawData as opn
import pathlib_variable_names as my_vars

from pathlib import Path

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
    """_summary_

    Args:
        slope (_type_): _description_
        center (_type_): _description_
        side (int): 0|1 basically which side should be colored white and which black
        x_max (int, optional): _description_. Defaults to 100.
        y_max (int, optional): _description_. Defaults to 100.

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
    x_center_og = old_center[0]
    y_center_og = old_center[1]
    preturb_x = np.random.randint(-max_offset, max_offset)
    preturb_y = np.random.randint(-max_offset, max_offset)
    return x_center_og + preturb_x, y_center_og + preturb_y
    
    
def gen_fake_data(num_clus=3, np_pts_per_clus=3, display_imgs=False, save_files=False):
    all_data = dict()
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
            if save_files:
                current_filename = file_name%(n, i)
                current_filename += ".pickle"
                file_path = Path(my_vars.synthDataPath)
                full_file_path = file_path.joinpath(current_filename)
                with open(full_file_path, 'wb') as handle:
                    pickle.dump(my_pt, handle)
        all_data[n] = clus_data
    return all_data

            
            
        
    
if __name__ == "__main__":
    gen_fake_data(display_imgs=True, save_files=True)