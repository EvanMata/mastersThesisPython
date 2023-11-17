import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib_variable_names as my_new_vars

from pathlib import Path

def openBaseHolo(holoNumber, pathtype='z', proced=False, mask=False):
    '''
    Opens a raw hologram, processed hologram (proced=True, mask=False)
    or proc'd holo mask (mask=True) and return the raw array of pixel vals
    '''
    if proced or mask:
        if mask:
            if pathtype=='z':
                holoPath = my_new_vars.maskNameZ % holoNumber
            elif pathtype=='f':
                holoPath = my_new_vars.maskNameF % holoNumber
            else:
                holoPath = my_new_vars.maskNameS % holoNumber
        else:
            if pathtype=='z':
                holoPath = my_new_vars.procedHoloNameZ % holoNumber
            elif pathtype=='f':
                holoPath = my_new_vars.procedHoloNameF % holoNumber
            else:
                holoPath = my_new_vars.procedHoloNameS % holoNumber
        dim_1 = 43
        dim_2 = 43
    else:
        if pathtype=='z':
            holoPath = my_new_vars.rawHoloNameZ % holoNumber
        elif pathtype=='f':
            holoPath = my_new_vars.rawHoloNameF % holoNumber
        else:
            holoPath = my_new_vars.rawHoloNameS % holoNumber
        dim_1 = 960
        dim_2 = 972
    with open(holoPath, mode='rb') as file:
        holo = file.read()

    holoArray = np.frombuffer(holo, dtype="float64")
    holoArray.resize((dim_2, dim_1))
    return holoArray


def openTopo(topoNumber, pathtype='z'):
    if pathtype == 'z':
        my_path = my_new_vars.rawTopoNameZ % topoNumber
    if pathtype.lower() == 's':
        my_path = my_new_vars.rawTopoNameS % topoNumber
    if pathtype.lower() == 'f':
        my_path = my_new_vars.rawTopoNameF % topoNumber
    dim_1 = 960
    dim_2 = 972
    with open(my_path, mode='rb') as file:
        topo = file.read()

    topoArray = np.frombuffer(topo, dtype="float64")
    topoArray.resize((dim_2, dim_1))
    return topoArray


def heatMapImg(holoArray):
    plt.imshow(holoArray, cmap='hot', interpolation='none')
    plt.colorbar()
    plt.show()
    return plt


def openAllSynthData(n_clus, n_pts, disply=False):
    clus_dict = dict()
    for clus in range(n_clus):
        clus_dict[clus] = [0]*n_pts
        
    print(clus_dict)
    
    my_dir = my_new_vars.synthDataPath
    all_file_paths = list(Path(my_dir).iterdir())
    for synth_data_path in all_file_paths:
        with open(synth_data_path, 'rb') as handle:
            data_pt = pickle.load(handle)
        data_pt_name = str(synth_data_path.name)
        data_pt_name_parts = data_pt_name.split('_')
        data_pt_clus = int(data_pt_name_parts[1])
        data_pt_item = int(data_pt_name_parts[3].strip('.pickle'))
        clus_list = clus_dict[data_pt_clus]
        clus_list[data_pt_item] = data_pt
        clus_dict[data_pt_clus] = clus_list
        
    if disply:
        for clus, items in clus_dict.items():
            for item in items:
                heatMapImg(item)
        
    return clus_dict


def holoExample(holoNumber = '00001'):
    holoData = openBaseHolo(holoNumber)
    holoViz1 = heatMapImg(holoData)


def openAdjMatrix(adjMatrixFolder, reduce=True, downTo=1000):
    adjMatrixPath = adjMatrixFolder + "\\Correlation_Map.bin"
    with open(adjMatrixPath, mode='rb') as file:
        adjMatrixRawData = file.read()
    i = np.arange(28800*28800, dtype=np.double).reshape(28800, 28800)
    adjMatType = i.dtype
    del i
    adjMat = np.frombuffer(adjMatrixRawData, dtype=adjMatType)
    adjMat.resize((28800, 28800))
    if reduce:
        adjMat = adjMat[:downTo, :downTo]
    print("Correlation Matrix Opened!")
    return adjMat


def parseDataTable(pathtype='z'):
    if pathtype.lower().strip() == 'z':
        dataTablePath = my_new_vars.dataTablePathZ
        skiprows = 9
    elif pathtype.lower().strip() == 'f':
        dataTablePath = my_new_vars.dataTablePathF
        skiprows = 9
    else:
        dataTablePath = my_new_vars.dataTablePathS
        skiprows = 9
    df = pd.read_csv(dataTablePath, delimiter='\t', skiprows=skiprows, index_col=False)
    return df


def grab_mode_items(my_mode=' 1-1', use_helicty=False, helicity=1, and_topos=False, pathtype='f'):
    '''
    Returns a list of file numbers (eg '00001.bin') labeled with the given mode,
    eg mode ' 1-1'. Can additionally return the topography holograms associated w. them

    Inputs:
    --------
        my_mode (str) : name of the mode having its items processed
        use_helicity (bool) : whether to grab all items, or only those with the given helicity
        helicity (1|-1) : the helicity of the items to grab in the given mode.
        and_topos (bool) : If true, also returns the names of the topography holograms associated 
            with the given holograms
        pathtype (f) : Indicates loading the data based of the full table - do Not change.

    Returns:
    --------
        file_nums (lst of strs) : The numbers, w. .bin, of the holograms in the given mode
        *topo_nums (lst of strs) : The numbers of the topography holograms associated w. the 
                holograms
        * -> sometimes returned
    '''
    df = parseDataTable(pathtype)
    if use_helicty and not and_topos:
        df_mode_names = df[(df[' Mode:'] == my_mode) & (df[' Helicitiy:'] == helicity)][' FileName:']
    elif not use_helicty and not and_topos:
        df_mode_names = df[df[' Mode:'] == my_mode][' FileName:']
    elif use_helicty and and_topos:
        df_mode_names = df[(df[' Mode:'] == my_mode) & (df[' Helicitiy:'] == helicity)][' FileName:']
        df_mode_topo_names = df[(df[' Mode:'] == my_mode) & (df[' Helicitiy:'] == helicity)][' Topography-Nr:']
    else:
        df_mode_names = df[df[' Mode:'] == my_mode][' FileName:']
        df_mode_topo_names = df[df[' Mode:'] == my_mode][' Topography-Nr:']
    file_nums = list(df_mode_names)
    file_nums = [f.strip() for f in file_nums]
    if and_topos:
        topo_nums = list(df_mode_topo_names)
        topo_nums = [str(t).zfill(3) for t in topo_nums]
        return file_nums, topo_nums
    return file_nums


def grab_calced_modes():
    """
    Returns 2 lists of numpy arrays of opened pre-generated 
    pos & neg calculated mode pieces.
    """
    dim_1 = 960
    dim_2 = 972
    pos_calced_pieces = []
    neg_calced_pieces = []
    rough_filename = my_new_vars.calcedRawFolder
    mode_nums = list(range(1,72+1))
    mode_nums = [str(m).zfill(2) for m in mode_nums]
    
    for mode_num in mode_nums:
        mode_name = "Mode_" + mode_num
        folder_path = rough_filename.joinpath(mode_name)

        pos_path_name = "Pos_Holo_Calculated_Mode_%s.bin"%mode_num
        pos_path = folder_path.joinpath(pos_path_name)
        pos_path = str(pos_path)

        neg_path_name = "Neg_Holo_Calculated_Mode_%s.bin"%mode_num
        neg_path = folder_path.joinpath(neg_path_name)
        neg_path = str(neg_path)

        with open(pos_path, mode='rb') as file:
            holo = file.read()
            holoArray = np.frombuffer(holo, dtype="float64")
            holoArray.resize((dim_2, dim_1))
            pos_calced_pieces.append(holoArray)

        with open(neg_path, mode='rb') as file:
            holo = file.read()
            holoArray = np.frombuffer(holo, dtype="float64")
            holoArray.resize((dim_2, dim_1))
            neg_calced_pieces.append(holoArray)

    return pos_calced_pieces, neg_calced_pieces


def yield_mode_pieces():
    '''
    YIELD FUNCS DO NOT RUN/COMPILE UNTIL USED
    '''
    folder = Path(my_new_vars.modesPathZ)
    path, dirs, files = next(os.walk(folder))
    file_count = len(dirs)
    
    list_modes = list(np.arange(1,file_count+1))

    for mode in list_modes:
        filename = folder.joinpath("Mode_%02d"%mode)
        filename = filename.joinpath("Pos_Holo_Original_Mode_%02d.bin"%mode)
        mode_pos_holo = np.fromfile(str(filename))
        mode_pos_holo.resize((972, 960))
        yield mode_pos_holo


def open_and_combine_pieces(pieces, avg=True):
    """
    Returns the avg (or sum) of the holograms labeled with the given mode.

    Inputs:
    --------
        pieces (lst of strs) : The numbers, w. .bin, of the holograms in the given mode
        avg (bool) : If true, returns the avg of the holos in the mode, 
                     If false returns their sum

    Returns: 
    --------
        out_arr (np array) : Array of the ~mode
    """
    num_holos = 0
    base_arr = np.zeros((972, 960))
    raw_path = my_new_vars.rawHoloNameF
    for holo_name in pieces:
        holoNumber = holo_name.strip(".bin")
        holo_arr = openBaseHolo(holoNumber, pathtype='f', proced=False, mask=False)
        base_arr += holo_arr
        num_holos += 1

    if avg:
        out_arr = base_arr / num_holos
    else:
        out_arr = base_arr
    return out_arr


def pre_gen_d_open_and_combine_pieces(my_mode, helicity, avg=False):
    """
    Returns the avg (or sum) of the holograms labeled with the given mode 
    in the pre-generated data. Basically run open_and_combine_pieces on 
    provided data

    Inputs:
    --------
        my_mode (str) : Name of the mode label in Claudio's data
        helicity (1|-1) : The helicity of the images to be loaded
        avg (bool) : If true, returns the avg of the holos in the mode, 
                        if false returns their sum

    Returns: 
    --------
        out_arr (np array) : Array of the ~mode
    """
    pieces = grab_mode_items(my_mode=my_mode, use_helicty=True, helicity=helicity, and_topos=False)
    return open_and_combine_pieces(pieces, avg)


if __name__ == "__main__":
    #openAllSynthData(n_clus=3, n_pts=3, disply=True)
    df = parseDataTable()
    #print(len(set(df[' Mode:']))) #There are 73 different modes, including " Not assigned"
    print(grab_mode_items(use_helicty=True))