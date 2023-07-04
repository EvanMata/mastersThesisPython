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
    elif pathtype.lower().strip() == 'f':
        dataTablePath = my_new_vars.dataTablePathF
    else:
        dataTablePath = my_new_vars.dataTablePathS
    df = pd.read_csv(dataTablePath, delimiter='\t', skiprows=9, index_col=False)
    return df


def grab_mode_items(my_mode=' 1-1', use_helicty=False, helicity=1):
    '''
    Returns a list of file numbers (eg '00001.bin') in mode the given mode,
    eg mode ' 1-1'.
    '''
    df = parseDataTable(pathtype='s')
    if use_helicty:
        df_mode_1 = df[(df[' Mode:'] == my_mode) & (df[' Helicitiy:'] == helicity)][' FileName:']
    else:
        df_mode_1 = df[df[' Mode:'] == my_mode][' FileName:']
    file_nums = list(df_mode_1)
    file_nums = [f.strip() for f in file_nums]
    return file_nums


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

if __name__ == "__main__":
    #openAllSynthData(n_clus=3, n_pts=3, disply=True)
    df = parseDataTable()
    print(df[' Helicitiy:'])
    #print(len(set(df[' Mode:']))) #There are 73 different modes, including " Not assigned"
    print(grab_mode_items(use_helicty=True))