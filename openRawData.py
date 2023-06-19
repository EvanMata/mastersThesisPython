import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import variables_names as my_vars
import pathlib_variable_names as my_new_vars

from pathlib import Path

def openBaseHolo(holoNumber, proced=False, mask=False):
    '''
    Opens a raw hologram, processed hologram (proced=True, mask=False)
    or proc'd holo mask (mask=True) and return the raw array of pixel vals
    '''
    if proced or mask:
        if mask:
            holoPath = my_vars.masksFolder + \
                        my_vars.maskName % holoNumber
        else:
            holoPath = my_vars.procedHolosFolder + \
                        my_vars.procedHoloName % holoNumber
        dim_1 = 43
        dim_2 = 43
    else:
        holoPath = my_vars.holosFolder + my_vars.rawHoloName % holoNumber
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

def openAdjMatrix(adjMatrixFolder=my_vars.adjMatrixFolder, reduce=True, downTo=1000):
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

def parseDataTable(dataTablePath=my_vars.dataTablePath):
    df = pd.read_csv(dataTablePath, delimiter='\t', skiprows=9, index_col=False)
    return df



if __name__ == "__main__":
    openAllSynthData(n_clus=3, n_pts=3, disply=True)