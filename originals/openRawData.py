import pickle
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib_variable_names as my_vars

from pathlib import Path
from os import listdir
from os.path import isfile, join


def openBaseHolo(holoNumber, f=False, proced=False, mask=False):
    '''
    Opens a raw hologram, processed hologram (proced=True, mask=False)
    or proc'd holo mask (mask=True) and return the raw array of pixel vals
    '''
    if proced or mask:
        if mask:
            if f:
                holoPath = my_vars.maskNameF % holoNumber
            else:
                holoPath = my_vars.maskNameS % holoNumber
        else:
            if f:
                holoPath = my_vars.procedHoloNameF % holoNumber
            else:
                holoPath = my_vars.procedHoloNameS % holoNumber
        dim_1 = 43
        dim_2 = 43
    else:
        if f:
            holoPath = my_vars.rawHoloNameF % holoNumber
        else:
            holoPath = my_vars.rawHoloNameS % holoNumber
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

def holoExample(holoNumber = '00001'):
    holoData = openBaseHolo(holoNumber)
    holoViz1 = heatMapImg(holoData)

def openAdjMatrix(f=False, reduce=True, downTo=1000):
    if f:
        adjMatrixFolder = my_vars.adjMatrixFolderF
    else:
        adjMatrixFolder = my_vars.adjMatrixFolderS
    adjMatrixPath = pathlib.Path( adjMatrixFolder ).joinpath("Correlation_Map.bin")
    adjMatrixPath = str( adjMatrixPath )
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
        dataTablePath = my_vars.dataTablePathZ
    elif pathtype.lower().strip() == 'f':
        dataTablePath = my_vars.dataTablePathF
    else:
        dataTablePath = my_vars.dataTablePathS
    df = pd.read_csv(dataTablePath, delimiter='\t', skiprows=9, index_col=False)
    return df

def openCentroid(num, wasser, f):
    if wasser:
        nameCore = "barycenters_centroid_%d"
    else:
        nameCore = "correlation_centroid_%d"

    if f:
        clusterCentroidPath = pathlib.Path(my_vars.centroidsFolderF)
    else:
        clusterCentroidPath = pathlib.Path(my_vars.centroidsFolderS)

    clusterCentroidPath = str(clusterCentroidPath.joinpath(nameCore))%num + ".pkl"
    with open(clusterCentroidPath, 'rb') as myf:
        clusterCentroid = pickle.load(myf)

    return clusterCentroid

def openSyntheticData(data_dir=my_vars.generatedDataPath):
    """
    Opens synthetic data, returning list of tups of [(clus_name, clus_array),(),...()]
    clus_name is formatted as:
        "cluster_%d_item_%d.pickle"
    """
    my_imgs = []
    files = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
    for clus_name in files:
        filepath = data_dir.joinpath(clus_name)
        with open(filepath, "rb") as my_filename:
            arr = pickle.load(my_filename)

        my_imgs.append((clus_name, arr))
    return my_imgs

if __name__ == "__main__":
    openSyntheticData()