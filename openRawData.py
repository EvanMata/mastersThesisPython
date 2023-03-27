import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import variables_names as my_vars


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
    #holoExample('01901')
    #adjMatR = openAdjMatrix(adjMatrixFolder, downTo=2000)
    #heatMapImg(adjMatR) #WILL CRASH if Not using DownTo, 30k x 30k pixels ~ 100Bil Pixels to image.
    #print(adjMatR[0])
    h = openBaseHolo(holoNumber='00001', proced=True)
    #heatMapImg(h)
    #parseDataTable()

    mask_path = "C:\\Users\\evana\\Documents\\Thesis\\Provided\\SMALL_SAMPLE\\Hologram_Masks"
    maskFullPath = mask_path + '\\holo_mask_%s.bin' % "00001"
    print(maskFullPath)
    dim_1 = 43
    dim_2 = 43
    with open(maskFullPath, mode='rb') as file:
        holo = file.read()

    holoArray = np.frombuffer(holo, dtype="float64")
    holoArray.resize((dim_2, dim_1))

    '''
    mask_path = "C:\\Users\\evana\\Documents\\Thesis\\Provided\\SMALL_SAMPLE\\Hologram_Masks"
    maskFullPath = mask_path + '\\holo_mask_%s.bin' % "00010"
    print(maskFullPath)
    dim_1 = 43
    dim_2 = 43
    with open(maskFullPath, mode='rb') as file:
        holo = file.read()

    #holoArray2 = np.frombuffer(holo, dtype="float64")
    #holoArray2.resize((dim_2, dim_1))
    #print(m)
    '''
    heatMapImg(holoArray)