import pathlib
from pathlib import Path

small_sample_paths = ["Provided", "SMALL_SAMPLE"]
full_sample_paths = ["Provided", "FULL_SAMPLE"]
python_paths = ["Created", "mastersThesisPython"]

p = pathlib.Path(__file__)
print()
print("Currently Accessed Variables File: ")
print(p)
print()

core = list( p.parts[:-3] )
corePath = pathlib.Path(*core)

smallSampPath = core + small_sample_paths
smallSampPath = pathlib.Path(*smallSampPath)

fullSampPath = core + full_sample_paths
fullSampPath = pathlib.Path(*fullSampPath)

pythonPath = core + python_paths
pythonPath = pathlib.Path(*pythonPath)

'''
Paths for Python Stuff, such as Synthetic Data
'''

synthDataPath = str(pythonPath.joinpath(   "syntheticData" ) )


'''
Paths for the SMALL sample folders
'''

centroidsFolderS = str(smallSampPath.joinpath(   "Cluster_Centroids" ) )
holosFolderS = str(smallSampPath.joinpath(       "Raw_Holograms" ) )
adjMatrixFolderS = str(smallSampPath.joinpath(   "Clustering_Analysis" ) )
procedHolosFolderS = str(smallSampPath.joinpath( "Fully_Processed_Holograms" ) )
masksFolderS = str(smallSampPath.joinpath(       "Hologram_Masks") )

dataTablePathS = str(smallSampPath.joinpath(     "Log_Data_Coherent_Correlation_Imaging.txt" ) )

rawHoloNameS = str(smallSampPath.joinpath(    "Raw_Hologram_%s.bin" ) )
procedHoloNameS = str(smallSampPath.joinpath( "preprocessed_Holo_%s.bin" ) )
maskNameS = str(smallSampPath.joinpath(       "holo_mask_%s.bin" ) )


'''
Paths for the FULL sample folders
'''

centroidsFolderF = str(fullSampPath.joinpath(   "Cluster_Centroids" ) )
holosFolderF = str(fullSampPath.joinpath(       "Raw_Holograms" ) )
adjMatrixFolderF = str(fullSampPath.joinpath(   "Clustering_Analysis" ) )
procedHolosFolderF = str(fullSampPath.joinpath( "Fully_Processed_Holograms" ) )
masksFolderF = str(fullSampPath.joinpath(       "Hologram_Masks") )

dataTablePathF = str(fullSampPath.joinpath(     "Log_Data_Coherent_Correlation_Imaging.txt" ) )

rawHoloNameF = str(fullSampPath.joinpath(    "Raw_Hologram_%s.bin" ) )
procedHoloNameF = str(fullSampPath.joinpath( "preprocessed_Holo_%s.bin" ) )
maskNameF = str(fullSampPath.joinpath(       "holo_mask_%s.bin" ) )
