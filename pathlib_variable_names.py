import pathlib
from pathlib import Path

image_paths = ["Images"]
zenodo_paths = ["zenodoFiles"]
small_sample_paths = ["SMALL_SAMPLE"]
full_sample_paths = ["Provided", "FULL_SAMPLE"]
synth_data_paths = ["Thesis", "mastersThesisPython", "syntheticData"]

p = pathlib.Path(__file__)
print()
print("Currently Accessed Variables File: ")
print(p)
print()

core = list( p.parts[:-3] )
corePath = pathlib.Path(*core)

core2 = list( p.parts[:-2] )
corePath2 = pathlib.Path(*core2)

imagesPath = core2 + image_paths
imagesPath = pathlib.Path(*imagesPath)

smallSampPath = core2 + small_sample_paths
smallSampPath = pathlib.Path(*smallSampPath)

fullSampPath = core + full_sample_paths
fullSampPath = pathlib.Path(*fullSampPath)

generatedDataPath = core + synth_data_paths
generatedDataPath = pathlib.Path(*generatedDataPath)

zenodoPath = core2 + zenodo_paths
zenodoPath = pathlib.Path(*zenodoPath)

'''
Paths for Visuals
'''
stateToDestF = imagesPath.joinpath( "stateToDest" )
stateToDestP = str(stateToDestF.joinpath( "state_moving_frame_%d" ))
vidsFolder = imagesPath.joinpath( "videos" )
stateImgsF = imagesPath.joinpath( "stateExpectations" )
stateImgsP = str(stateImgsF.joinpath( "state_%d_expectation" ))
orbsToStateF = imagesPath.joinpath( "orbsToState" )
orbsToStateP = str(orbsToStateF.joinpath( "state_moving_frame_%d" ))
rawArraysF = imagesPath.joinpath( "orbArrays" )
rawArraysP = str(rawArraysF.joinpath( "orb_array_frame_%d" ))



'''
Paths for the SMALL sample folders
'''

centroidsFolderS = str(smallSampPath.joinpath(   "Cluster_Centroids" ) )
adjMatrixFolderS = str(smallSampPath.joinpath(   "Clustering_Analysis" ) )
dataTablePathS = str(smallSampPath.joinpath(     "Log_Data_Coherent_Correlation_Imaging.txt" ) )

procedHolosFolderS = smallSampPath.joinpath( "Fully_Processed_Holograms" )
procedHoloNameS = procedHolosFolderS.joinpath(( "preprocessed_Holo_%s.bin" ))
procedHolosFolderS = str( procedHolosFolderS )
procedHoloNameS = str( procedHoloNameS )

masksFolderS = smallSampPath.joinpath("Hologram_Masks")
maskNameS = masksFolderS.joinpath("holo_mask_%s.bin")
masksFolderS = str( masksFolderS )
maskNameS = str( maskNameS )

holosFolderS = smallSampPath.joinpath("Raw_Holograms")
rawHoloNameS = holosFolderS.joinpath("Raw_Hologram_%s.bin")
holosFolderS = str(holosFolderS)
rawHoloNameS = str(rawHoloNameS)

toposFolderS = smallSampPath.joinpath("Topography_Holograms")
rawTopoNameS = toposFolderS.joinpath("Topography_Hologram_%s.bin")
toposFolderS = str(toposFolderS)
rawTopoNameS = str(rawTopoNameS)

'''
Paths for the FULL sample folders
'''

centroidsFolderF = str(fullSampPath.joinpath(   "Cluster_Centroids" ) )
adjMatrixFolderF = str(fullSampPath.joinpath(   "Clustering_Analysis" ) )
dataTablePathF = str(fullSampPath.joinpath(     "Log_Data_Coherent_Correlation_Imaging.txt" ) )

procedHolosFolderF = fullSampPath.joinpath( "Fully_Processed_Holograms" )
procedHoloNameF = procedHolosFolderF.joinpath(( "preprocessed_Holo_%s.bin" ))
procedHolosFolderF = str( procedHolosFolderF )
procedHoloNameF = str( procedHoloNameF )

masksFolderF = fullSampPath.joinpath("Hologram_Masks")
maskNameF = masksFolderF.joinpath("holo_mask_%s.bin")
masksFolderF = str( masksFolderF )
maskNameF = str( maskNameF )

holosFolderF = fullSampPath.joinpath("Raw_Holograms")
rawHoloNameF = holosFolderF.joinpath("Raw_Hologram_%s.bin")
holosFolderF = str(holosFolderF)
rawHoloNameF = str(rawHoloNameF)

toposFolderF = fullSampPath.joinpath("Topography_Holograms")
rawTopoNameF = toposFolderF.joinpath("Topography_Hologram_%s.bin")
toposFolderF = str(toposFolderF)
rawTopoNameF = str(rawTopoNameF)

calcedRawFolder = fullSampPath.joinpath("Coherent_Diffraction_Imaging", "Input_Holograms")

'''
Paths for Zenodo Files
'''

centroidsFolderZ = str(zenodoPath.joinpath(   "Cluster_Centroids" ) )
adjMatrixFolderZ = str(zenodoPath.joinpath(   "Clustering_Analysis" ) )
dataTablePathZ = str(zenodoPath.joinpath(     "Log_Data_Coherent_Correlation_Imaging.txt" ) )
modesPathZ = zenodoPath.joinpath("Coherent_Diffraction_Imaging")
modesPathZ = str(modesPathZ.joinpath("Input_Holograms"))

procedHolosFolderZ = zenodoPath.joinpath( "Fully_Processed_Holograms" )
procedHoloNameZ = procedHolosFolderZ.joinpath(( "preprocessed_Holo_%s.bin" ))
procedHolosFolderZ = str( procedHolosFolderZ )
procedHoloNameZ = str( procedHoloNameZ )

masksFolderZ = zenodoPath.joinpath("Hologram_Masks")
maskNameZ = masksFolderZ.joinpath("holo_mask_%s.bin")
masksFolderZ = str( masksFolderZ )
maskNameZ = str( maskNameZ )

holosFolderZ = zenodoPath.joinpath("Raw_Holograms")
rawHoloNameZ = holosFolderZ.joinpath("Raw_Hologram_%s.bin")
holosFolderZ = str(holosFolderZ)
rawHoloNameZ = str(rawHoloNameZ)

toposFolderZ = zenodoPath.joinpath("Topography_Holograms")
rawTopoNameZ = toposFolderZ.joinpath("Topography_Hologram_%s.bin")
toposFolderZ = str(toposFolderZ)
rawTopoNameZ = str(rawTopoNameZ)


"""
Unique Paths
"""

modeMasks = fullSampPath.joinpath("Coherent_Diffraction_Imaging", "processed", 'masks')