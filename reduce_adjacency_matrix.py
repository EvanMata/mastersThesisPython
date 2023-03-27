matrixPath = "C:\\Users\\evana\\Documents\\Thesis\\Provided\\SMALL_SAMPLE\\Clustering_Analysis"
matrixFullPath = matrixPath + "\\Correlation_Map.bin"
with open(matrixFullPath, mode='rb') as file:
    matrixContent = file.read()

print(type(matrixContent.from_bytes()))