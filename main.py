
#import Mole
##import writer.py
#import classification.py
import writer as w
import classification as c

datacsv = "metadata.csv" #Path to metadata.csv
dataExtracted = "dataExtracted.csv" #Path to dataExtracted.csv

#Make sure your folder paths are the same
molepngFolder = "\\Images"
molemaskFolder = "\\Masks_png"

Trained = False #Set to True if you want to train the model
MetaDataWritten = True #Set to false if you want to extract metadata from dataset






def main(data, Trained, MetaDataWritten, datacsv):

    if MetaDataWritten == True:
        results =c.test_melanomas(data, Trained)
    if MetaDataWritten == False:
        w.main(datacsv)
        results =c.test_melanomas(data, Trained)
    
    for key, val in results.items():
        print(key, " : ", val)
        print("\n")
    
    return print("Finished!")




if __name__ == "__main__":
    main(dataExtracted, Trained, MetaDataWritten, datacsv)