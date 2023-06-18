
#import Mole
##import writer.py
#import classification.py
import writer as w
import classification as c


def main():
    datacsv = "metadata.csv" #Path to metadata.csv
    dataExtracted = "dataExtracted.csv" #Path to dataExtracted.csv

    #Make sure your folder paths are the same
    molepngFolder = "\\Images"
    molemaskFolder = "\\Masks_png"

    Trained = True #Set to True if you want to train the model
    MetaDataWritten = True #Set to false if you want to extract metadata from dataset

    if MetaDataWritten == True:
        if Trained == False:
            results =c.test_melanomas(dataExtracted, Trained)
        else:
            results, mole_evaluation =c.test_melanomas(dataExtracted, Trained)
    if MetaDataWritten == False:
        w.main(datacsv)
        if Trained == False:
            results =c.test_melanomas(dataExtracted, Trained)
        else:
            results, mole_evaluation =c.test_melanomas(dataExtracted, Trained)
    
    if Trained == True:
        for key, val in mole_evaluation.items():
            print("Classifier ", key," evaluation of given dataset:")
            print(val)
            print("\n")

    for key, val in results.items():
        print(val)
        print("\n")
    
    return print("Finished!")




if __name__ == "__main__":
    main()