
#import Mole
#import writer.py
#import classification.py
import writer as w

datacsv = "metadata.csv" #Path to metadata.csv

#Make sure your folder paths are the same
molepngFolder = ".\\Images"
molemaskFolder = ".\\Masks_png"







def main(data):

    w.main(data)

    return




if __name__ == "__main__":
    main(datacsv)