
#import Mole
#import writer.py
#import classification.py
import writer.py as w

datacsv = "metadata.csv"
molepng = "PAT_21_982_266.png"
molemask = "mask_PAT_21_982_266.png"






def main(data, mask, mole):

    w.main(data)

    return




if __name__ == "__main__":
    main(datacsv, molemask, datacsv)