import csv
import Mole as Molepy
import os
def csvwrite(row, file):
   

    # create the csv writer
    writer = csv.writer(file)
    
    # write a row to the csv file
    writer.writerow(row)

    return print("done")

firstrow = ["id", "color", "symmetry", "compactness", "diagnosis", "smoker", "inheritance"]

def snatchData(id):
    if isinstance(listscv, list):
        with open('metadata.csv', 'r') as r:
            reader_obj = csv.reader(r)
            listscv = []  
            diagnoses = []
            bodyplaces = []
            # Iterate over each row in the csv file 
            # using reader object
            for row in reader_obj:
                listscv.append(row)
            for el in listscv:
                if el[1] == id[-4]:
                    print("found!")
                    return el[2], el[9], el[17], el[24]
                else:
                    print("searching for file..")
                    
    else:
            for el in listscv:
                if el[1] == id[:-4]:
                    return el[2], el[9], el[17], el[24]
                
                else:
                    print("file not found")

    return print("This should not be printed. XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

def featureExtractor(filename):
    for el in os.listdir(".\\Images"):
        if "mask_" + el in os.listdir(".\\Masks_png"):
            pic = Molepy.Mole(el[:-4])
            """
            pic.color()
            pic.borders()
            pic.symmetry()
            pic.compactness()
            """
            smoker, inheritance, diagnosis, id = snatchData(el)
            answlist = [[id, pic.color(), pic.symmetry(), pic.compactness(), diagnosis, smoker, inheritance]]
            
            csvwrite(answlist, filename)
            print("done")
        else:
            print("file not found")
    pic = Molepy.Mole()

def main():
    with open('dataExtracted.csv', 'a+') as w:
        if w.readline() != firstrow:
            csvwrite(firstrow, w)
            print("file created, writing data..")
            featureExtractor(w)
            print("Data written to file.")
        else:
            
            print("file already exists.")
            print("writing data..")
            featureExtractor(w)
            print("Data written to file.")

   

    return  print("Success!")
    




if __name__ == "__main__":
    main()