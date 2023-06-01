import csv
import Mole as Molepy
import os
global counter
counter = 0
def csvwrite(row, file):
   

    # create the csv writer
    writer = csv.writer(file)
    
    # write a row to the csv file
    writer.writerow(row)

    return print("done")

firstrow = ["id", "color", "symmetry", "compactness",'border', "diagnosis", "smoker", "inheritance"]

def snatchData(id, metadata_file):
    try:
        isinstance(listscv, list)
        
    except NameError:

        #if file is not created
        with open(metadata, 'r') as r:
            reader_obj = csv.reader(r)
            listscv = []  
            
            
            for row in reader_obj:
                listscv.append(row)
            print("searching for file..")
            for el in listscv:
                #print(el[24], id)
                if el[24] == id:
                    print("found!")
                    
                    return el[2], el[9], el[17], el[24]
                    
                else:
                    continue
                    
    else:
        #if file is created
        print("searching for file..")
        for el in listscv:
            if el[24] == id:
                this, a, dumb, solution =el[2], el[9], el[17], el[24]
                listscv.remove(el)
                return this, a, dumb, solution
                
            else:
                print("file not found")

    return print("This should not be printed. XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

def featureExtractor(filename, metadata_file):
    global counter
    for el in os.listdir(".\\Images"):
        if "mask_" + el in os.listdir(".\\Masks_png"):

            pic = Molepy.Mole(el[:-4])
            smoker, inheritance, diagnosis, id = snatchData(el,metadata_file)
            counter += 1
            print("writing file", counter)
            color = pic.color_extraction()
            symmetry = pic.symmetry()
            compactness = pic.comp
            border = pic.border

            answlist = [id, color, symmetry, compactness, border, diagnosis, smoker, inheritance]
            
            csvwrite(answlist, filename)
            
        else:
            print("file not found")
    return print("done")

def main(metadata_file):

    with open('dataExtracted.csv', 'a+') as w:
        if w.readline() != firstrow:
            csvwrite(firstrow, w)
            print("file created, writing data..")
            featureExtractor(w, metadata_file)
            print("Data written to file.")
        else:
            
            print("file already exists.")
            print("writing data..")
            featureExtractor(w, metadata_file)
            print("Data written to file.")

   

    return  print("Success!", counter, "files written")
    



"""
if __name__ == "__main__":
    main()
    """