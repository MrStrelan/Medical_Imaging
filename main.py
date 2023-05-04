import numpy as np
import Mole
import numpy as np
import os
import json

def main():


    return

def symmetry_to_json():
    symValues = {}
    for pic in os.listdir(".\\Medical_Imaging\\Images"):
        mole = Mole.Mole(pic[:-4])

    #  mole.show_per()
        

        print(f"Perimeter Sum: {np.sum(mole.perim)}")
        
    # mole.symmetric()
        values = mole.symmetry_detection()
        symValues.update({pic: values})
   # mole.show_seg_mask()
        json_object = json.dumps(symValues, indent=4)
        with open("symValues.json", "w") as outfile:
            outfile.write(json_object)
        print(pic, "done")
    
    return  print("Done!")




if __name__ == "__main__":
    main()