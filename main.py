import numpy as np
import Mole
import numpy as np

def main():
    mole = Mole.Mole('PAT_91_140_664')

  #  mole.show_per()
    

    print(f"Perimeter Sum: {np.sum(mole.perim)}")
    
   # mole.symmetric()
    print(mole.symmetry_detection())
   # mole.show_seg_mask()

    print("Done!")

if __name__ == "__main__":
    main()