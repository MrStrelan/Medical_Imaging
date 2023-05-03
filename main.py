import Mole
import numpy as np

def main():
    mole = Mole.Mole('PAT_91_140_664')
    mole.show_per()
    mole.symmetry_detection()
    print(f"Perimeter Sum: {np.sum(mole.perim)}")
    
 #   symmetry = mole.symmetry_detection()
  #  print(f"Symmetry Detected: {symmetry}")
    print("Done!")

if __name__ == "__main__":
    main()