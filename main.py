import numpy as np
import Mole
import numpy as np

def main():
    mole = Mole.Mole('PAT_92_141_551')
    mole.show_per()
    print(f"Convex Perimeter: {mole.conv}")
    print(f"Perimeter Sum: {np.sum(mole.perim)}")
    symmetry = mole.symmetry_detection()
    print(f"Symmetry Detected: {symmetry}")
    print("Done!")

if __name__ == "__main__":
    main()