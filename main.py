import numpy as np
import Mole

def main():
	
  m1 = Mole.Mole('PAT_92_141_551')
  print(np.sum(m1.perim))
  m1.color_regions()
  print("Done!")

if __name__ == "__main__":
	main()
