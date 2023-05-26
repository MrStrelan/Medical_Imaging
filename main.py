import numpy as np
import Mole

def main():

    mole = Mole.Mole("PAT_10_18_830")
    #a,b,v,g,d,e,j = mole.find_colors
    #print("count", a, " a ", b, " b ", v, " v ", g, " d ", d, " e ", e, " j ", j)
    mole.print_all()
    return

main()