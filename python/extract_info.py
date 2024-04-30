import math
import numpy as np

file = open("sign2n_EXFOR_with_mass_B.dat", 'r')

Z = 0
A = 0
info = []
for line in file:
    curLine = line.strip().split()
    if curLine[0] == Z and curLine[1]==A:
        continue
    Z = curLine[0]
    A = curLine[1]
    info.append(np.asfarray(curLine[0:2]+curLine[3:5]+curLine[8:9]+curLine[10:11]))
info = np.array(info)
np.savetxt("nulide_info_1.dat",info, fmt="%f", delimiter="  ")
