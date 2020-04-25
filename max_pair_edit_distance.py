#Code to find pair of strains with the maximum edit distance among a list of strains.

#For GPU related questions refer to https://www.geeksforgeeks.org/running-python-script-on-gpu/

import glob
import numba
from numba import jit, cuda 
import codecs
import numpy as np
def readGenome(filename):
    genome = ''
    with codecs.open(filename, 'r') as f:
        for line in f:
            # ignore header line with genome information
            if not line[0] == '>':
                genome += line.rstrip()
    return genome

file_list = glob.glob('*.txt')
lines = []
for file_path in file_list:
    lines.append((readGenome(file_path)))
lines=np.array(lines)    

#Gives a way to measure how a sequenced strain compares to reference strain. Worth repeating over many identified strains.
@jit(target ="cuda")
def editDistance(x, y):
    # Create distance matrix
    D = []
    for i in range(len(x)+1):
        D.append([0]*(len(y)+1))
    # Initialize first row and column of matrix
    for i in range(len(x)+1):
        D[i][0] = i
    for i in range(len(y)+1):
        D[0][i] = i
    # Fill in the rest of the matrix
    for i in range(1, len(x)+1):
        for j in range(1, len(y)+1):
            distHor = D[i][j-1] + 1
            distVer = D[i-1][j] + 1
            if x[i-1] == y[j-1]:
                distDiag = D[i-1][j-1]
            else:
                distDiag = D[i-1][j-1] + 1
            D[i][j] = min(distHor, distVer, distDiag)
    # Edit distance is the value in the bottom right corner of the matrix
    return D[-1][-1]

ed=[]
indices=[]


for i in range(0,len(lines)):
    for j in range(i+1,len(lines)):
        ed.append(editDistance(lines[i],lines[j]))
        indices.append([i,j])

max_edit_distance = max(ed)
max_index=ed.index(max_edit_distance)

print("The pair with the maximum edit distance "+indices[max_index])



