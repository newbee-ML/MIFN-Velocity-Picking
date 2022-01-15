import numpy as np
import sys
import re

"""
Generate npy label file from the .dat file which is generate by GeoEast
"""

def LabTxt2Npy(txtPath, npyPath):
    LabDict = {}
    with open(txtPath, 'r', encoding='utf8') as f:
        rows = f.readlines()
        for row in rows[1:]:
            row = [int(i) for i in re.findall("\d+.?\d*", row)]
            LabDict.setdefault(row[0], {})
            LabDict[row[0]].setdefault(row[1], [])
            LabDict[row[0]][row[1]].append(row[2:])

    for line in LabDict.keys():
        for cdp in LabDict[line].keys():
            LabDict[line][cdp] = np.array(LabDict[line][cdp])
    np.save(npyPath, LabDict)

if __name__ == '__main__':
    Dat_path = sys.argv[1]
    Npy_path = sys.argv[2]
    LabTxt2Npy(Dat_path, Npy_path)