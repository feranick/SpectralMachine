#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************
* PlotASCIIData
* Plot single ASCII Data
* File must be in ASCII
* version: v2023.12.15.1
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************
'''
print(__doc__)

import numpy as np
import sys, os.path, getopt, glob, csv
import matplotlib.pyplot as plt

#************************************
# Main
#************************************
def main():
    
    if len(sys.argv) < 1:
        print(' Usage:\n')
        print(' (Single File): python3 PlotASCIIData <filename>\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    
    try:
        with open(sys.argv[1], 'r') as f:
            M = np.loadtxt(f, skiprows = 0, delimiter = '\t', unpack=False)
        print(" Plotting:", sys.argv[1])
        plot(M[:,0],M[:,1], sys.argv[1])
    
    except:
        print('\033[1m ' + str(sys.argv[1]) + ' file not found \n' + '\033[0m')
        return
    
def plot(x,y, title):
    plt.plot(x,y,'ro')
    plt.plot(x,y, 'b')
    plt.title(title)
    plt.show()
    

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
