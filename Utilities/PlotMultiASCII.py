#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************
* PlotMultiASCII
* Plot Multiple ASCII Data
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
    
    if len(sys.argv) < 2:
        print(' Usage:\n')
        print(' (Single File): python3 PlotMultiASCII <folder>\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    
    try:
        for ind, file in enumerate(sorted(os.listdir(sys.argv[1]))):
            if os.path.splitext(file)[-1] == '.txt':
                with open(sys.argv[1]+"/"+file, 'r') as f:
                    M = np.loadtxt(f, skiprows = 0, delimiter = '\t', unpack=False)
                M[:,1] -= np.min(M[:,1])
                M[:,1] = M[:,1]/np.max(M[:,1])
            
                print(" Plotting:", file)
                plt.plot(M[:,0],M[:,1],label=file)
            else:
                pass
    except:
        print('\033[1m ' + str(sys.argv[1]) + ' file not found \n' + '\033[0m')
        return
    
    plt.xlabel('Raman shift [1/cm]')
    plt.ylabel('Raman Intensity [arb. units]')
    #plt.legend()
    #plt.savefig("MultiASCII" + '.png', dpi = 160, format = 'png')  # Save plot
    plt.show()

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
