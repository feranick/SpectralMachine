#!/usr/bin/env python3
'''
***********************************************
* SaveExcelCSV
* Adds data from single file to Master Doc
* version: v2023.12.15.1
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************
'''
print(__doc__)
import pandas as pd
import sys, os.path, h5py

#************************************
# Parameters definition
#************************************
class dP:
    saveAsCsv = True
    saveAsHDF = False

#************************************
# Main
#************************************
def main():
    if len(sys.argv) < 2:
        print(' Usage:\n  python3 SaveExcelCSV.py <Excel File>')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    
    rootFile = os.path.splitext(sys.argv[1])[0]
    os.mkdir(rootFile)
    root = "./"+rootFile+"/"+rootFile
    print(" Opening Excel File:",sys.argv[1],"...\n")
    ws = pd.read_excel(sys.argv[1], sheet_name=None)

    for name, key in ws.items():
        #print(name)
        #print(ws[name])
        if dP.saveAsCsv:
            CSVfile = root+'_'+name+".csv"
            print(" Sheet:",name,"saved in:",CSVfile)
            ws[name].to_csv(CSVfile)
        if dP.saveAsHDF:
            HDFfile = root+'_'+name+".h5"
            print(" Sheet:",name,"saved in:",HDFfile)
            ws[name].to_hdf(HDFfile, key='df', mode='w')
    print("\n")

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
