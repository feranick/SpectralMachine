#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***************************************************
* Convert from multiline string into single line
* By: Nicola Ferralis <feranick@hotmail.com>
* version v2023.12.15.1
***************************************************
'''
print(__doc__)

import sys, os.path

def main():

    saveInText = True
    
    if(len(sys.argv)<2):
        print(' Usage:\n  python3 Multi2SingleLine.py <file>\n')
        return
    
    file = sys.argv[1]
    fileRoot = os.path.splitext(file)[0]
    outfile = fileRoot+"_single.txt"
    
    with open(file, 'r') as f:
        print(" Opening text file with multiple lines:",file,"\n")
        data = f.read().replace('\n', '')
        print(" Single line string: \n")
        print(data,"\n")
           
    if saveInText:
        with open(outfile, "w") as of:
            of.write(data)
            print(" Single line text file saved in:",outfile,"\n")

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
