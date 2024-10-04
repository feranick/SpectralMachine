#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***************************************************
* Convert from multiline string into single line
* By: Nicola Ferralis <feranick@hotmail.com>
* version v2024.10.04.1
***************************************************
'''
print(__doc__)

import sys, os.path, getopt

def main():

    saveInText = True
    
    if(len(sys.argv)<3):
        print(' Usage:\n  python3 Multi2SingleLine.py -t \"<multi-line text>\"')
        print('  python3 Multi2SingleLine.py -f <file>\n')
        return
    
    opts, args = getopt.getopt(sys.argv[1:],
        "tfh:", ["text", "file", "help"])
    
    for o, a in opts:
        if o in ("-t" , "--text"):
            string = sys.argv[2]
            print(string.replace('\n', ''),"\n")
    
        if o in ("-f" , "--file"):
            file = sys.argv[2]
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
                    
        if o in ("-h" , "--help"):
            print(' Usage:\n  python3 Multi2SingleLine.py <file>\n')
            return
            
#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
