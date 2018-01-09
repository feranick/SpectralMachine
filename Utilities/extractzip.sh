#!/bin/bash
#
# ExtractZip - a script to extract Zips into folder
#
# By Nicola Ferralis <feranick@hotmail.com>
#
# This is script is licensed throughthe GNU Public license v.2.0
#

version="20180110a"
if [ "$1" = "" -o "$1" = "-h" ]; then
    echo
    echo " ExtractZip v."$version
    echo " Usage:"
    echo "  extractzip.sh <destination folder>"
    echo
else
    echo " Create folder: "$1
    mkdir $1
    pathfiles=./
    for i in $( ls $pathfiles );
        do
        if [ "${i##*.}" = "zip" ]; then
            unzip $i -d $1
        fi
    echo
    echo " Done!"
    echo
    done
fi

