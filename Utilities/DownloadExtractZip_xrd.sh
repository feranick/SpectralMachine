#!/bin/bash
#
# DownloadExtractZip_xrd - a script to extract Zips into folder
#
# By Nicola Ferralis <feranick@hotmail.com>
#
# This is script is licensed throughthe GNU Public license v.2.0
#

version="20250223a"
if [ "$1" = "" -o "$1" = "-h" ]; then
    echo
    echo " ExtractZip v."$version
    echo " Usage:"
    echo "  DownloadExtractZip_xrd.sh <destination folder>"
    echo
else
    folder="rruff-powder_"$1
    mkdir $folder
    cd $folder
    wget https://rruff.info/zipped_data_files/powder/XY_Processed.zip
    wget https://rruff.info/zipped_data_files/powder/XY_RAW.zip
    wget https://rruff.info/zipped_data_files/powder/DIF.zip
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

