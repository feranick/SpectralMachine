#!/bin/bash
#
# DownloadExtractZip_raman - a script to extract Zips into folder
#
# By Nicola Ferralis <feranick@hotmail.com>
#
# This is script is licensed throughthe GNU Public license v.2.0
#

version="20241113a"
if [ "$1" = "" -o "$1" = "-h" ]; then
    echo
    echo " ExtractZip v."$version
    echo " Usage:"
    echo "  DownloadExtractZip_raman <destination folder>"
    echo
else
    folder="rruff-raman_"$1
    mkdir $folder
    cd $folder
    wget http://rruff.info/zipped_data_files/raman/LR-Raman.zip
    wget http://rruff.info/zipped_data_files/raman/excellent_oriented.zip
    wget http://rruff.info/zipped_data_files/raman/excellent_unoriented.zip
    wget http://rruff.info/zipped_data_files/raman/fair_oriented.zip
    wget http://rruff.info/zipped_data_files/raman/fair_unoriented.zip
    wget http://rruff.info/zipped_data_files/raman/ignore_unoriented.zip
    wget http://rruff.info/zipped_data_files/raman/poor_oriented.zip
    wget http://rruff.info/zipped_data_files/raman/poor_unoriented.zip
    wget http://rruff.info/zipped_data_files/raman/unrated_oriented.zip
    wget http://rruff.info/zipped_data_files/raman/unrated_unoriented.zip
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

