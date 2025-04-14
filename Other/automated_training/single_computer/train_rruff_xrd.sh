#!/bin/bash
#
# TrainLatestRRuff
#
# By Nicola Ferralis <feranick@hotmail.com>
#
# This is script is licensed throughthe GNU Public license v.2.0
#

#base="/Users/feranick/Desktop/"
base=""

version=$1
masterFile="AAA-powder-"$version
enInit1=5
enFin1=90
enStep1=0.02
hfsel1=10
val1=22

folder="rruff-xrd_"$1
mkdir $folder
cd $folder
    wget https://rruff.info/zipped_data_files/powder/DIF.zip
    wget https://rruff.info/zipped_data_files/powder/Reference_PDF.zip
    wget https://rruff.info/zipped_data_files/powder/Refinement_Data.zip
    wget https://rruff.info/zipped_data_files/powder/Refinement_Output_Data.zip
    wget https://rruff.info/zipped_data_files/powder/XY_Processed.zip
    https://rruff.info/zipped_data_files/powder/XY_RAW.zip
echo " Create folder: "$1
mkdir $1
pathfiles=./
for i in $( ls $pathfiles );
   do
   if [ "${i##*.}" = "zip" ]; then
       unzip $i -d $1
   fi
   done

cd $1
"${base}RruffDataMaker.py" $masterFile $enInit1 $enFin1 $enStep1
mv $masterFile* ..
cd ..
rm -r $1
mkdir raw
mv *.zip raw
"${base}ConvMineralListCSVtoH5.py" *.csv
"${base}NormLearnFile.py" $masterFile".h5"
"${base}ThresholdCrossValidMaker.py" $masterFile"_norm1.h5" $hfsel1 $val1
dir1="hfsel"$hfsel1"_val"$val1
mkdir $dir1
cd $dir1
pwd
cp ../../SpectraKeras_CNN.ini .

"SpectraKeras_CNN" -t "../"$masterFile"_norm1_train-cv_"$dir1".h5" "../"$masterFile"_norm1_test-cv_"$dir1".h5"
