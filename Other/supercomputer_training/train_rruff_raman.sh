#!/bin/bash
#
# TrainLatestRRuff
#
# By Nicola Ferralis <feranick@hotmail.com>
#
# This is script is licensed throughthe GNU Public license v.2.0
#

#base="/Users/feranick/Desktop/"
base="/global/homes/f/feranick/exec/SpectraKeras/"

version=$1
masterFile="AAA-"$version
enInit=110
enFin1=1200
enStep1=1
hfsel1=20
val1=30

export XLA_FLAGS=--xla_gpu_cuda_data_dir=/global/homes/f/feranick/.conda/envs/fera_env
module load python
conda create -y --name fera_env
conda activate fera_env
conda install -y tensorflow numpy pandas pydot graphviz scipy tf-keras keras

folder="rruff-raman_"$1
mkdir $folder
cd $folder
    wget https://rruff.info/zipped_data_files/raman/LR-Raman.zip
    wget https://rruff.info/zipped_data_files/raman/excellent_oriented.zip
    wget https://rruff.info/zipped_data_files/raman/excellent_unoriented.zip
    wget https://rruff.info/zipped_data_files/raman/fair_oriented.zip
    wget https://rruff.info/zipped_data_files/raman/fair_unoriented.zip
    wget https://rruff.info/zipped_data_files/raman/ignore_unoriented.zip
    wget https://rruff.info/zipped_data_files/raman/poor_oriented.zip
    wget https://rruff.info/zipped_data_files/raman/poor_unoriented.zip
    wget https://rruff.info/zipped_data_files/raman/unrated_oriented.zip
    wget https://rruff.info/zipped_data_files/raman/unrated_unoriented.zip
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
"${base}RruffDataMaker.py" $masterFile enInit1 enFin1 enStep1
mv $masterFile* ..
cd ..
"${base}ConvMineralListCSVtoH5.py" *.csv
"${base}NormLearnFile.py" $masterFile".h5"
"${base}ThresholdCrossValidMaker.py" $masterFile"_norm1.h5" $hfsel1 $val1
dir1="hfsel"$hfsel1"_val"$val1
mkdir $dir1
cd $dir1

cp "${base}SpectraKeras_CNN.ini" .

"${base}SpectraKeras_CNN.py" -t "../"$masterFile"_norm1_train-cv_"$dir1".h5" "../"$masterFile"_norm1_test-cv_"$dir1".h5"

conda deactivate
module unload python 

rm -r /global/homes/f/feranick/.conda
