#!/bin/bash

URL="https://rruff.info/zipped_data_files/raman/"
HTML_CONTENT=$(curl -s "$URL")
newDate=$(echo "$HTML_CONTENT" | grep -o -E '[0-9]{4}-[0-9]{2}-[0-9]{2}' | sort -u | tr -d '-')

echo "Training for data:" $newDate

cd rruff-raman
mkdir $newDate
cd $newDate
sbatch ../sub_rruff_raman.sh

cd ../../rruff-powder
mkdir $newDate
cd $newDate
sbatch ../sub_rruff_xrd.sh
