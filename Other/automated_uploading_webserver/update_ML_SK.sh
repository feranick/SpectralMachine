#!/bin/bash
#
# update_ML_SK
#
# By Nicola Ferralis <feranick@hotmail.com>
#
# This is script is licensed throughthe GNU Public license v.2.0
#

if [[ -z "$1" ]]; then
echo "The first argument ($1) is empty or unset."
exit
fi

version=$1
server="nicola@mochi.mit.edu:/home/nicola"

cp -r /var/www/html/SpectraKeras .
cd SpectraKeras
#mv ml-raman ml-raman-old
#mv ml-xrd ml-xrd-old
#mkdir ml-raman
#mkdir ml-xrd

cd ml-raman
rm -r *
scp $server/ml/rruff-raman/$1/log_SK_Raman.o* .
scp $server/ml/rruff-raman/$1/rruff-*/*.csv .
scp $server/ml/rruff-raman/$1/rruff-*/AAA_table_names.h5 .
scp $server/ml/rruff-raman/$1/rruff-*/hfsel*/* .
csvRaman=`ls *.csv`
cd ..

cd ml-xrd
rm -r *
scp $server/ml/rruff-powder/$1/log_SK_Powder.o* .
scp $server/ml/rruff-powder/$1/rruff-*/*.csv .
scp $server/ml/rruff-powder/$1/rruff-*/AAA_table_names.h5 .
scp $server/ml/rruff-powder/$1/rruff-*/hfsel*/* .
csvXrd=`ls *.csv`
cd ..

echo $csvRaman
echo $csvXrd

search_string_Raman='<br>Current Raman ML model:'
old_string_Raman=`grep "$search_string_Raman" index.php`
new_string_Raman="    <br>Current Raman ML model: <a href=\"ml-raman/$csvRaman\">AAA-$1_norm1_train-cv_hfsel20_val37</a>"
sed -i "s|$old_string_Raman|$new_string_Raman|g" index.php

search_string_Xrd='<br>Current XRD ML model:'
old_string_Xrd=`grep "$search_string_Xrd" index.php`
new_string_Xrd="    <br>Current XRD ML model: <a href=\"ml-xrd/$csvXrd\">AAA-Powder-$1_norm1_train-cv_hfsel10_val22</a>"
sed -i "s|$old_string_Xrd|$new_string_Xrd|g" index.php

cd ..
