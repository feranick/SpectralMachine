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

cp -r /var/www/html/SpectraKeras .
cd SpectraKeras
#mv ml-raman ml-raman-old
#mv ml-xrd ml-xrd-old
#mkdir ml-raman
#mkdir ml-xrd

cd ml-raman
rm -r *
rsync -azv user@server.com:/home/user/ml/rruff-raman/$1/log_SK_Raman.o* .
rsync -azv user@server.com:/home/user/ml/rruff-raman/$1/rruff-*/*.csv .
rsync -azv user@server.com:/home/user/ml/rruff-raman/$1/rruff-*/AAA_table_names.h5 .
rsync -azv user@server.com:/home/user/ml/rruff-raman/$1/rruff-*/hfsel*/* .
csvRaman=`ls *.csv`
cd ..

cd ml-xrd
rm -r *
rsync -azv user@server.com:/home/user/ml/rruff-powder/$1/log_SK_Powder.o* .
rsync -azv user@server.com:/home/user/ml/rruff-powder/$1/rruff-*/*.csv .
rsync -azv user@server.com:/home/user/ml/rruff-powder/$1/rruff-*/AAA_table_names.h5 .
rsync -azv user@server.com:/home/user/ml/rruff-powder/$1/rruff-*/hfsel*/* .
csvXrd=`ls *.csv`
cd ..

echo $csvRaman
echo $csvXrd

if [[ -z "$csvRaman" || -z "$csvXrd" ]]; then
    echo "ERROR: could not find a .csv in ml-raman and/or ml-xrd"
    exit 1
fi

# Replace the whole matching line rather than grepping for its old contents:
# no need to escape whatever happens to already be in index.php.
search_string_Raman='<br>Current Raman ML model:'
new_string_Raman="    <br>Current Raman ML model: <a href=\"ml-raman/$csvRaman\">AAA-$1_norm1_train</a>"
sed -i "s|^.*${search_string_Raman}.*$|${new_string_Raman}|" index.php

search_string_Xrd='<br>Current XRD ML model:'
new_string_Xrd="    <br>Current XRD ML model: <a href=\"ml-xrd/$csvXrd\">AAA-Powder-$1_norm1_train</a>"
sed -i "s|^.*${search_string_Xrd}.*$|${new_string_Xrd}|" index.php

cd ..
