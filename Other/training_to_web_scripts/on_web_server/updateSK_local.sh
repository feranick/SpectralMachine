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
    exit 1
fi

version=$1

cp -r /var/www/html/SpectraKeras .
cd SpectraKeras || exit 1

#mv ml-raman ml-raman-old
#mv ml-xrd ml-xrd-old
#mkdir ml-raman
#mkdir ml-xrd

cd ml-raman || exit 1
#rm -r *
cp ~/ml/rruff-raman/"$1"/log_SK_Raman.o* .
cp ~/ml/rruff-raman/"$1"/rruff-*/*.csv .
cp ~/ml/rruff-raman/"$1"/rruff-*/AAA_table_names.h5 .
cp ~/ml/rruff-raman/"$1"/rruff-*/hfsel*/* .
# newest .csv only -- a multi-line value here breaks the sed calls below
csvRaman=$(ls -1t *.csv | head -n 1)
cd ..

cd ml-xrd || exit 1
#rm -r *
cp ~/ml/rruff-powder/"$1"/log_SK_Powder.o* .
cp ~/ml/rruff-powder/"$1"/rruff-*/*.csv .
cp ~/ml/rruff-powder/"$1"/rruff-*/AAA_table_names.h5 .
cp ~/ml/rruff-powder/"$1"/rruff-*/hfsel*/* .
csvXrd=$(ls -1t *.csv | head -n 1)
cd ..

echo "$csvRaman"
echo "$csvXrd"

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
