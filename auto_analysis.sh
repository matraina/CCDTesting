#!/bin/bash
executable=$1
imgname=$2
prcimgname=$3
startimg=$4
endimg=$5

for par in $(seq $startimg 1 $endimg);
do
    py3 ${executable}.py ${imgname}${par} ${prcimgname}${par}
done