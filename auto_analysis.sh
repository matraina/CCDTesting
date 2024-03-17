#!/bin/bash
executable=$1
imgname=$2
prcimgname=$3
startimg=$4
endimg=$5

for par in $(seq $startimg 1 $endimg);
do
    ./${executable}.py ${imgname}${par} ${prcimgname}${par}
done

today=$(date +"%Y_%m_%d")
label1="dc_values"
label2="col_profile_L"
label3="col_profile_U"
sep="_"
oldfilename1="$today$sep$label1.txt"
filename1="$today$sep$label1$sep$startimg$sep$endimg.txt"
oldfilename2="$today$sep$label2.txt"
filename2="$today$sep$label2$sep$startimg$sep$endimg.txt"
oldfilename3="$today$sep$label3.txt"
filename3="$today$sep$label3$sep$startimg$sep$endimg.txt"

mv $oldfilename1 $filename1
mv $oldfilename2 $filename2
mv $oldfilename3 $filename3

./plot_analysis_results.py $startimg $endimg $filename1 $filename2 $filename3
