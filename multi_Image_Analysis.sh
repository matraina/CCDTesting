#bash script to analyse images produced scanning across different values of setup parameters                                                                     
#!/bin/bash                                                                                                                                                      

dir=$1

#put aside relics from previous scan
DIRECTORY="${dir}/processed"
DIRECTORYOLD="${dir}/old/old_processed"
if [ -d "$DIRECTORY" ]; then
    mkdir ${dir}/old
    if [ -d "$DIRECTORYOLD" ]; then
        echo "Several runs have been done on this parameters scan images. Please check scan directory and ensure that it contains only relevant code and directory 'raw' with .fits files"
        return 0
    fi
    mv ${dir}/processed ${dir}/old/old_processed
fi

DIRECTORY="${dir}/header"
DIRECTORYOLD="${dir}/old/old_header"
if [ -d "$DIRECTORY" ]; then
    if [ -d "$DIRECTORYOLD" ]; then
        echo "Several runs have been performed on this parameters scan images. Please check scan directory and ensure that it contains only relevant code and directory 'raw' with .fits files"
        return 0
    fi
    mv ${dir}/header ${dir}/old/old_header
fi

DIRECTORY="$1/reports"
DIRECTORYOLD="$1/old/old_reports"
if [ -d "$DIRECTORY" ]; then
    if [ -d "$DIRECTORYOLD" ]; then
        echo "Several runs have been performed on this parameters scan images. Please check scan directory and ensure that it contains only relevant code and directory 'raw' with .fits files"
        return 0
    fi
    mv ${dir}/reports ${dir}/old/old_reports
fi

#create directory structure
mkdir ${dir}/processed
mkdir ${dir}/header
mkdir ${dir}/reports

imgname=$2
prcimgname=$3
for par in $(seq $4 1 $5);
do
    python3 main.py ${dir} raw/${imgname}${par} processed/${prcimgname}${par} 5 -1 
done

rm -r ${dir}/reports/*.tex
