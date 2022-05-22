#!/bin/bash

RAW_DIR=$PWD/data/raw
for PLANET_FILE in planet.33S
do
    echo $PLANET_FILE
    PLANET_FILE_DIR=$RAW_DIR/$PLANET_FILE

    # The data server also offers downloads with rsync (password m1650201):
    # rsync rsync://m1650201@dataserv.ub.tum.de/m1650201/ 
    # Download file from mediaTUM to local
    echo "Downloading file $PLANET_FILE from Dynamic Earth Net server"
    rsync -P rsync://m1650201@dataserv.ub.tum.de/m1650201/$PLANET_FILE.zip $PLANET_FILE_DIR.zip

    echo "Unzipping file: $PLANET_FILE_DIR.zip to $PLANET_FILE_DIR"
    mkdir $PLANET_FILE_DIR
    # Exclude QA files and only get first file of the month
    unzip $PLANET_FILE_DIR.zip "*01.tif" -x "*QA*" -d $PLANET_FILE_DIR 

    echo "Deleting zipped file: $PLANET_FILE_DIR.zip"
    rm $PLANET_FILE_DIR.zip
    echo "Processing files"
    python scripts/process_files.py $PLANET_FILE
done
