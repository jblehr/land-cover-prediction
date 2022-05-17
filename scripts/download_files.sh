#!/bin/bash

RAW_DIR=$PWD/data/raw
PLANET_FILE="planet.22S"
echo $PLANET_FILE
PLANET_FILE_DIR=$RAW_DIR/$PLANET_FILE
echo $PLANET_FILE_DIR

# The data server also offers downloads with rsync (password m1650201):
# rsync rsync://m1650201@dataserv.ub.tum.de/m1650201/ 
# Download file from mediaTUM to local
echo "Downloading file $PLANET_FILE from Dynamic Earth Net server"
rsync -P rsync://m1650201@dataserv.ub.tum.de/m1650201/$PLANET_FILE.zip $PLANET_FILE_DIR.zip

echo "Unzipping file: $PLANET_FILE_DIR.zip to $PLANET_FILE_DIR"
mkdir $PLANET_FILE_DIR
# Exclude QA files
unzip $PLANET_FILE_DIR.zip -x "*QA*" -d $PLANET_FILE_DIR 

echo "Deleting zipped file: $PLANET_FILE_DIR.zip"
rm $PLANET_FILE_DIR.zip

echo "File download complete"