#!/bin/bash

## path management
IN_DIR=data/in
OUT_DIR=data/out
DATASET_NAME=test

## clean out dir
rm $OUT_DIR/*dcm $OUT_DIR/*h5

## send h5 MRD dataset using th client -> the server will process them
TARGET_CONFIG=i2i-save-original-images
python python-ismrmrd-server/client.py -o $OUT_DIR/OR_$DATASET_NAME.h5 -c $TARGET_CONFIG $IN_DIR/$DATASET_NAME.h5

# convert fresh OR processed MRD dataset into DICOM for visu
python python-ismrmrd-server/mrd2dicom.py -o $OUT_DIR/ $OUT_DIR/OR_$DATASET_NAME.h5

# use your favorite DICOM viewer
mrview $OUT_DIR/ -mode 2

