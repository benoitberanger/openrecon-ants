# openrecon-template

This repo is a template made to help building OpenRecon apps.

The puropose is to build the `i2i` demo provided by Siemens for the SDK on https://www.magnetom.net, using a single build script, without any modification.

For later developments, the first step of a new OpenRecon project it to create a new repo based on this `openrecon-template`, then modify the `app` dir (or any other dir), to finnaly call the building process.


# Requirement

 A Python environment manager is **strongly** recomanded.

- Python
    - `python 3.10` minimum : 100% sure  
    OR
    - `python 3.12` : don't know
- Additional (non-builtin) modules
    - `jsonschema` : a simple `pip install jsonschema` should work


# Build

Simply run the `build.py` script :
```bash
python build.py
```
It will build the `demo-i2i` directory.
All output files will be in the `build` directory

The script is quite verbose, helping troubleshooting at each step of the process in case of error.

To build a specific dir, such as the `app` dir :
```bash
python build.py --dirname app
```

# Outputs
All output files will be placed in a _build_ subdir.
The finale file, ready for the upload on the magnet will be the _.zip_ file.

Building the `demo-i2i` dir (default option) you should have this :
```bash
$ ls -1 build
i2i_json_ui.json
i2i.py
OpenReconSchema_1.1.0.json
OpenRecon_SiemensHealthineersAG_PythonMRDi2i_V1.0.0.Dockerfile
OpenRecon_SiemensHealthineersAG_PythonMRDi2i_V1.0.0.pdf
OpenRecon_SiemensHealthineersAG_PythonMRDi2i_V1.0.0.tar
OpenRecon_SiemensHealthineersAG_PythonMRDi2i_V1.0.0.zip
```

Building the `app` dir : 
```bash
$ ls -1 build
i2i-save-original-images_json_ui.json
i2i-save-original-images.py
OpenRecon_openrecon-template_i2i-save-original-images_V1.0.0.Dockerfile
OpenRecon_openrecon-template_i2i-save-original-images_V1.0.0.pdf
OpenRecon_openrecon-template_i2i-save-original-images_V1.0.0.tar
OpenRecon_openrecon-template_i2i-save-original-images_V1.0.0.zip
OpenReconSchema_1.1.0.json
```

# App

## Objective : templatization & separation of environments

The `app` dir will contained the versioned code.  
The `python-ismrmrd-server` dir is **NOT** versioned.  
To _add_ your app files files in the `python-ismrmrd-server`, create symbolic links.

## How to test locally the reconstruction

### Prepare the python environment 

Open a Terminal in the repository directory, then :

```bash
# setup Conda environment
conda create --name openrecon-template
conda activate openrecon-template
conda install python=3.12
conda install ipython
pip install ismrmrd
pip install pydicom
pip install pynetdicom

# create .py file symbolic link in `python-ismrmrd-server`
# this is how the MRD client/server will behave : all files/modules in the same dir
cd python-ismrmrd-server
ln -s ../app/i2i-save-original-images.py .
```
Repeat the `ln -s` for the files you need.



### Convert your test dicom into MRD

Put DICOMs in a dir then convert to MRD (.h5 file)
```bash
python python-ismrmrd-server/dicom2mrd.py -o <path_to_data>/in/test.h5 <path_to_data>/in/
```

### Start the reco

In a terminal or an IDE like VSCode, start the `main.py`. This will help a lot debugging.

Send data using `client.py`
```bash
python python-ismrmrd-server/client.py -o <path_to_data>/out/reco_test.h5 -c <myrecon> <path_to_data>/in/test.h5 
```

### Convert the output of your reco to DICOM for visualization
```bash
python python-ismrmrd-server/mrd2dicom.py -o <path_to_data>/out/ <path_to_data>/out/reco_test.h5
```

Use any DICOM viewer to check visually.

### All-in-one

Start the `main.py` in VSCode.
It can be in Debug mode, with breakpoints.
Then : 

```bash
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
```

## VSCode tips

I found that, when you modify the `<reco>.py` file when the `main.py` is running, the code is not updated => you need to restart the server (started by the main.py) so the `<reco>.py` is reloaded.

