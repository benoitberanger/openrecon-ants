# openrecon-ants

[ANTs](https://github.com/ANTsX/ANTs) using [ANTsPy](https://github.com/ANTsX/ANTsPy) in OpenRecon.  
Brain masking is performed by [SynthStrip](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/)

Based on https://github.com/benoitberanger/openrecon-template

# Features

This OR performs ANTs image operations : 
- N4BiasFieldCorrection
- DenoiseImage
- N4BiasFieldCorrection then DenoiseImage (default)
- DenoiseImage then N4BiasFieldCorrection

There is an option, a checkbox, to **save original images** and intermediate images. (default is _True_)

There is an option, a checkbox, to **apply in brainmask** using SynthStrip. (default is _True_)

# Build

Requirements for building :
- python **3.12**
- jsonschema

 Python environment manager is **strongly** recomanded :
```bash
conda create --name openrecon-ants
conda install python=3.12
pip install jsonschema
```

Build with :
```bash
python build.py
```


# Offline test and dev

Python modules :
- ismrmrd
- pydicom
- pynetdicom
- antspyx
- torch
- surfa

``` bash
pip install ismrmrd pydicom pynetdicom antspyx torch surfa
```
Follow guidelines in https://github.com/benoitberanger/openrecon-template

# TODO

Add fields in the UI to tune `N4BiasFieldCorrection` and `DenoiseImage`

