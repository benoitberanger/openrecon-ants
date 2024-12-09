# openrecon-ants

![ANTs V2 in XA60A on a 7T Terra.X](doc/OpenRecon_ANTS_V2_FLAIR-CP_FLAIR-UP_DIR-UP_SAG_blur.png)  
MR Host screenshot from a Siemens 7T Terra.X in XA60A.  
_Sequence_ Non-selective 3D SPACE.  
_From left to right_ Original, brain mask (SynthStrip), N4BiasFieldCorrection (ANTs) in brain mask, DenoiseImage (ANTs) in brain mask after N4BiasFieldCorrection.  
_From top to bottom_ 3D FLAIR with Circular Polarization (CP), 3D FLAIR with Universal Pulses (UP), 3D DIR with UP.  

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

