# openrecon-ants

[ANTs](https://github.com/ANTsX/ANTs) using [ANTsPy](https://github.com/ANTsX/ANTsPy) in OpenRecon.

Based on https://github.com/benoitberanger/openrecon-template

# Features

This OR performs ANTs image operations : 
- N4BiasFieldCorrection
- DenoiseImage
- N4BiasFieldCorrection + DenoiseImage
- DenoiseImage + N4BiasFieldCorrection

There is an option, a checkbox, to **save original images** and intermediate images.


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

``` bash
pip install ismrmrd pydicom pynetdicom antspyx
```
Follow guidelines in https://github.com/benoitberanger/openrecon-template

# TODO

Add fields in the UI to tune `N4BiasFieldCorrection` and `DenoiseImage`

