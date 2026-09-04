""" Other server utility functions (sending original images, reading OpenRecon arguments, diagnostics and normalisation)"""

import base64
import logging

import ismrmrd
import numpy as np

from server.connection import Connection


def send_original_images(images: list[ismrmrd.Image], connection: Connection) -> None:
    """
    Return a copy of original images unprocessed.

    Parameters
    ----------
    images : list of ismrmrd.Image
        List of original MRD Images
    connection : Connection
        Active MRD connection.
    """

    images_copy = []

    for image in images:
        tmpImg = image

        # Ensure Keep_image_geometry is set to not reverse image orientation
        tmpMeta = ismrmrd.Meta.deserialize(tmpImg.attribute_string)
        tmpMeta['Keep_image_geometry'] = 1
        tmpImg.attribute_string = tmpMeta.serialize()

        images_copy.append(tmpImg)
        
    connection.send_image(images_copy)


def check_OR_arguments(configJSON: dict | None, arg_name: str, arg_type: type, arg_default: any = None) -> any:
    """
    Return the value of the OpenRecon arguments with the appropriate type.
    
    In OpenRecon, the config passes all parameter values as strings, 
    regardless of their declared type in the JSON UI definition. This 
    function reads the value from ``configJSON['parameters'][arg_name]`` 
    and casts it to the requested type. If the parameter is missing or 
    the config is invalid, the default value is returned.
    
    Parameters
    ----------
    configJSON : dict or None
        JSON configuration sent by the client. Expected to
        contain a ``'parameters'`` key mapping parameter names to their
        string values. If None or not a dict, arg_default is returned.
    arg_name : str
        Name of the parameter to look up in ``configJSON['parameters']``.
    arg_type : type
        Expected Python type of the parameter. Supported types:
        ``str``, ``bool``, ``int``, ``float``.

    arg_default : any, optional
        Value returned when configJSON is not a dict, when
        ``'parameters'`` is absent, or when arg_name is not found.
        Default is None.

    Returns
    -------
    any
        Parameter value cast to arg_type, or arg_default if not found.
    """
    
    if not isinstance(configJSON, dict):
        logging.warning(f"config is not a dictionary. {arg_name} set to {arg_default} by default.")
        return arg_default

    if ('parameters' in configJSON) and (arg_name in configJSON['parameters']):
        logging.info(f"found config['parameters']['{arg_name}'] : type={type(configJSON['parameters'][arg_name])} content={configJSON['parameters'][arg_name]}")
        arg_value =  configJSON['parameters'][arg_name]
    else:
        logging.warning(f"config['parameters']['{arg_name}'] NOT FOUND !! Value set to {arg_default}.")
        return arg_default

    # in OR, the config only provides strings, so need to cast to the correct type
    if arg_type is str:
        pass
    elif arg_type is bool:
        if type(arg_value) is not bool:
            if   arg_value.lower() == 'true' : arg_value = True
            elif arg_value.lower() == 'false': arg_value = False
            else: raise ValueError(f"{arg_name} is detected as `str` but is not 'True' or 'False' ! Cannot cast it to `bool`")
    elif arg_type is int:
        if type(arg_value) is not int:
            arg_value = int(arg_value)
    elif arg_type is float:
        if type(arg_value) is not float:
            arg_value = float(arg_value)
    else:
        raise TypeError('wrong type in the config)')

    logging.info(f'{arg_name} = {arg_value}')
    return arg_value


def display_diagnostic(head: ismrmrd.ImageHeader, meta: ismrmrd.Meta, ICEminihead_decode: bool=False) -> dict:
    """
    Log geometric and acquisition properties of an image group.

    Extracts key spatial parameters from one image header and
    optionally decodes the Siemens ICE MiniHeader from one Meta
    object if present.

    Parameters
    ----------
    head : list of ismrmrd.ImageHeader
        Image headers.
    meta : list of ismrmrd.Meta
        Deserialised Meta objects.
    ICEminihead_decode : bool
        Set to True to decode and log the Siemens ICE MiniHeader 
        from one Meta object if present. (Default to False).

    Returns
    -------
    dict with the following keys:

    - ``'matrix'``    — np.ndarray [x, y, z]
    - ``'fov'``       — np.ndarray [x, y, z]
    - ``'voxelsize'`` — np.ndarray [x, y, z]
    - ``'read_dir'``  — np.ndarray [x, y, z]
    - ``'phase_dir'`` — np.ndarray [x, y, z]
    - ``'slice_dir'`` — np.ndarray [x, y, z]
    """

    # Optional serialization of ICE MiniHeader
    if 'IceMiniHead' in meta and ICEminihead_decode:
        logging.info("IceMiniHead: %s", base64.b64decode(meta['IceMiniHead']).decode('utf-8'))

    # Diagnostic info
    matrix    = np.array(head.matrix_size  [:]) 
    fov       = np.array(head.field_of_view[:])
    voxelsize = fov/matrix
    read_dir  = np.array(head.read_dir )
    phase_dir = np.array(head.phase_dir)
    slice_dir = np.array(head.slice_dir)
    logging.info(f'MRD computed maxtrix [x y z] : {matrix   }')
    logging.info(f'MRD computed fov     [x y z] : {fov      }')
    logging.info(f'MRD computed voxel   [x y z] : {voxelsize}')
    logging.info(f'MRD read_dir         [x y z] : {read_dir }')
    logging.info(f'MRD phase_dir        [x y z] : {phase_dir}')
    logging.info(f'MRD slice_dir        [x y z] : {slice_dir}')

    diagnostic = {
        'matrix': matrix,
        'fov'   : fov,
        'voxelsize' : voxelsize,
        'read_dir'  : read_dir,
        'phase_dir' : phase_dir,
        'slice_dir' : slice_dir
    }

    return diagnostic


def normalise(data: np.array) -> np.array:
    """
    Normalise pixel data of MRD images.

    Parameters
    ----------
    data : np.ndarray
        Stacked MRD image data.
    
    Returns
    -------
    np.ndarray
        normalise data.
    """
    BitsStored = 12
    maxVal = 2**BitsStored - 1

    if data.max() == 0:
        raise ZeroDivisionError

    data *= maxVal/data.max()
    np.around(data, out=data)

    return data


def MRD5Dto3D(data_mrd5D: np.array) -> np.array:
    """
    Convert a 5D MRD image stack to a 3D array (y, x, img)

    Transposes the MRD axis order [img, cha, z, y, x] to the spatial
    convention [y, x, img], keeping only the first channel (cha=0) and
    first z-slice (z=0). This assumes single-channel, single-slice 2D
    images as typically produced by MRI reconstructions.

    Parameters
    ----------
    data_mrd5D : np.ndarray
        Stacked MRD image data, shape [img, cha, z, y, x].

    Returns
    -------
    np.ndarray
        3D array of shape [y, x, img].
    """

    # Reformat data to [y x z cha img], i.e. [row col] for the first two dimensions
    data_mrd5D = data_mrd5D.transpose((3, 4, 2, 1, 0))

    logging.debug("Original image data is size %s" % (data_mrd5D.shape,))

    # Reformat data from [y x z cha img] to [y x img]
    data_3d = data_mrd5D[:,:,0,0,:]
        
    return data_3d
