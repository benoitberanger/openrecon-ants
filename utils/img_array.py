"""Building and manipulating the structured nD array of MRD images."""

import logging
from enum import IntEnum

import ismrmrd
import numpy as np


class mrd_indexes(IntEnum):
    """
    Defines the axis order of the MRD image array.

    Each member maps a dimension name to its axis index in the array.
    To change the axis order or add a dimension, edit only this enum,
    all functions in this module use it automatically, EXCEPT get_subarray()
    which requires a manual update if you add or delete a dimension.

    To add a dimension (e.g. 'channels'):
        1. Add it here:   channels = 8
        2. The attribute name must match the ismrmrd.Image attribute exactly.
           (see documentation: https://ismrmrd.readthedocs.io/en/latest/mrd_image_data.html)
        3. Add the corresponding parameter to get_subarray():
               def get_subarray(..., img_channels: int | slice | None = None)
        4. Add it to the args dict in get_subarray():
               mrd_indexes.channels: img_channels,

    To change the axis order, reassign the integer values.
    """
    slice               = 0
    contrast            = 1
    average             = 2
    phase               = 3
    repetition          = 4
    set                 = 5
    image_type          = 6
    image_series_index  = 7


def build_image_array(img_list: list[ismrmrd.Image]) -> np.ndarray[ismrmrd.Image] :
    """
    Build a nD structured array from a flat list of MRD images.

    Each cell of the array corresponds to a unique combination of loop
    counters and image type. The array axes follow this order of the enum,
    if unchanged :

    [slice, contrast, average, phase, repetition, set, image_type, image_series_index]

    MRD index values can be used directly as array indices.

    Parameters
    ----------
    img_list : list of ismrmrd.Image
        Flat list of MRD images, typically as received from the client.

    Returns
    -------
    np.ndarray
        nD object array of shape (if enum unchanged):
        (n_slices, n_contrasts, n_averages, n_phases,n_repetitions, 
        n_sets, n_image_types, n_image_series_index).
        Empty cells contain None.
    """
    
    dimensions = list(mrd_indexes)
    dim_names = [dim.name for dim in dimensions]

    # Found the max value for each dimension
    max_idx = { dim: 0 for dim in mrd_indexes}

    for img in img_list:
        for dim in mrd_indexes:
            max_idx[dim] = max(max_idx[dim], getattr(img, dim.name))

    shape = tuple(max_idx[dim] + 1 for dim in dimensions)

    # Initialize an empty array with the right size
    img_array = np.full(shape, None, dtype=object)

    # Fill the array with the images
    for img in img_list:
        key = tuple(getattr(img, dim.name) for dim in dimensions)

        if img_array[key] is None:
            img_array[key] = [img]
        else:
            img_array[key].append(img)

    header_format = "  [" + " ".join(f"%-{max(len(n), 6)}s" for n in dim_names) + "]"
    value_format  = "  [" + " ".join(f"%-{max(len(n), 6)}d" for n in dim_names) + "]"

    logging.info("MRD image array :")
    logging.info(header_format, *dim_names)
    logging.info(value_format,  *img_array.shape)

    return img_array


def validate_index(value, dim_size: int, dim_name: str) -> bool:
    """
    Validate that an index (int or slice) is within bounds for a given dimension.
    The purpose of this function is mainly to make clear error message in case of
    invalid index.

    Parameters
    ----------
    value : None, int, or slice
        Index to validate. None is always valid (selects the full axis).
    dim_size : int
        Size of the dimension in the array.
    dim_name : str
        Human-readable dimension name used in error messages.
    
    Returns
    -------
    bool
        True if the index is valid, False otherwise.
    """
    if value is None:
        return True

    if isinstance(value, int):
        if not (-dim_size <= value and value < dim_size):
            logging.warning(
                "Index out of range for dimension '%s': requested index %d, but dimension size is %d (valid range: [%d, %d]).",
                dim_name, value, dim_size, -dim_size, dim_size - 1
            )
            return False

    elif isinstance(value, slice):
        start, stop, step = value.indices(dim_size)
        indices = range(start, stop, step)
        if len(indices) == 0:
            logging.warning(
                "Slice out of range for dimension '%s': slice(%s, %s, %s) selects 0 elements from a dimension of size %d.",
                dim_name, value.start, value.stop, value.step, dim_size
            )
            return False
    
    return True


def get_subarray(img_array: np.ndarray[ismrmrd.Image], 
                 img_slice: int | slice | None = None, 
                 img_contrast: int | slice | None = None, 
                 img_average: int | slice | None = None, 
                 img_phase: int | slice | None = None, 
                 img_repetition: int | slice | None = None, 
                 img_set: int | slice | None = None, 
                 img_image_type: int | slice | None = None,
                 img_image_series_index: int | slice | None = None) -> np.ndarray[ismrmrd.Image]:
    """
    Extract a subarray from a 8D MRD images array by selecting specific indices
    along one or more dimensions.

    Array axis order:
    [slice, contrast, average, phase, repetition, set, image_type, image_series_index]
        
    Any argument left as ``None`` selects all indices along that dimension
    (equivalent to ``:`` in NumPy slice notation).
    
    Parameters
    ----------
    img_array : np.ndarray
        nD MRD images array as returned by build_image_array().
    img_slice : int or slice or None
        Index along the slice axis, or None for all slices.
    img_contrast : int or slice or None
        Index along the contrast axis, or None for all contrasts.
    img_average : int or slice or None
        Index along the average axis, or None for all averages.
    img_phase : int or slice or None
        Index along the phase axis, or None for all phases.
    img_repetition : int or slice or None
        Index along the repetition axis, or None for all repetitions.
    img_set : int or slice or None
        Index along the set axis, or None for all sets.
    img_image_type : int or slice or None
        Index along the image_type axis. Use ismrmrd.IMTYPE_* constants
        directly (e.g. ismrmrd.IMTYPE_MAGNITUDE = 1), or None for all types.
    img_image_series_index : int or slice or None
        Index along the image_series_index axis, or None for all sets.

    Returns
    -------
    np.ndarray
        Subarray view corresponding to the requested indices.
    """

    dimension_names = list(mrd_indexes)

    args = {
        mrd_indexes.slice:              img_slice,
        mrd_indexes.contrast:           img_contrast,
        mrd_indexes.average:            img_average,
        mrd_indexes.phase:              img_phase,
        mrd_indexes.repetition:         img_repetition,
        mrd_indexes.set:                img_set,
        mrd_indexes.image_type:         img_image_type,
        mrd_indexes.image_series_index: img_image_series_index
    }

    # Check if mrd_indexes and args are sync
    missing = set(mrd_indexes) - set(args)
    if missing:
        raise NotImplementedError(
            "get_subarray() is missing parameters for the following "
            "mrd_indexes members: %s. "
            "Add the corresponding 'img_<name>' parameter and entry in args. "
            "See mrd_indexes docstring for instructions."
            % [dim.name for dim in missing]
        )

    # Validate every requested index against the actual array shape
    for dim in dimension_names:
        valid = validate_index(args[dim], img_array.shape[dim], dim.name)
        if valid == False :
            return np.array([])
    
    def to_index(x):
        if x is None:
            return slice(None)
        if isinstance(x, slice):
            return x
        # return slice(None) if x is the last element
        stop = x + 1 if x != -1 else None
        return slice(x, stop)
    
    idx = tuple(to_index(args[dim]) for dim in dimension_names)
    
    new_array = img_array[idx]
    logging.info(f"Subarray shape: {new_array.shape}")
    logging.info(f"number of images in the subarray: {np.count_nonzero(new_array)}")
    return new_array


def get_type_magnitude(img_array: np.ndarray[ismrmrd.Image]) -> np.ndarray[ismrmrd.Image]:
    """
    Extract the magnitude images from the MRD images array.
    Shorthand for get_subarray(img_array, img_image_type=ismrmrd.IMTYPE_MAGNITUDE).

    Parameters
    ----------
    img_array : np.ndarray
        nD MRD images array as returned by build_image_array().

    Returns
    -------
    np.ndarray
        nD subarray of shape [slice, contrast, average, phase, repetition, set, image_series_index]
        containing only magnitude images.
    """
    return get_subarray(img_array, img_image_type=ismrmrd.IMTYPE_MAGNITUDE)


def get_type_phase(img_array: np.ndarray[ismrmrd.Image]) -> np.ndarray[ismrmrd.Image]:
    """
    Extract the phase images from the MRD images array.
    Shorthand for get_subarray(img_array, img_image_type=ismrmrd.IMTYPE_PHASE).

    Parameters
    ----------
    img_array : np.ndarray
        nD MRD images array as returned by build_image_array().

    Returns
    -------
    np.ndarray
        nD subarray of shape [slice, contrast, average, phase, repetition, set, image_series_index]
        containing only phase images.
    """
    return get_subarray(img_array, img_image_type=ismrmrd.IMTYPE_PHASE)


def get_contrast(img_array: np.ndarray[ismrmrd.Image], img_contrast: int) -> np.ndarray[ismrmrd.Image]:
    """
    Extract all images for a specific contrast index.
    Shorthand for get_subarray(img_array, img_contrast=contrast).

    Parameters
    ----------
    img_array : np.ndarray
        nD MRD images array as returned by build_image_array().
    contrast : int
        Contrast index to extract (0-based, matches MRD idx.contrast).

    Returns
    -------
    np.ndarray
        nD subarray of shape [slice, average, phase, repetition, set, image_type, image_series_index]
        containing all images for the requested contrast.
    """
    return get_subarray(img_array, img_contrast = img_contrast)


def flatten(arr: np.ndarray[ismrmrd.Image]) -> list[ismrmrd.Image]:
    """
    Return a flat list of all MRD images contained in an array or subarray.

    Iterates over every cell of the array in row-major order, skipping
    None cells, and extends the output list with the images found in each
    non-empty cell. Cell order matches NumPy's default flat iteration:
    slice -> contrast -> average -> phase -> repetition -> set -> image_type.

    Parameters
    ----------
    arr : np.ndarray
        Full nD MRD image array or any subarray.

    Returns
    -------
    list of ismrmrd.Image
        All non-None images in the array, in row-major iteration order.
    """
    images = []

    for cell in arr.flat:
        if cell is None:
            continue
        images.extend(cell)

    return images


def stack_images(images: list[ismrmrd.Image], dtype = np.float32) -> tuple[np.ndarray, list, list]:
    """
    Stack pixel data, headers and metadata from a list of MRD images.

    Parameters
    ----------
    images : list of ismrmrd.Image
        list of all the images to stack.
    dtype : np.dtype, optional
        Output array dtype. Default is np.float32.

    Returns
    -------
    data : np.ndarray
        Stacked pixel data, shape [img, cha, z, y, x], cast to dtype.
    head : list of ismrmrd.ImageHeader
        Headers in the same order as data.
    meta : list of ismrmrd.Meta
        Deserialised Meta objects in the same order as data.

    Raises
    ------
    ValueError
        If img_array contains no images (all cells are None).
    """

    if not images:
        raise ValueError("stack_images: no images found.")
    
    logging.debug("Stacking %d images of dtype %s", len(images), images[0].data.dtype)
    
    # MRD supposed organisation [img cha z y x]
    data = np.stack([img.data for img in images]).astype(dtype)
    head = [img.getHead()                                  for img in images]
    meta = [ismrmrd.Meta.deserialize(img.attribute_string) for img in images]

    logging.debug("Stacked data shape [img, cha, z, y, x]: %s", data.shape)
    
    return data, head, meta
