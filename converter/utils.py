"""Converter utility functions: MRD file validation and slice-position computation."""

import logging
import os
import re

import h5py
import ismrmrd
import numpy as np


def slice_pos(img: ismrmrd.Image) -> float:
    """
    Compute the scalar position of an image along the slice direction.
 
    Projects the image corner position (LPS) onto the slice normal vector
    (also LPS) using a dot product. The result is a signed scalar that
    increases monotonically from the first to the last slice of the stack,
    regardless of patient orientation.
 
    Parameters
    ----------
    img : ismrmrd.Image
        A single MRD image. The following ImageHeader fields are used:
        - position  : [x, y, z] LPS coordinates of the image corner (mm)
        - slice_dir : [x, y, z] unit vector normal to the slice plane (LPS)
 
    Returns
    -------
    float
        Signed scalar position along the slice normal (mm).
    """

    h = img.getHead()
    position = np.array(h.position,  dtype=float)
    slice_dir = np.array(h.slice_dir, dtype=float)

    return float(np.dot(position, slice_dir))


#### MRD file checks ##########################################################

def check_MRDfile(filename: str, in_group: str, out_folder: str) -> str | None:
    """
    Validate an MRD HDF5 file and resolve the input group and output folder.
 
    Performs the following checks and side effects:
      - Opens the file and lists top-level HDF5 groups.
      - Selects the last group if in_group is not specified.
      - Creates out_folder on disk if it does not already exist;
        defaults to the original filename if not specified.
      - Verifies that each image sub-group inside the selected group contains
        the three mandatory datasets: data, header, attributes.
 
    Parameters
    ----------
    filename : str
        Path to the MRD (.h5) file.
    in_group : str or None
        Top-level HDF5 group to read from. If None or empty, the last group
        in the file is selected automatically.
    out_folder : str or None
        Directory where NIfTI files will be written. Created if absent.
        If None or empty, defaults to the original filename.
 
    Returns
    -------
    str or None
        The resolved group name to use for reading, or None if validation
        failed (file unreadable, group not found, or malformed image data).
    """

    # Check file and get group name
    dset = h5py.File(filename, 'r')
    if not dset:
        logging.error(f"Not a valid dataset: {filename}")
        return None

    dsetNames = list(dset.keys())
    logging.info(f"File {filename} contains {len(dsetNames)} groups:")
    logging.info(" \n  ".join(dsetNames))

    if not in_group:
        if len(dsetNames) > 1:
            logging.info("Input group not specified -- selecting most recent")
        in_group = dsetNames[-1]

    if not out_folder:
        out_folder = re.sub('.h5$', '', filename)
        logging.info(f"Output folder not specified -- using {out_folder}")

    if in_group not in dset:
        logging.error(f"Could not find group {in_group}")
        return None

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    group = dset.get(in_group)

    logging.info(f"Reading data from group '{in_group}' in file '{filename}'")

    # mrdImg data is stored as:
    #   /group/config              text of recon config parameters (optional)
    #   /group/xml                 text of ISMRMRD flexible data header (optional)
    #   /group/image_0/data        array of IsmrmrdImage data
    #   /group/image_0/header      array of ImageHeader
    #   /group/image_0/attributes  text of mrdImg MetaAttributes

    isImage = True
    imageNames = group.keys()
    logging.info(f"Found {len(imageNames)} mrdImg sub-groups: {', '.join(imageNames)}")

    for imageName in imageNames:
        if imageName in ("config", "config_file", "xml", "configAdditional"):
            continue

        mrdImg = group[imageName]
        if not (('data' in mrdImg) and ('header' in mrdImg) and ('attributes' in mrdImg)):
            isImage = False

    dset.close()

    if (isImage is False):
        logging.error("File does not contain properly formatted MRD raw or mrdImg data")
        return None

    return in_group
