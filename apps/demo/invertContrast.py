"""Example application module: inverts the contrast of the received images."""

import gc
import logging
import os

import ismrmrd
import numpy as np

from converter.mrd2nifti import nifti_from_image_array
from utils.OutputSeries import OutputSeries, ProcessImageResult
from utils.img_array import flatten, get_subarray, mrd_indexes, stack_images
from utils.memory import log_memory, log_memory_delta, timeit
from utils.utils import check_OR_arguments, display_diagnostic, normalise


# Folder for debug output files
debugFolder = "/tmp/share/debug"

@timeit
def process_image(img_array: np.ndarray[ismrmrd.Image], configJSON: dict | None, metadata) -> ProcessImageResult:
    """
    Invert contrast process image.

    Parameters
    ----------
    img_array : np.ndarray
        nD MRD image array as returned by build_image_array().
    configJSON : dict or None
        JSON configuration from the client.
    metadata : ismrmrd.xsd.ismrmrdHeader or str
        MRD header.

    Returns
    -------
    list of tuple (np.ndarray, list of ismrmrd.ImageHeader, list of ismrmrd.Meta)
        One tuple per image type present in the dataset, in the order
        they were encountered. Each data array has shape
        [img, cha, z, y, x], dtype int16
    """
    
    # Create debug folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    logging.info(f'------------------------------------------------')
    logging.info(f'     invertContrast called')
    logging.info(f'------------------------------------------------')
    
    mem = log_memory("process_image", "Beginning")
    
    BitsStored = 12
    maxVal = 2**BitsStored - 1

    # --- Dimensions ------------------------------------------------------
    # Get the number of image_type (img_array axis 6)
    n_image_type = img_array.shape[mrd_indexes.image_type]
    image_type_name = ('', 'MAGNITUDE', 'PHASE', 'REAL', 'IMAG', 'COMPLEX', 'RGB')

    # --- Treat all types of images ---------------------------------------
    series = OutputSeries()
    
    for serie_index in range(0, img_array.shape[mrd_indexes.image_series_index]):

        for image_type in range(0, n_image_type):
            sub_array = get_subarray(img_array, img_image_type=image_type, img_image_series_index=serie_index)
            if not sub_array.any():
                continue
            nifti_from_image_array(sub_array, "test/data")
            # --- stack images ------------------------------------------------
            images = flatten(sub_array)
            data, head, meta = stack_images(images)
            mem = log_memory_delta("process_image", "After stack_images", mem)
            del sub_array, images
            
            actual_image_type = head[0].image_type
            logging.info("  --- Invert contrast on %d %s images ---", len(head), image_type_name[actual_image_type])

            # display diagnostic info in the log
            if series is None:
                display_diagnostic(head[0], meta[0])
            
            # --- Normalise to 12-bit range and convert to int16 --------------
            data = normalise(data)
            data = data.astype(np.int16)
            gc.collect()
            mem = log_memory_delta("process_image", "After normalisation", mem)

            # --- Invert contrast ---------------------------------------------
            data = maxVal-data
            data = np.abs(data)
            np.save(debugFolder + "/" + "imgInverted.npy", data)
            mem = log_memory_delta("process_image", "After inversion", mem)

            # --- Add series --------------------------------------------------
            series.add(data, head, meta, 
                process_history = ["PYTHON", "INVERT"], 
                sequence_description = "invertcontrast")
            del data
            gc.collect()
            mem = log_memory_delta("process_image", "After series.add", mem)

    if series is None:
        logging.error("No images found in img_array. Returning empty result.")
        return []
    
    log_memory_delta("process_image", "End", mem)
    logging.info("--- End of invertContrast ---------------------")
    return series.get()
