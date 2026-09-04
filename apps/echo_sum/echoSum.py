import gc
import logging
import os

import ismrmrd
import numpy as np

from utils.OutputSeries import OutputSeries, ProcessImageResult
from utils.img_array import flatten, get_subarray, mrd_indexes, stack_images
from utils.memory import log_memory, log_memory_delta, timeit
from utils.utils import check_OR_arguments, display_diagnostic, normalise


# Folder for debug output files
debugFolder = "/tmp/share/debug"

@timeit
def process_image(img_array: np.ndarray[ismrmrd.Image], configJSON: dict | None, metadata) -> ProcessImageResult:
    """
    Combine multi-echo magnitude images into a single image per slice.

    Two combination modes are supported, selected via the 'EchoSumConfig'
    JSON parameter:

    - ``'SimpleSum'`` (default) : direct sum of magnitude echoes
    - ``'SoS'`` : sum of squares followed by a square root

    The result is normalised to 12-bit range and transposed to the
    [y, x, z, cha, img] layout expected by send_volume_as_slices().

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

    logging.info(f'-----------------------------------------------')
    logging.info(f'     Echos summation called')
    logging.info(f'-----------------------------------------------')
    
    mem = log_memory("process_image", "Beginning")

    # --- OR Parameters ---------------------------------------------------
    sum_config = check_OR_arguments(configJSON, 
                                    arg_name='EchoSumConfig', 
                                    arg_type=str, 
                                    arg_default='SimpleSum')
    logging.info(f"Echos summation config: {sum_config}")
    
    # --- Dimensions ------------------------------------------------------
    # Get the number of contrasts
    n_contrasts = img_array.shape[mrd_indexes.contrast]
    logging.info("Summing %d echoes (contrasts)", n_contrasts)

    series = OutputSeries()
    for serie_index in range(0, img_array.shape[mrd_indexes.image_series_index]):    
        # --- Stack first echo (reference for head and meta) ------------------
        # Head and meta are taken from contrast 0, magnitude.
        # Initialise data_sum with the first echo
        # SoS: accumulate squared values, then take sqrt at the end
        # SimpleSum: accumulate raw values directly
        first_echo_array = get_subarray(img_array, 
                                         img_contrast=0, 
                                         img_image_type=ismrmrd.IMTYPE_MAGNITUDE,
                                         img_image_series_index=serie_index)
        if not first_echo_array.any():
            continue
        first_echo_images = flatten(first_echo_array)
        data_sum, head, meta = stack_images(first_echo_images) #[img, cha, z, y, x], head, meta
        del first_echo_array, first_echo_images

        # display diagnostic info in the log
        display_diagnostic(head[0], meta[0])
        
        if (sum_config == 'SoS'):
            np.square(data_sum, out=data_sum)
        mem = log_memory_delta("process_image", "After stacking echo 0", mem)

        # --- Sum with remaining echoes ---------------------------------------
        # SoS: accumulate squared values, then take sqrt at the end
        # SimpleSum: accumulate raw values directly
        for co in range(1, n_contrasts):
            echo_array = get_subarray(img_array, 
                                     img_contrast=co, 
                                     img_image_type=ismrmrd.IMTYPE_MAGNITUDE,
                                     img_image_series_index=serie_index)
            if not echo_array.any():
                continue
            echo_images = flatten(echo_array)
            data_co, _, _ = stack_images(echo_images)
            if (sum_config == 'SoS'):
                np.square(data_co, out=data_co)
            data_sum += data_co
            del echo_array, echo_images, data_co
            gc.collect()
            mem = log_memory_delta("process_image", f"After adding echo {co}", mem)

        gc.collect()

        # SoS finalisation: square root of the accumulated squared sum
        data_sum /= n_contrasts
        if (sum_config == 'SoS'):
            np.sqrt(data_sum, out=data_sum)

        # --- Normalisation to 12-bit range and convert to int16 --------------
        data_sum = normalise(data_sum)
        data_sum = data_sum.astype(np.int16)
        mem = log_memory_delta("process_image", "After normalisation", mem)

        np.save(debugFolder + "/imgMagnitudeSum.npy", data_sum)

        # --- Update metadata -------------------------------------------------
        if sum_config == 'SoS':
            series.add(data_sum, head, meta, 
                    process_history = ["PYTHON", "SOS"], 
                    sequence_description = "SoS")
        else:
            series.add(data_sum, head, meta, 
                    process_history = ["PYTHON", "ECHO_SUM_SIMPLE"], 
                    sequence_description = "EchoSumSimple")
    
    return series.get()
