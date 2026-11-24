import ismrmrd
import numpy as np

from utils.OutputSeries import OutputSeries, ProcessImageResult

def process_image(
        img_array: np.ndarray[ismrmrd.Image],
        configJSON: dict | None,
        metadata: ismrmrd.xsd.ismrmrdHeader | str
        )-> ProcessImageResult:

    series = OutputSeries()
    return series.get()
