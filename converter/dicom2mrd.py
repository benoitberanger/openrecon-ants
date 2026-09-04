"""Converts classic DICOM files into an MRD (.h5) file."""

import argparse
import base64
import ctypes
import logging
import os
import re

import dateutil.parser
import ismrmrd
import numpy as np
import pydicom


# Defaults for input arguments
defaults = {
    'outGroup':       'dataset',
}

# Lookup table between DICOM and MRD image types
imtype_map = {'M': ismrmrd.IMTYPE_MAGNITUDE,
              'P': ismrmrd.IMTYPE_PHASE,
              'R': ismrmrd.IMTYPE_REAL,
              'I': ismrmrd.IMTYPE_IMAG}

# Lookup table between DICOM and Siemens flow directions
venc_dir_map = {'rl'  : 'FLOW_DIR_R_TO_L',
                'lr'  : 'FLOW_DIR_L_TO_R',
                'ap'  : 'FLOW_DIR_A_TO_P',
                'pa'  : 'FLOW_DIR_P_TO_A',
                'fh'  : 'FLOW_DIR_F_TO_H',
                'hf'  : 'FLOW_DIR_H_TO_F',
                'in'  : 'FLOW_DIR_TP_IN',
                'out' : 'FLOW_DIR_TP_OUT'}

def CalcFieldOfView(dset: pydicom.Dataset) -> tuple[float]:
    """
    Compute the field of view (x, y, z) in mm for a DICOM dataset.
 
    Parameters
    ----------
    dset : pydicom.Dataset
        DICOM dataset to read. Must expose ``SOPClassUID`` and the fields
        relevant to that SOP class.
 
    Returns
    -------
    tuple of float
        Field of view in millimetres, as ``(fov_x, fov_y, fov_z)``.
    """
    if dset.SOPClassUID.name == 'Enhanced MR Image Storage':
        try:
            PixelMeasuresSequence = dset.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0]
        except:
            PixelMeasuresSequence = dset.PerFrameFunctionalGroupsSequence[0].PixelMeasuresSequence[0]

            uSliceThickness  = set([float(s.PixelMeasuresSequence[0].SliceThickness)  for s in dset.PerFrameFunctionalGroupsSequence])
            uPixelSpacingRow = set([float(s.PixelMeasuresSequence[0].PixelSpacing[0]) for s in dset.PerFrameFunctionalGroupsSequence])
            uPixelSpacingCol = set([float(s.PixelMeasuresSequence[0].PixelSpacing[1]) for s in dset.PerFrameFunctionalGroupsSequence])

            if (len(uSliceThickness) > 1) or (len(uPixelSpacingRow) > 1) or (len(uPixelSpacingCol) > 1):
                logging.warning('Enhanced DICOM has frames with different PixelSpacing or SliceThickness -- only using information from first frame for MRD header')

        return (      PixelMeasuresSequence[0].PixelSpacing[1]*dset.Columns,
                      PixelMeasuresSequence[0].PixelSpacing[0]*dset.Rows,
                float(PixelMeasuresSequence[0].SliceThickness))

    elif dset.SOPClassUID.name == 'MR Image Storage':
        return (      dset.PixelSpacing[1]*dset.Columns,
                      dset.PixelSpacing[0]*dset.Rows,
                float(dset.SliceThickness))
    elif dset.SOPClassUID.name == 'MR Spectroscopy Storage':
        return (dset.VolumeLocalizationSequence[0].SlabThickness,
                dset.VolumeLocalizationSequence[1].SlabThickness,
                dset.VolumeLocalizationSequence[2].SlabThickness)


def set_study_information(dset: pydicom.Dataset, mrdHead: ismrmrd.xsd.ismrmrdHeader):
    """
    Populate ``mrdHead.studyInformation`` from a DICOM dataset.
 
    Parameters
    ----------
    dset : pydicom.Dataset
        Source DICOM dataset.
    mrdHead : ismrmrd.xsd.ismrmrdHeader
        MRD header to update. Modified in place; ``studyInformation`` is
        (re)created on it.
    """

    mrdHead.studyInformation = ismrmrd.xsd.studyInformationType()
    try:
        studyDateTime = dateutil.parser.parse(getattr(dset, 'StudyDate', '1970-01-01') + ' ' + getattr(dset, 'StudyTime', ''))
        mrdHead.studyInformation.studyDate = studyDateTime.strftime('%Y-%m-%d')
        mrdHead.studyInformation.studyTime = studyDateTime.strftime('%H:%M:%S')
    except:
        pass

    mrdHead.studyInformation.studyID                = getattr(dset, 'StudyID',                None)
    mrdHead.studyInformation.accessionNumber        = getattr(dset, 'AccessionNumber',        None)
    # mrdHead.studyInformation.referringPhysicianName = getattr(dset, 'ReferringPhysicianName', None)
    mrdHead.studyInformation.studyDescription       = getattr(dset, 'StudyDescription',       None)
    mrdHead.studyInformation.studyInstanceUID       = getattr(dset, 'StudyInstanceUID',       None)
    mrdHead.studyInformation.bodyPartExamined       = getattr(dset, 'BodyPartExamined',       None)


def set_measurment_information(dset: pydicom.Dataset, mrdHead: ismrmrd.xsd.ismrmrdHeader):
    """
    Populate ``mrdHead.measurementInformation`` from a DICOM dataset.
 
    Parameters
    ----------
    dset : pydicom.Dataset
        Source DICOM dataset.
    mrdHead : ismrmrd.xsd.ismrmrdHeader
        MRD header to update. Modified in place; ``measurementInformation``
        is (re)created on it.
    """

    mrdHead.measurementInformation                             = ismrmrd.xsd.measurementInformationType()
    mrdHead.measurementInformation.measurementID               = getattr(dset, 'SeriesInstanceUID',   None)
    mrdHead.measurementInformation.patientPosition             = getattr(dset, 'PatientPosition',     None)
    mrdHead.measurementInformation.protocolName                = getattr(dset, 'SeriesDescription',   None)
    mrdHead.measurementInformation.frameOfReferenceUID         = getattr(dset, 'FrameOfReferenceUID', None)


def set_acquisition_system_information(dset: pydicom.Dataset, mrdHead: ismrmrd.xsd.ismrmrdHeader):
    """
    Populate ``mrdHead.acquisitionSystemInformation`` from a DICOM dataset.
 
    Parameters
    ----------
    dset : pydicom.Dataset
        Source DICOM dataset.
    mrdHead : ismrmrd.xsd.ismrmrdHeader
        MRD header to update. Modified in place;
        ``acquisitionSystemInformation`` is (re)created on it.
    """

    mrdHead.acquisitionSystemInformation                       = ismrmrd.xsd.acquisitionSystemInformationType()
    mrdHead.acquisitionSystemInformation.systemVendor          = getattr(dset, 'Manufacturer',          None)
    mrdHead.acquisitionSystemInformation.systemModel           = getattr(dset, 'ManufacturerModelName', None)
    mrdHead.acquisitionSystemInformation.systemFieldStrength_T = float(getattr(dset, 'MagneticFieldStrength', '0'))
    mrdHead.acquisitionSystemInformation.institutionName       = getattr(dset, 'InstitutionName',       None)
    mrdHead.acquisitionSystemInformation.stationName           = getattr(dset, 'StationName',           None)


def set_experimental_conditions(dset: pydicom.Dataset, mrdHead: ismrmrd.xsd.ismrmrdHeader):
    """
    Populate ``mrdHead.experimentalConditions`` from a DICOM dataset.
 
    Parameters
    ----------
    dset : pydicom.Dataset
        Source DICOM dataset.
    mrdHead : ismrmrd.xsd.ismrmrdHeader
        MRD header to update. Modified in place; ``experimentalConditions``
        is (re)created on it.
    """
    mrdHead.experimentalConditions                             = ismrmrd.xsd.experimentalConditionsType()
    if hasattr(dset, 'TransmitterFrequency'):
        mrdHead.experimentalConditions.H1resonanceFrequency_Hz = int(getattr(dset, 'TransmitterFrequency')*1e6)
    elif hasattr(dset, 'ImagingFrequency'):
        mrdHead.experimentalConditions.H1resonanceFrequency_Hz = int(getattr(dset, 'ImagingFrequency')*1e6)
    else:
        mrdHead.experimentalConditions.H1resonanceFrequency_Hz = int(getattr(dset, 'MagneticFieldStrength')*4258e4)


def set_encoding_type(dset: pydicom.Dataset, mrdHead: ismrmrd.xsd.ismrmrdHeader):
    """
    Build and append a Cartesian ``encodingType`` entry to ``mrdHead.encoding``.
 
    Parameters
    ----------
    dset : pydicom.Dataset
        Source DICOM dataset.
    mrdHead : ismrmrd.xsd.ismrmrdHeader
        MRD header to update. A new encodingType is appended to
        ``mrdHead.encoding``.
    """
    enc = ismrmrd.xsd.encodingType()
    enc.trajectory          = ismrmrd.xsd.trajectoryType('cartesian')

    encSpace                = ismrmrd.xsd.encodingSpaceType()
    encSpace.matrixSize     = ismrmrd.xsd.matrixSizeType()
    encSpace.matrixSize.x   = dset.Columns
    encSpace.matrixSize.y   = dset.Rows
    encSpace.matrixSize.z   = 1
    encSpace.fieldOfView_mm = ismrmrd.xsd.fieldOfViewMm(*CalcFieldOfView(dset))

    enc.encodedSpace = encSpace
    enc.reconSpace   = encSpace

    enc.encodingLimits                     = ismrmrd.xsd.encodingLimitsType()
    enc.parallelImaging                    = ismrmrd.xsd.parallelImagingType()
    enc.parallelImaging.accelerationFactor = ismrmrd.xsd.accelerationFactorType()
    if hasattr(dset, 'SharedFunctionalGroupsSequence'):
        if dset.SharedFunctionalGroupsSequence[0].MRModifierSequence[0].ParallelAcquisition == 'NO':
            enc.parallelImaging.accelerationFactor.kspace_encoding_step_1 = 1
            enc.parallelImaging.accelerationFactor.kspace_encoding_step_2 = 1
        else:
            enc.parallelImaging.accelerationFactor.kspace_encoding_step_1 = dset.SharedFunctionalGroupsSequence[0].MRModifierSequence[0].ParallelReductionFactorInPlane
            enc.parallelImaging.accelerationFactor.kspace_encoding_step_2 = dset.SharedFunctionalGroupsSequence[0].MRModifierSequence[0].ParallelReductionFactorOutOfPlane
    else:
        enc.parallelImaging.accelerationFactor.kspace_encoding_step_1 = 1
        enc.parallelImaging.accelerationFactor.kspace_encoding_step_2 = 1

    mrdHead.encoding.append(enc)


def set_sequence_parameters(dset: pydicom.Dataset, mrdHead: ismrmrd.xsd.ismrmrdHeader):
    """
    Populate ``mrdHead.sequenceParameters`` (TR, flip angle, TE) from a DICOM dataset.
 
    Parameters
    ----------
    dset : pydicom.Dataset
        Source DICOM dataset.
    mrdHead : ismrmrd.xsd.ismrmrdHeader
        MRD header to update. Modified in place; ``sequenceParameters``
        is (re)created on it.
    """
    mrdHead.sequenceParameters               = ismrmrd.xsd.sequenceParametersType()
    if hasattr(dset, 'SharedFunctionalGroupsSequence'):
        mrdHead.sequenceParameters.TR            = float(dset.SharedFunctionalGroupsSequence[0].MRTimingAndRelatedParametersSequence[0].RepetitionTime)
        mrdHead.sequenceParameters.flipAngle_deg = float(dset.SharedFunctionalGroupsSequence[0].MRTimingAndRelatedParametersSequence[0].FlipAngle)
        mrdHead.sequenceParameters.TE            =       dset.SharedFunctionalGroupsSequence[0].MREchoSequence[0].EffectiveEchoTime
    else:
        mrdHead.sequenceParameters.TR            = float(dset.RepetitionTime)
        mrdHead.sequenceParameters.flipAngle_deg = float(dset.FlipAngle)
        mrdHead.sequenceParameters.TE            = float(dset.EchoTime)


def water_suppresion(dset: pydicom.Dataset, userParameters: ismrmrd.xsd.userParametersType):
    """
    Detect a Siemens water-saturation flag and record it as a user parameter.
 
    Parameters
    ----------
    dset : pydicom.Dataset
        Source DICOM dataset.
    userParameters : ismrmrd.xsd.userParametersType
        MRD user parameters container. Modified in place by appending a
        userParameterString entry when water saturation is detected.
    """
    try:
        if hasattr(dset, 'SharedFunctionalGroupsSequence'):
            MeasurementOptions = dset.SharedFunctionalGroupsSequence[0][0x002110FE][0][0x0021105C].value
        else:
            MeasurementOptions = dset[0x0021105C].value

        if isinstance(MeasurementOptions, str):
            if MeasurementOptions == 'WS':
                userParameterString = ismrmrd.xsd.userParameterStringType('FatWaterContrast', 'WATER_SATURATION')
                userParameters.userParameterString.append(userParameterString)
        else:
            if 'WS' in list(MeasurementOptions):
                userParameterString = ismrmrd.xsd.userParameterStringType('FatWaterContrast', 'WATER_SATURATION')
                userParameters.userParameterString.append(userParameterString)
    except (AttributeError, IndexError, KeyError, TypeError) as e:
        logging.debug(f"No water-suppression flag found: {e}")


def set_spectroscopy_readout_points(dset: pydicom.Dataset, userParameters: ismrmrd.xsd.userParametersType):
    """
    Record the spectroscopy acquisition vector size as a user parameter.
 
    Parameters
    ----------
    dset : pydicom.Dataset
        Source DICOM dataset.
    userParameters : ismrmrd.xsd.userParametersType
        MRD user parameters container. Modified in place by appending a
        userParameterLong entry when SpecVectorSize is found.
    """
    try:
        SpecVectorSize = dset.SharedFunctionalGroupsSequence[0].MRSpectroscopyFOVGeometrySequence[0].SpectroscopyAcquisitionDataColumns
        userParameterLong = ismrmrd.xsd.userParameterLongType('SpecVectorSize', SpecVectorSize)
        userParameters.userParameterLong.append(userParameterLong)
    except (AttributeError, IndexError) as e:
        logging.debug(f"No SpecVectorSize found: {e}")


def set_readout_oversampling(dset: pydicom.Dataset, userParameters: ismrmrd.xsd.userParametersType):
    """
    Record spectroscopy readout oversampling parameters as user parameters.
 
    Parameters
    ----------
    dset : pydicom.Dataset
        Source DICOM dataset.
    userParameters : ismrmrd.xsd.userParametersType
        MRD user parameters container. Modified in place by appending
        userParameterDouble entries for whichever values were found.
    """   
    ReadoutOS = None
    SpectralWidth = None

    # Readout oversampling
    try:
        if hasattr(dset, 'SharedFunctionalGroupsSequence'):
            ReadoutOS = dset.SharedFunctionalGroupsSequence[0][0x002110FE][0][0x00211012].value
        else:
            ReadoutOS = dset[0x00211012].value
        userParameterDouble = ismrmrd.xsd.userParameterDoubleType('ReadoutOS', ReadoutOS)
        userParameters.userParameterDouble.append(userParameterDouble)
    except (AttributeError, IndexError, KeyError, TypeError) as e:
        logging.debug(f"No ReadoutOS found: {e}")

    # Spectral Width (Hz)
    try:
        SpectralWidth = dset.SpectralWidth
        userParameterDouble = ismrmrd.xsd.userParameterDoubleType('SpectralWidth', SpectralWidth)
        userParameters.userParameterDouble.append(userParameterDouble)
    except AttributeError as e:
        logging.debug(f"No SpectralWidth found: {e}")

    # Dwell time (oversampled)
    try:
        DwellTime = 1e6 / SpectralWidth / ReadoutOS
        userParameterDouble = ismrmrd.xsd.userParameterDoubleType('DwellTime_0', DwellTime)
        userParameters.userParameterDouble.append(userParameterDouble)
    except (TypeError, ZeroDivisionError) as e:
        logging.debug(f"Could not compute DwellTime_0 (missing SpectralWidth/ReadoutOS): {e}")


def set_spectroscopy_VOI(dset: pydicom.Dataset, userParameters: ismrmrd.xsd.userParametersType):
    """
    Record spectroscopy volume-of-interest (VOI) dimensions as user parameters.
 
    Parameters
    ----------
    dset : pydicom.Dataset
        Source DICOM dataset.
    userParameters : ismrmrd.xsd.userParametersType
        MRD user parameters container. Modified in place by appending
        up to 3 userParameterDouble entries.
    """
    # Spectroscopy volume of interest dimensions
    try:
        for dim in dset.VolumeLocalizationSequence:
            # Determine if this is x, y, or z
            if all(np.abs(np.cross([0, 0, 1], np.array(dim.SlabOrientation))) < 1e-5):
                name = 'SpecVoiThickness'
            elif all(np.abs(np.cross([1, 0, 0], np.array(dim.SlabOrientation))) < 1e-5):
                name = 'SpecVoiPhaseFOV'
            elif all(np.abs(np.cross([0, 1, 0], np.array(dim.SlabOrientation))) < 1e-5):
                name = 'SpecVoiReadoutFOV'
            else:
                logging.info(f'Could not determine spectroscopy VOI dimension for orientation {dim.SlabOrientation}')
                continue

            userParameterDouble = ismrmrd.xsd.userParameterDoubleType(name, dim.SlabThickness)
            userParameters.userParameterDouble.append(userParameterDouble)
    except AttributeError as e:
        logging.debug(f"No VolumeLocalizationSequence found: {e}")


def CreateMrdHeader(dset: pydicom.Dataset) -> ismrmrd.xsd.ismrmrdHeader:
    """
    Assemble a complete MRD XML header from a single reference DICOM dataset.
 
    Parameters
    ----------
    dset : pydicom.Dataset
        Reference DICOM dataset (typically the first file of the first
        series) used to derive all header fields.
 
    Returns
    -------
    ismrmrd.xsd.ismrmrdHeader
        Fully populated MRD XML header, ready to be serialised with
        ``.toXML()`` and written via ``ismrmrd.Dataset.write_xml_header()``.
    """
    mrdHead = ismrmrd.xsd.ismrmrdHeader()

    set_study_information(dset, mrdHead)
    set_measurment_information(dset, mrdHead)
    set_acquisition_system_information(dset, mrdHead)
    set_experimental_conditions(dset, mrdHead)
    set_encoding_type(dset, mrdHead)
    set_sequence_parameters(dset, mrdHead)

    userParameters = ismrmrd.xsd.userParametersType()

    water_suppresion(dset, userParameters)
    set_spectroscopy_readout_points(dset, userParameters)
    set_readout_oversampling(dset, userParameters)
    set_spectroscopy_VOI(dset, userParameters)

    mrdHead.userParameters = userParameters
    return mrdHead

###############################################################################

def GetDicomFiles(directory: str):
    """
    Recursively yield paths to all DICOM files under a directory.
 
    Parameters
    ----------
    directory : str
        Root directory to scan.
 
    Yields
    ------
    str
        Path to each DICOM file found.
    """
    for entry in os.scandir(directory):
        if entry.is_file() and (entry.path.lower().endswith(".dcm") or entry.path.lower().endswith(".ima")):
            yield entry.path
        elif entry.is_dir():
            yield from GetDicomFiles(entry.path)


def renumber_split_series(dsetsAll: list[pydicom.Dataset]) -> list[pydicom.Dataset]:
    """
    Re-group DICOM series that were split during multi-frame to single-frame conversion.
 
    Parameters
    ----------
    dsetsAll : list of pydicom.Dataset
        All loaded DICOM datasets, potentially spanning multiple series.
 
    Returns
    -------
    list of pydicom.Dataset
        The same list, with ``SeriesNumber`` renumbered in place when a
        split was detected.
    """
    # Group by series number
    uSeriesNum = np.unique([dset.SeriesNumber for dset in dsetsAll])

    # Re-group series that were split during conversion from multi-frame to single-frame DICOMs
    if all(uSeriesNum > 1000):
        for i in range(len(dsetsAll)):
            dsetsAll[i].SeriesNumber = int(np.floor(dsetsAll[i].SeriesNumber / 1000))
    uSeriesNum = np.unique([dset.SeriesNumber for dset in dsetsAll])
    return dsetsAll


def load_dicom_series(folder: str) -> tuple[list[pydicom.Dataset], np.ndarray]:
    """
    Load every DICOM file found under a folder and identify the distinct series.
 
    Parameters
    ----------
    folder : str
        Root directory to scan for DICOM files.
 
    Returns
    -------
    dsetsAll : list of pydicom.Dataset
        All loaded datasets.
    uSeriesNum : np.ndarray
        Sorted array of unique series numbers found across ``dsetsAll``.
    """
    dsetsAll = []
    for entryPath in GetDicomFiles(folder):
        dsetsAll.append(pydicom.dcmread(entryPath))
    dsetsAll = renumber_split_series(dsetsAll)
    uSeriesNum = np.unique([dset.SeriesNumber for dset in dsetsAll])

    logging.info("Found %d unique series from %d files in folder %s" % (len(uSeriesNum), len(dsetsAll), folder))

    return dsetsAll, uSeriesNum


def build_meta(dset: pydicom.Dataset) -> ismrmrd.Meta:
    """
    Build an ismrmrd.Meta object carrying non-MRD DICOM metadata for one image.
 
    Parameters
    ----------
    dset : pydicom.Dataset
        Source DICOM dataset for one image/frame. Mutated in place:
        ``PixelData`` and ``SpectroscopyData`` are deleted from it if
        present.
 
    Returns
    -------
    ismrmrd.Meta
        Populated Meta object, ready to be serialised into
        ``ismrmrd.Image.attribute_string``.
    """

    meta = ismrmrd.Meta()

    try:
        meta['ImageType'] = dset.ImageType
    except:
        pass

    try:
        res  = re.search(r'(?<=_v).*$',     dset.SequenceName)
        venc = re.search(r'^\d+',           res.group(0))
        dir  = re.search(r'(?<=\d)[^\d]*$', res.group(0))

        meta['FlowVelocity']   = float(venc.group(0))
        meta['FlowDirDisplay'] = venc_dir_map[dir.group(0)]
    except (AttributeError, KeyError) as e:
        logging.debug(f"No flow/VENC information found: {e}")

    try:
        meta['ImageComments'] = dset.ImageComments
    except AttributeError as e:
        logging.debug(f"No ImageComments found: {e}")

    meta['SequenceDescription'] = dset.SeriesDescription

    # Remove pixel data from pydicom class before serializing metadata
    if hasattr(dset, 'PixelData'):
        del dset['PixelData']
    if hasattr(dset, 'SpectroscopyData'):
        del dset['SpectroscopyData']

    # Store the complete base64, json-formatted DICOM header so that non-MRD fields can be
    # recapitulated when generating DICOMs from MRD images
    meta['DicomJson'] = base64.b64encode(dset.to_json().encode('utf-8')).decode('utf-8')

    return meta


def build_mrd_image(dset: pydicom.Dataset, serieIndex: int, uSliceLoc: np.ndarray, uTrigTime: np.ndarray) -> ismrmrd.Image | None:
    """
    Convert a single DICOM instance into an ismrmrd.Image.
 
    Parameters
    ----------
    dset : pydicom.Dataset
        Source DICOM dataset.
    serieIndex : int
        MRD ``image_series_index`` to assign to the resulting image.
    uSliceLoc : np.ndarray
        Sorted array of unique SliceLocation values for the series this
        image belongs to, used to compute the ``slice`` index.
    uTrigTime : np.ndarray
        Sorted array of unique TriggerTime values for the series this
        image belongs to, used to compute the ``phase`` index.
 
    Returns
    -------
    ismrmrd.Image or None
        The populated MRD image, or None if ``dset`` exposes neither
        pixel nor spectroscopy data.
    """
    if hasattr(dset, 'pixel_array'):
        # pixel_array data has shape [row col], i.e. [y x].
        mrdImage = ismrmrd.Image.from_array(dset.pixel_array, transpose=False)
    elif hasattr(dset, 'SpectroscopyData'):
        mrdImage = ismrmrd.Image.from_array(np.frombuffer(dset.SpectroscopyData, dtype=np.complex64), transpose=False)
    else:
        logging.error(f'Could not find imaging or spectroscopy data for file {dset.filename}')
        return None
    
    try:
        mrdImage.image_type                = imtype_map[dset.ImageType[2]]
    except:
        logging.info("Unsupported ImageType %s -- defaulting to IMTYPE_MAGNITUDE" % dset.ImageType[2])
        mrdImage.image_type                = ismrmrd.IMTYPE_MAGNITUDE

    if hasattr(dset, 'PerFrameFunctionalGroupsSequence'):
        ImagePositionPatient    = dset.PerFrameFunctionalGroupsSequence[0].PlanePositionSequence[0].ImagePositionPatient
        ImageOrientationPatient = dset.PerFrameFunctionalGroupsSequence[0].PlaneOrientationSequence[0].ImageOrientationPatient
        AcquisitionTime         = dset.PerFrameFunctionalGroupsSequence[0].FrameContentSequence[0].FrameAcquisitionDateTime[8:]  # Strip out date
        try:
            TriggerTime = dset.PerFrameFunctionalGroupsSequence[0].CardiacSynchronizationSequence[0].NominalCardiacTriggerDelayTime
        except:
            TriggerTime = None
    else:
        ImagePositionPatient    = dset.ImagePositionPatient
        ImageOrientationPatient = dset.ImageOrientationPatient
        AcquisitionTime         = dset.AcquisitionTime
        try:
            TriggerTime = float(dset.TriggerTime)
        except:
            TriggerTime = None

    mrdImage.field_of_view            = CalcFieldOfView(dset)
    mrdImage.position                 = tuple(np.stack(ImagePositionPatient))
    mrdImage.read_dir                 = tuple(np.stack(ImageOrientationPatient[0:3]))
    mrdImage.phase_dir                = tuple(np.stack(ImageOrientationPatient[3:7]))
    mrdImage.slice_dir                = tuple(np.cross(np.stack(ImageOrientationPatient[0:3]), np.stack(ImageOrientationPatient[3:7])))
    mrdImage.acquisition_time_stamp   = round((int(AcquisitionTime[0:2])*3600 + int(AcquisitionTime[2:4])*60 + int(AcquisitionTime[4:6]) + float(AcquisitionTime[6:]))*1000/2.5)
    if TriggerTime:
        mrdImage.physiology_time_stamp[0] = round(int(TriggerTime/2.5))

    try:
        ImaAbsTablePosition = dset.get_private_item(0x0019, 0x13, 'SIEMENS MR HEADER').value
        mrdImage.patient_table_position = (ctypes.c_float(ImaAbsTablePosition[0]), ctypes.c_float(ImaAbsTablePosition[1]), ctypes.c_float(ImaAbsTablePosition[2]))
    except (AttributeError, KeyError, TypeError) as e:
        logging.debug(f"No private table-position tag found: {e}")

    mrdImage.image_series_index     = serieIndex
    mrdImage.image_index            = dset.get('InstanceNumber', 0)
    mrdImage.slice                  = uSliceLoc.tolist().index(getattr(dset, 'SliceLocation', 0))
    try:
        mrdImage.phase                  = uTrigTime.tolist().index(dset.TriggerTime)
    except (AttributeError, ValueError) as e:
        logging.debug(f"Could not determine cardiac phase index: {e}")
    

    # In dicom the echo indice began at 1, but it MRD they seems to began at 0
    mrdImage.contrast              = dset.EchoNumbers - 1
    # try:
    #     mrdImage.repetition            = tmpDset.TemporalPositionIdentifier
    # except:
    #     logging.debug()

    # try:
    #     mrdImage.average           = tmpDset.NumberOfAverages
    # except:
    #     logging.debug()
    
    mrdImage.attribute_string = build_meta(dset).serialize()

    return mrdImage


def build_series_images(dsets: pydicom.Dataset, seriesIndex: int) -> list[ismrmrd.Image]:
    """
    Convert every DICOM instance of one series into a list of ismrmrd.Image.
 
    Parameters
    ----------
    dsets : list of pydicom.Dataset
        All DICOM datasets belonging to a single series (same
        SeriesNumber).
    seriesIndex : int
        MRD ``image_series_index`` to assign to every resulting image.
 
    Returns
    -------
    list of ismrmrd.Image
        One image per convertible input dataset, in InstanceNumber
        order.
    """
    dsets = sorted(dsets, key=lambda d: d.InstanceNumber)

    # Build a list of unique SliceLocation and TriggerTimes, as the MRD
    # slice and phase counters index into these
    try:
        uSliceLoc = np.unique([dset.SliceLocation for dset in dsets])
        if dsets[0].SliceLocation != uSliceLoc[0]:
            uSliceLoc = uSliceLoc[::-1]
    except AttributeError as e:
        logging.debug(f"No SliceLocation found, defaulting to a single slice: {e}")
        uSliceLoc = np.zeros(1)

    try:
        # This field may not exist for non-gated sequences
        uTrigTime = np.unique([dset.TriggerTime for dset in dsets])
        if dsets[0].TriggerTime != uTrigTime[0]:
            uTrigTime = uTrigTime[::-1]
    except AttributeError as e:
        logging.debug(f"No TriggerTime found (likely a non-gated sequence): {e}")
        uTrigTime = np.zeros_like(uSliceLoc)

    logging.info("Series %d has %d images with %d slices and %d phases" % (dsets[0].SeriesNumber, len(dsets), len(uSliceLoc), len(uTrigTime)))

    images = []
    for dset in dsets:
        mrd_img = build_mrd_image(dset, seriesIndex, uSliceLoc, uTrigTime)
        if mrd_img is not None:
            images.append(mrd_img)
 
    return images


### MRD Writing ###############################################################

def write_mrd_dataset(outFile: str, outGroup: str, mrdHead: ismrmrd.xsd.ismrmrdHeader, imgAll: list):
    """
    Write an MRD header and a list of image series to a new HDF5 file.
 
    Parameters
    ----------
    outFile : str
        Path to the output MRD (.h5) file. Removed and recreated if it
        already exists.
    outGroup : str
        Top-level HDF5 group name to create the dataset under.
    mrdHead : ismrmrd.xsd.ismrmrdHeader
        MRD header to write via ``write_xml_header()``.
    imgAll : list of list of ismrmrd.Image or None
        Images grouped by series. None entries are skipped.
    """

    # Create an MRD file
    logging.info("Creating MRD file %s with group %s" % (outFile, outGroup))
    if os.path.exists(outFile):
        os.remove(outFile)             # Delete the outfile, if it already exist because ismrmrd.Dataset is in append mode

    mrdDset = ismrmrd.Dataset(outFile, outGroup)
    mrdDset._file.require_group(outGroup)

    # Write MRD Header
    mrdDset.write_xml_header(bytes(mrdHead.toXML(), 'utf-8'))

    # Write all images
    for iSer in range(len(imgAll)):
        for iImg in range(len(imgAll[iSer])):
            if imgAll[iSer][iImg] is not None:
                mrdDset.append_image("image_%d" % imgAll[iSer][iImg].image_series_index, imgAll[iSer][iImg])

    mrdDset.close()


### CLI #######################################################################

def main(args):
    dsetsAll, uSeriesNum = load_dicom_series(args.folder)

    logging.info("Creating MRD XML header from file %s" % dsetsAll[0].filename)
    mrdHead = CreateMrdHeader(dsetsAll[0])
    logging.info(mrdHead.toXML())

    imgAll = [None]*len(uSeriesNum)

    for iSer in range(len(uSeriesNum)):
        dsets = [dset for dset in dsetsAll if dset.SeriesNumber == uSeriesNum[iSer]]
        imgAll[iSer] = build_series_images(dsets, uSeriesNum[iSer])

    write_mrd_dataset(args.outFile, args.outGroup, mrdHead, imgAll)


if __name__ == '__main__':
    """Basic conversion of a folder of DICOM files to MRD .h5 format"""

    parser = argparse.ArgumentParser(description='Convert DICOMs to MRD file',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('folder',                                   help='Input folder of DICOMs')
    parser.add_argument('-o', '--outFile',                          help='Output MRD file')
    parser.add_argument('-g', '--outGroup',                         help='Group name in output MRD file')
    parser.add_argument('-v', '--verbose',  action='store_true',    help='Verbose output.')

    parser.set_defaults(**defaults)

    args = parser.parse_args()

    if args.outFile is None:
        args.outFile = os.path.basename(args.folder) + '.h5'
    
    if args.verbose:
        logLevel = logging.DEBUG
    else:
        logLevel = logging.INFO

    # setup logging
    logging.basicConfig(
        level=logLevel,
        format=f"%(levelname)8s: %(message)s"
    )

    main(args)
    