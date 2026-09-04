import base64
import json

import ismrmrd
import numpy as np
import pydicom
import pytest

import converter.dicom2mrd as dicom2mrd


def make_dataset(**kwargs):
    ds = pydicom.Dataset()
    for k, v in kwargs.items():
        setattr(ds, k, v)
    return ds


def make_bare_header():
    return ismrmrd.xsd.ismrmrdHeader(
        experimentalConditions=ismrmrd.xsd.experimentalConditionsType(H1resonanceFrequency_Hz=0)
    )


def make_pixel_dataset(rows=2, cols=3, pixel_value=0, dtype=np.int16, **extra):
    ds = pydicom.Dataset()
    ds.file_meta = pydicom.dataset.FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    ds.Rows = rows
    ds.Columns = cols
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = np.full((rows, cols), pixel_value, dtype=dtype).tobytes()
    for k, v in extra.items():
        setattr(ds, k, v)
    return ds


# ---------------------------------------------------------------------------
# CalcFieldOfView()
# ---------------------------------------------------------------------------
class TestCalcFieldOfView:

    def test_mr_image_storage(self):
        ds = make_dataset(
            SOPClassUID=pydicom.uid.MRImageStorage,
            PixelSpacing=[2.0, 3.0],  # [row_spacing, col_spacing]
            Columns=10, Rows=5,
            SliceThickness=6.0,
        )
        fov = dicom2mrd.CalcFieldOfView(ds)
        assert fov == pytest.approx((30.0, 10.0, 6.0))  # (col_spacing*Columns, row_spacing*Rows, thickness)

    def test_spectroscopy_storage(self):
        ds = make_dataset(
            SOPClassUID=pydicom.uid.MRSpectroscopyStorage,
            VolumeLocalizationSequence=[
                make_dataset(SlabThickness=10.0),
                make_dataset(SlabThickness=20.0),
                make_dataset(SlabThickness=30.0),
            ],
        )
        fov = dicom2mrd.CalcFieldOfView(ds)
        assert fov == (10.0, 20.0, 30.0)


# ---------------------------------------------------------------------------
# set_study_information()
# ---------------------------------------------------------------------------
class TestSetStudyInformation:

    def test_valid_date_and_time_parsed(self):
        ds = make_dataset(StudyDate="20260804", StudyTime="143000")
        head = make_bare_header()
        dicom2mrd.set_study_information(ds, head)
        assert head.studyInformation.studyDate == "2026-08-04"
        assert head.studyInformation.studyTime == "14:30:00"

    def test_other_fields_mapped(self):
        ds = make_dataset(
            StudyID="STUDY1", AccessionNumber="ACC1",
            StudyDescription="Brain MRI", StudyInstanceUID="1.2.3",
            BodyPartExamined="BRAIN",
        )
        head = make_bare_header()
        dicom2mrd.set_study_information(ds, head)
        assert head.studyInformation.studyID == "STUDY1"
        assert head.studyInformation.accessionNumber == "ACC1"
        assert head.studyInformation.studyDescription == "Brain MRI"
        assert head.studyInformation.studyInstanceUID == "1.2.3"
        assert head.studyInformation.bodyPartExamined == "BRAIN"

    def test_missing_fields_default_to_none_without_crashing(self):
        ds = make_dataset()  # empty dataset
        head = make_bare_header()
        dicom2mrd.set_study_information(ds, head)
        assert head.studyInformation.studyID is None


# ---------------------------------------------------------------------------
# set_measurment_information()
# ---------------------------------------------------------------------------
class TestSetMeasurementInformation:

    def test_fields_mapped(self):
        ds = make_dataset(
            SeriesInstanceUID="1.2.3.4", PatientPosition="HFS",
            SeriesDescription="Test", FrameOfReferenceUID="5.6.7.8",
        )
        head = make_bare_header()
        dicom2mrd.set_measurment_information(ds, head)
        assert head.measurementInformation.measurementID == "1.2.3.4"
        assert head.measurementInformation.patientPosition == "HFS"
        assert head.measurementInformation.protocolName == "Test"
        assert head.measurementInformation.frameOfReferenceUID == "5.6.7.8"


# ---------------------------------------------------------------------------
# set_acquisition_system_information()
# ---------------------------------------------------------------------------
class TestSetAcquisitionSystemInformation:

    def test_fields_mapped_and_field_strength_cast_to_float(self):
        ds = make_dataset(
            Manufacturer="Siemens", ManufacturerModelName="Prisma",
            MagneticFieldStrength="3.0", InstitutionName="ICM", StationName="MRI1",
        )
        head = make_bare_header()
        dicom2mrd.set_acquisition_system_information(ds, head)
        assert head.acquisitionSystemInformation.systemVendor == "Siemens"
        assert head.acquisitionSystemInformation.systemModel == "Prisma"
        assert head.acquisitionSystemInformation.systemFieldStrength_T == pytest.approx(3.0)
        assert head.acquisitionSystemInformation.institutionName == "ICM"
        assert head.acquisitionSystemInformation.stationName == "MRI1"

    def test_default_field_strength_when_absent(self):
        ds = make_dataset()
        head = make_bare_header()
        dicom2mrd.set_acquisition_system_information(ds, head)
        assert head.acquisitionSystemInformation.systemFieldStrength_T == 0.0


# ---------------------------------------------------------------------------
# set_experimental_conditions()
# ---------------------------------------------------------------------------
class TestSetExperimentalConditions:

    def test_uses_transmitter_frequency_when_present(self):
        ds = make_dataset(TransmitterFrequency=127.74, ImagingFrequency=999.0, MagneticFieldStrength=3.0)
        head = make_bare_header()
        dicom2mrd.set_experimental_conditions(ds, head)
        assert head.experimentalConditions.H1resonanceFrequency_Hz == int(127.74 * 1e6)

    def test_uses_imaging_frequency_when_transmitter_absent(self):
        ds = make_dataset(ImagingFrequency=63.87, MagneticFieldStrength=1.5)
        head = make_bare_header()
        dicom2mrd.set_experimental_conditions(ds, head)
        assert head.experimentalConditions.H1resonanceFrequency_Hz == int(63.87 * 1e6)

    def test_falls_back_to_field_strength_when_both_absent(self):
        ds = make_dataset(MagneticFieldStrength=3.0)
        head = make_bare_header()
        dicom2mrd.set_experimental_conditions(ds, head)
        assert head.experimentalConditions.H1resonanceFrequency_Hz == int(3.0 * 4258e4)


# ---------------------------------------------------------------------------
# set_encoding_type()
# ---------------------------------------------------------------------------
class TestSetEncodingType:

    def test_matrix_and_fov_populated(self):
        ds = make_dataset(
            SOPClassUID=pydicom.uid.MRImageStorage,
            Columns=64, Rows=64, PixelSpacing=[2.0, 2.0], SliceThickness=5.0,
        )
        head = make_bare_header()
        dicom2mrd.set_encoding_type(ds, head)
        enc = head.encoding[0]
        assert enc.encodedSpace.matrixSize.x == 64
        assert enc.encodedSpace.matrixSize.y == 64
        assert enc.encodedSpace.matrixSize.z == 1
        assert enc.encodedSpace.fieldOfView_mm.x == pytest.approx(128.0)

    def test_parallel_imaging_defaults_without_shared_functional_groups(self):
        ds = make_dataset(SOPClassUID=pydicom.uid.MRImageStorage, Columns=64, Rows=64,
                           PixelSpacing=[2.0, 2.0], SliceThickness=5.0)
        head = make_bare_header()
        dicom2mrd.set_encoding_type(ds, head)
        acc = head.encoding[0].parallelImaging.accelerationFactor
        assert acc.kspace_encoding_step_1 == 1
        assert acc.kspace_encoding_step_2 == 1

    def test_parallel_imaging_no_acceleration_flag(self):
        modifier = make_dataset(ParallelAcquisition='NO')
        ds = make_dataset(
            SOPClassUID=pydicom.uid.MRImageStorage, Columns=64, Rows=64,
            PixelSpacing=[2.0, 2.0], SliceThickness=5.0,
            SharedFunctionalGroupsSequence=[make_dataset(MRModifierSequence=[modifier])],
        )
        head = make_bare_header()
        dicom2mrd.set_encoding_type(ds, head)
        acc = head.encoding[0].parallelImaging.accelerationFactor
        assert acc.kspace_encoding_step_1 == 1
        assert acc.kspace_encoding_step_2 == 1

    def test_parallel_imaging_with_acceleration_factors(self):
        modifier = make_dataset(
            ParallelAcquisition='YES',
            ParallelReductionFactorInPlane=2,
            ParallelReductionFactorOutOfPlane=1,
        )
        ds = make_dataset(
            SOPClassUID=pydicom.uid.MRImageStorage, Columns=64, Rows=64,
            PixelSpacing=[2.0, 2.0], SliceThickness=5.0,
            SharedFunctionalGroupsSequence=[make_dataset(MRModifierSequence=[modifier])],
        )
        head = make_bare_header()
        dicom2mrd.set_encoding_type(ds, head)
        acc = head.encoding[0].parallelImaging.accelerationFactor
        assert acc.kspace_encoding_step_1 == 2
        assert acc.kspace_encoding_step_2 == 1


# ---------------------------------------------------------------------------
# set_sequence_parameters()
# ---------------------------------------------------------------------------
class TestSetSequenceParameters:

    def test_without_shared_functional_groups(self):
        ds = make_dataset(RepetitionTime="2000.0", FlipAngle="90.0", EchoTime="35.0")
        head = make_bare_header()
        dicom2mrd.set_sequence_parameters(ds, head)
        assert head.sequenceParameters.TR == pytest.approx(2000.0)
        assert head.sequenceParameters.flipAngle_deg == pytest.approx(90.0)
        assert head.sequenceParameters.TE == pytest.approx(35.0)

    def test_with_shared_functional_groups(self):
        timing = make_dataset(RepetitionTime="1500.0", FlipAngle="70.0")
        echo = make_dataset(EffectiveEchoTime=28.0)
        ds = make_dataset(SharedFunctionalGroupsSequence=[
            make_dataset(MRTimingAndRelatedParametersSequence=[timing], MREchoSequence=[echo])
        ])
        head = make_bare_header()
        dicom2mrd.set_sequence_parameters(ds, head)
        assert head.sequenceParameters.TR == pytest.approx(1500.0)
        assert head.sequenceParameters.flipAngle_deg == pytest.approx(70.0)
        assert head.sequenceParameters.TE == pytest.approx(28.0)


# ---------------------------------------------------------------------------
# renumber_split_series()
# ---------------------------------------------------------------------------
class TestRenumberSplitSeries:

    def test_series_all_above_1000_get_divided(self):
        dsets = [make_dataset(SeriesNumber=1001), make_dataset(SeriesNumber=1002)]
        result = dicom2mrd.renumber_split_series(dsets)
        assert [d.SeriesNumber for d in result] == [1, 1]

    def test_normal_series_numbers_untouched(self):
        dsets = [make_dataset(SeriesNumber=1), make_dataset(SeriesNumber=2)]
        result = dicom2mrd.renumber_split_series(dsets)
        assert [d.SeriesNumber for d in result] == [1, 2]

    def test_mixed_series_numbers_untouched(self):
        dsets = [make_dataset(SeriesNumber=5), make_dataset(SeriesNumber=1002)]
        result = dicom2mrd.renumber_split_series(dsets)
        assert [d.SeriesNumber for d in result] == [5, 1002]


# ---------------------------------------------------------------------------
# CreateMrdHeader()
# ---------------------------------------------------------------------------
class TestCreateMrdHeader:

    def test_assembles_full_header_without_error(self):
        ds = make_dataset(
            SOPClassUID=pydicom.uid.MRImageStorage,
            StudyDate="20260804", StudyTime="143000",
            SeriesInstanceUID="1.2.3", PatientPosition="HFS", SeriesDescription="T1",
            Manufacturer="Siemens", MagneticFieldStrength="3.0",
            TransmitterFrequency=127.74,
            Columns=64, Rows=64, PixelSpacing=[2.0, 2.0], SliceThickness=5.0,
            RepetitionTime="2000.0", FlipAngle="90.0", EchoTime="35.0",
        )
        head = dicom2mrd.CreateMrdHeader(ds)

        assert head.studyInformation is not None
        assert head.measurementInformation.protocolName == "T1"
        assert head.acquisitionSystemInformation.systemVendor == "Siemens"
        assert head.experimentalConditions.H1resonanceFrequency_Hz == int(127.74 * 1e6)
        assert len(head.encoding) == 1
        assert head.sequenceParameters.TR == pytest.approx(2000.0)
        assert head.userParameters is not None


# ---------------------------------------------------------------------------
# build_meta()
# ---------------------------------------------------------------------------
class TestBuildMeta:

    def test_image_type_included(self):
        ds = make_dataset(ImageType=["ORIGINAL", "PRIMARY", "M"], SeriesDescription="T1")
        meta = dicom2mrd.build_meta(ds)
        assert list(meta["ImageType"]) == ["ORIGINAL", "PRIMARY", "M"]

    def test_sequence_description_from_series_description(self):
        ds = make_dataset(SeriesDescription="Test")
        meta = dicom2mrd.build_meta(ds)
        assert meta["SequenceDescription"] == "Test"

    def test_venc_regex_parses_velocity_and_direction(self):
        ds = make_dataset(SeriesDescription="flow", SequenceName="fl3d1_v150in")
        meta = dicom2mrd.build_meta(ds)
        assert meta["FlowVelocity"] == pytest.approx(150.0)
        assert meta["FlowDirDisplay"] == "FLOW_DIR_TP_IN"

    def test_missing_sequence_name_skips_venc_fields(self):
        ds = make_dataset(SeriesDescription="Test")
        meta = dicom2mrd.build_meta(ds)
        assert meta.get("FlowVelocity") is None

    def test_image_comments_included_when_present(self):
        ds = make_dataset(SeriesDescription="Test", ImageComments="test comment")
        meta = dicom2mrd.build_meta(ds)
        assert meta["ImageComments"] == "test comment"

    def test_pixel_data_removed_before_serialization(self):
        ds = make_pixel_dataset(rows=2, cols=2, pixel_value=5, SeriesDescription="T1")
        assert "PixelData" in ds
        dicom2mrd.build_meta(ds)
        assert "PixelData" not in ds

    def test_dicom_json_is_base64_encoded(self):
        ds = make_dataset(SeriesDescription="Test")
        meta = dicom2mrd.build_meta(ds)
        decoded = base64.b64decode(meta["DicomJson"]).decode("utf-8")
        parsed = json.loads(decoded)
        assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# build_mrd_image()
# ---------------------------------------------------------------------------
def make_classic_mr_dataset(rows=4, cols=6, pixel_value=100, image_type_letter="M",
                             contrast_echo_numbers=1, instance_number=1,
                             position=(1.0, 2.0, 3.0),
                             orientation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                             acquisition_time="120000.000000",
                             trigger_time=None, slice_location=0.0,
                             series_description="Test"):
    extra = dict(
        SOPClassUID=pydicom.uid.MRImageStorage,
        PixelSpacing=[2.0, 2.0], SliceThickness=5.0,
        ImageType=["ORIGINAL", "PRIMARY", image_type_letter],
        EchoNumbers=contrast_echo_numbers,
        InstanceNumber=instance_number,
        ImagePositionPatient=list(position),
        ImageOrientationPatient=list(orientation),
        AcquisitionTime=acquisition_time,
        SliceLocation=slice_location,
        SeriesDescription=series_description,
    )
    if trigger_time is not None:
        extra["TriggerTime"] = trigger_time
    return make_pixel_dataset(rows=rows, cols=cols, pixel_value=pixel_value, **extra)


class TestBuildMrdImage:

    def test_neither_pixel_array_nor_spectroscopy_returns_none(self):
        ds = make_dataset(filename="dummy.dcm")
        result = dicom2mrd.build_mrd_image(ds, serieIndex=1, uSliceLoc=np.zeros(1), uTrigTime=np.zeros(1))
        assert result is None

    def test_spectroscopy_branch_builds_image(self):
        spec_data = np.zeros(4, dtype=np.complex64).tobytes()
        ds = make_dataset(
            SOPClassUID=pydicom.uid.MRSpectroscopyStorage,
            SpectroscopyData=spec_data,
            VolumeLocalizationSequence=[
                make_dataset(SlabThickness=10.0), make_dataset(SlabThickness=10.0), make_dataset(SlabThickness=10.0)
            ],
            ImageType=["ORIGINAL", "PRIMARY", "M"],
            EchoNumbers=1, InstanceNumber=1,
            ImagePositionPatient=[0.0, 0.0, 0.0],
            ImageOrientationPatient=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            AcquisitionTime="120000.000000",
            SeriesDescription="spectro",
        )
        result = dicom2mrd.build_mrd_image(ds, serieIndex=1, uSliceLoc=np.zeros(1), uTrigTime=np.zeros(1))
        assert result is not None
        assert result.data.dtype == np.complex64

    @pytest.mark.parametrize("letter,expected", [
        ("M", ismrmrd.IMTYPE_MAGNITUDE),
        ("P", ismrmrd.IMTYPE_PHASE),
        ("R", ismrmrd.IMTYPE_REAL),
        ("I", ismrmrd.IMTYPE_IMAG),
    ])
    def test_image_type_mapping(self, letter, expected):
        ds = make_classic_mr_dataset(image_type_letter=letter)
        result = dicom2mrd.build_mrd_image(ds, serieIndex=1, uSliceLoc=np.array([0.0]), uTrigTime=np.zeros(1))
        assert result.image_type == expected

    def test_unsupported_image_type_defaults_to_magnitude(self):
        ds = make_classic_mr_dataset(image_type_letter="X")
        result = dicom2mrd.build_mrd_image(ds, serieIndex=1, uSliceLoc=np.array([0.0]), uTrigTime=np.zeros(1))
        assert result.image_type == ismrmrd.IMTYPE_MAGNITUDE

    def test_position_and_orientation_classic_branch(self):
        ds = make_classic_mr_dataset(position=(1.0, 2.0, 3.0), orientation=(1, 0, 0, 0, 1, 0))
        result = dicom2mrd.build_mrd_image(ds, serieIndex=1, uSliceLoc=np.array([0.0]), uTrigTime=np.zeros(1))
        assert result.position == pytest.approx((1.0, 2.0, 3.0))
        assert result.read_dir == pytest.approx((1, 0, 0))
        assert result.phase_dir == pytest.approx((0, 1, 0))

    def test_slice_dir_is_cross_product_of_read_and_phase(self):
        ds = make_classic_mr_dataset(orientation=(1, 0, 0, 0, 1, 0))
        result = dicom2mrd.build_mrd_image(ds, serieIndex=1, uSliceLoc=np.array([0.0]), uTrigTime=np.zeros(1))
        assert result.slice_dir == pytest.approx((0, 0, 1))

    def test_acquisition_time_stamp_formula(self):
        # 01:02:03.000000 -> 3723 sec -> ts = round(3723 * 1000 / 2.5)
        ds = make_classic_mr_dataset(acquisition_time="010203.000000")
        result = dicom2mrd.build_mrd_image(ds, serieIndex=1, uSliceLoc=np.array([0.0]), uTrigTime=np.zeros(1))
        assert result.acquisition_time_stamp == round(3723 * 1000 / 2.5)

    def test_trigger_time_sets_physiology_timestamp(self):
        ds = make_classic_mr_dataset(trigger_time=25.0)
        result = dicom2mrd.build_mrd_image(ds, serieIndex=1, uSliceLoc=np.array([0.0]), uTrigTime=np.array([0.0, 25.0]))
        assert result.physiology_time_stamp[0] == round(int(25.0 / 2.5))

    def test_missing_trigger_time_leaves_default_physiology_timestamp(self):
        ds = make_classic_mr_dataset(trigger_time=None)
        result = dicom2mrd.build_mrd_image(ds, serieIndex=1, uSliceLoc=np.array([0.0]), uTrigTime=np.zeros(1))
        assert result.physiology_time_stamp[0] == 0

    def test_series_index_and_image_index(self):
        ds = make_classic_mr_dataset(instance_number=7)
        result = dicom2mrd.build_mrd_image(ds, serieIndex=3, uSliceLoc=np.array([0.0]), uTrigTime=np.zeros(1))
        assert result.image_series_index == 3
        assert result.image_index == 7

    def test_slice_index_looked_up_from_uSliceLoc(self):
        ds = make_classic_mr_dataset(slice_location=10.0)
        result = dicom2mrd.build_mrd_image(ds, serieIndex=1, uSliceLoc=np.array([0.0, 10.0, 20.0]), uTrigTime=np.zeros(1))
        assert result.slice == 1

    def test_contrast_is_echo_numbers_minus_one(self):
        ds = make_classic_mr_dataset(contrast_echo_numbers=3)
        result = dicom2mrd.build_mrd_image(ds, serieIndex=1, uSliceLoc=np.array([0.0]), uTrigTime=np.zeros(1))
        assert result.contrast == 2

    def test_pixel_values_preserved(self):
        ds = make_classic_mr_dataset(rows=2, cols=2, pixel_value=42)
        result = dicom2mrd.build_mrd_image(ds, serieIndex=1, uSliceLoc=np.array([0.0]), uTrigTime=np.zeros(1))
        assert np.all(result.data == 42)


# ---------------------------------------------------------------------------
# build_series_images()
# ---------------------------------------------------------------------------
class TestBuildSeriesImages:

    def test_builds_one_image_per_dataset(self):
        ds1 = make_classic_mr_dataset(instance_number=1, slice_location=0.0)
        ds2 = make_classic_mr_dataset(instance_number=2, slice_location=10.0)
        ds1.SeriesNumber = 5
        ds2.SeriesNumber = 5

        images = dicom2mrd.build_series_images([ds2, ds1], seriesIndex=5)

        assert len(images) == 2
        assert images[0].image_index == 1
        assert images[1].image_index == 2

    def test_defaults_to_single_slice_without_slice_location(self):
        ds = make_classic_mr_dataset(instance_number=1)
        del ds.SliceLocation
        ds.SeriesNumber = 1

        images = dicom2mrd.build_series_images([ds], seriesIndex=1)

        assert len(images) == 1
        assert images[0].slice == 0

    def test_slice_order_matches_first_dataset_when_reversed(self):
        ds1 = make_classic_mr_dataset(instance_number=1, slice_location=10.0)  # premier, mais SliceLocation la PLUS GRANDE
        ds2 = make_classic_mr_dataset(instance_number=2, slice_location=0.0)
        ds1.SeriesNumber = 1
        ds2.SeriesNumber = 1

        images = dicom2mrd.build_series_images([ds1, ds2], seriesIndex=1)

        assert images[0].slice == 0
        assert images[1].slice == 1

    