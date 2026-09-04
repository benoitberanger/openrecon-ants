import ismrmrd
import nibabel as nib
import numpy as np
import pytest

import converter.mrd2nifti as mrd2nifti
from utils.img_array import build_image_array

def make_full_image(position=(0.0, 0.0, 0.0), 
                    slice_dir=(0.0, 0.0, 1.0),
                    read_dir=(1.0, 0.0, 0.0), 
                    phase_dir=(0.0, 1.0, 0.0),
                    matrix_size=(3, 2, 1), 
                    field_of_view=(30.0, 20.0, 5.0),
                    contrast=0, 
                    repetition=0, 
                    image_type=ismrmrd.IMTYPE_MAGNITUDE,
                    image_series_index=0, 
                    data_value=0.0, 
                    shape=(2, 3),
                    meta_kwargs=None):
    ny, nx = shape
    data = np.full((1, 1, ny, nx), data_value, dtype=np.float32)
    img = ismrmrd.Image.from_array(data, transpose=False)

    head = img.getHead()
    head.position = position
    head.slice_dir = slice_dir
    head.read_dir = read_dir
    head.phase_dir = phase_dir
    head.matrix_size[:3] = matrix_size
    head.field_of_view[:3] = field_of_view
    head.contrast = contrast
    head.repetition = repetition
    head.image_type = image_type
    head.image_series_index = image_series_index
    img.setHead(head)

    meta = ismrmrd.Meta()
    for k, v in (meta_kwargs or {}).items():
        meta[k] = v
    img.attribute_string = meta.serialize()

    img.contrast = contrast
    img.repetition = repetition
    img.image_type = image_type
    img.image_series_index = image_series_index

    return img

# ---------------------------------------------------------------------------
# rescale_phase()
# ---------------------------------------------------------------------------
class TestRescalePhase:

    def test_non_phase_type_returns_unchanged(self):
        vol = np.array([1, 2, 3], dtype=np.int16)
        result = mrd2nifti.rescale_phase(vol, {"image_type": "M"})
        assert result is vol

    def test_missing_slope_and_intercept_returns_unchanged(self):
        vol = np.array([1, 2, 3], dtype=np.int16)
        result = mrd2nifti.rescale_phase(vol, {"image_type": "P"})
        assert result is vol

    def test_applies_slope_and_intercept(self):
        vol = np.array([0, 1, 2], dtype=np.int16)
        result = mrd2nifti.rescale_phase(vol, {"image_type": "P", "RescaleSlope": "2.0", "RescaleIntercept": "10.0"})
        assert np.allclose(result, [10.0, 12.0, 14.0])
        assert result.dtype == np.float32

    def test_only_slope_present_intercept_defaults_to_zero(self):
        vol = np.array([1, 2, 3], dtype=np.int16)
        result = mrd2nifti.rescale_phase(vol, {"image_type": "P", "RescaleSlope": "3.0"})
        assert np.allclose(result, [3.0, 6.0, 9.0])

    def test_only_intercept_present_slope_defaults_to_one(self):
        vol = np.array([1, 2, 3], dtype=np.int16)
        result = mrd2nifti.rescale_phase(vol, {"image_type": "P", "RescaleIntercept": "5.0"})
        assert np.allclose(result, [6.0, 7.0, 8.0])


# # ---------------------------------------------------------------------------
# # detect_stack_dir()
# # ---------------------------------------------------------------------------
# class TestDetectStackDir:

#     def test_single_image_returns_positive_one(self):
#         img = make_full_image(position=(0, 0, 0))
#         assert mrd2nifti.detect_stack_dir([img]) == 1.0

#     def test_slice_dir_aligned_with_increasing_position(self):
#         img_low  = make_full_image(position=(0.0, 0.0, 0.0),  slice_dir=(0.0, 0.0, 1.0))
#         img_high = make_full_image(position=(0.0, 0.0, 10.0), slice_dir=(0.0, 0.0, 1.0))
#         assert mrd2nifti.detect_stack_dir([img_low, img_high]) == 1.0


# ---------------------------------------------------------------------------
# build_affine()
# ---------------------------------------------------------------------------
class TestBuildAffine:

    def test_lps_to_ras_conversion_and_voxel_scaling(self):
        img = make_full_image(
            position=(10.0, 20.0, 30.0),
            read_dir=(1.0, 0.0, 0.0),
            phase_dir=(0.0, 1.0, 0.0),
            slice_dir=(0.0, 0.0, 1.0),
            matrix_size=(2, 4, 1),
            field_of_view=(20.0, 40.0, 5.0),  # voxel_size = (10, 10, 5)
        )

        affine = mrd2nifti.build_affine(img)

        assert np.allclose(affine[:3, 3], [-10.0, -20.0, 30.0])
        assert np.allclose(affine[:3, 0], [-10.0, 0.0, 0.0])
        assert np.allclose(affine[:3, 1], [0.0, -10.0, 0.0])
        assert np.allclose(affine[:3, 2], [0.0, 0.0, 5.0])

    # def test_negative_stack_dir_flips_slice_column_only(self):
    #     img = make_full_image(
    #         matrix_size=(2, 2, 1), field_of_view=(20.0, 20.0, 5.0),
    #         read_dir=(1, 0, 0), phase_dir=(0, 1, 0), slice_dir=(0, 0, 1),
    #     )

    #     affine_pos = mrd2nifti.build_affine(img, stack_dir=1.0)
    #     affine_neg = mrd2nifti.build_affine(img, stack_dir=-1.0)

    #     assert np.allclose(affine_pos[:3, 0], affine_neg[:3, 0])
    #     assert np.allclose(affine_pos[:3, 1], affine_neg[:3, 1])
    #     assert np.allclose(affine_pos[:3, 2], -affine_neg[:3, 2])


# ---------------------------------------------------------------------------
# detect_extra_dims()
# ---------------------------------------------------------------------------
class TestDetectExtraDims:

    def test_no_variation_returns_empty_list(self):
        images = [make_full_image(contrast=0), make_full_image(contrast=0)]
        assert mrd2nifti.detect_extra_dims(images) == []

    def test_varying_contrast_detected(self):
        images = [make_full_image(contrast=0), make_full_image(contrast=1)]
        assert mrd2nifti.detect_extra_dims(images) == ["contrast"]

    def test_multiple_varying_dims_preserve_inspection_order(self):
        images = [
            make_full_image(contrast=0, repetition=0),
            make_full_image(contrast=1, repetition=1),
        ]
        assert mrd2nifti.detect_extra_dims(images) == ["contrast", "repetition"]


# ---------------------------------------------------------------------------
# assemble_volume()
# ---------------------------------------------------------------------------
class TestAssembleVolume:

    def test_empty_list_raises_value_error(self):
        with pytest.raises(ValueError):
            mrd2nifti.assemble_volume([], [])

    def test_two_slices_assembled_in_position_order(self):
        img_low  = make_full_image(position=(0.0, 0.0, 0.0),  data_value=10.0)
        img_high = make_full_image(position=(0.0, 0.0, 10.0), data_value=20.0)

        vol, _, _ = mrd2nifti.assemble_volume([img_low, img_high], extra_dims=[])

        assert vol.shape == (3, 2, 2)  # (nx, ny, n_slices)
        assert np.all(vol[:, :, 0] == 10.0)
        assert np.all(vol[:, :, 1] == 20.0)

    def test_meta_contains_expected_keys(self):
        img = make_full_image(image_type=ismrmrd.IMTYPE_PHASE, image_series_index=7)
        _, _, meta = mrd2nifti.assemble_volume([img], extra_dims=[])

        assert meta["image_type"] == "P"
        assert meta["series_index"] == 7
        assert meta["extra_dims"] == []

    def test_optional_meta_keys_populated_from_first_image(self):
        img = make_full_image(meta_kwargs={"SequenceDescription": "test", "EchoTime": "3.0"})
        _, _, meta = mrd2nifti.assemble_volume([img], extra_dims=[])

        assert meta["SequenceDescription"] == "test"
        assert meta["EchoTime"] == "3.0"

    def test_with_extra_dimension(self):
        img_c0 = make_full_image(contrast=0, data_value=1.0)
        img_c1 = make_full_image(contrast=1, data_value=2.0)

        vol, _, _ = mrd2nifti.assemble_volume([img_c0, img_c1], extra_dims=["contrast"])

        assert vol.shape == (3, 2, 1, 2)  # (nx, ny, n_slices, n_contrasts)
        assert np.all(vol[:, :, 0, 0] == 1.0)
        assert np.all(vol[:, :, 0, 1] == 2.0)


# ---------------------------------------------------------------------------
# make_nifti()
# ---------------------------------------------------------------------------
class TestMakeNifti:

    def test_3d_data_basic_header(self):
        data = np.zeros((3, 2, 1), dtype=np.float32)
        affine = np.eye(4)
        meta = {"SequenceDescription": "Test"}

        img = mrd2nifti.make_nifti(data, affine, meta)

        assert isinstance(img, nib.Nifti1Image)
        assert img.header["descrip"].tobytes().rstrip(b"\x00") == b"Test"

    def test_4d_repetition_zoom_uses_repetition_time(self):
        data = np.zeros((3, 2, 1, 2), dtype=np.float32)
        affine = np.eye(4)
        meta = {"extra_dims": ["repetition"], "RepetitionTime": "2000.0"}

        img = mrd2nifti.make_nifti(data, affine, meta)

        assert img.header.get_zooms()[3] == pytest.approx(2000.0)

    def test_4d_contrast_zoom_uses_echo_time(self):
        data = np.zeros((3, 2, 1, 2), dtype=np.float32)
        affine = np.eye(4)
        meta = {"extra_dims": ["contrast"], "EchoTime": "5.5"}

        img = mrd2nifti.make_nifti(data, affine, meta)

        assert img.header.get_zooms()[3] == pytest.approx(5.5)

    def test_4d_other_dim_defaults_to_one(self):
        data = np.zeros((3, 2, 1, 2), dtype=np.float32)
        affine = np.eye(4)
        meta = {"extra_dims": ["set"]}

        img = mrd2nifti.make_nifti(data, affine, meta)

        assert img.header.get_zooms()[3] == pytest.approx(1.0)

    def test_descrip_combines_sequence_description_and_extra_dims(self):
        data = np.zeros((3, 2, 1, 2), dtype=np.float32)
        affine = np.eye(4)
        meta = {"SequenceDescription": "Test", "extra_dims": ["contrast"]}

        img = mrd2nifti.make_nifti(data, affine, meta)

        assert img.header["descrip"].tobytes().rstrip(b"\x00") == b"Test, contrast"


# ---------------------------------------------------------------------------
# nifti_from_image_array()
# ---------------------------------------------------------------------------
class TestNiftiFromImageArray:

    def test_writes_file_and_returns_path(self, tmp_path):
        img = make_full_image(image_series_index=3, image_type=ismrmrd.IMTYPE_MAGNITUDE)
        image_array = build_image_array([img])

        out_path = mrd2nifti.nifti_from_image_array(image_array, str(tmp_path))

        assert out_path.endswith("3_M.nii")
        assert (tmp_path / "3_M.nii").exists()

    def test_filename_includes_sequence_description(self, tmp_path):
        img = make_full_image(
            image_series_index=1, image_type=ismrmrd.IMTYPE_PHASE,
            meta_kwargs={"SequenceDescription": "test"},
        )
        image_array = build_image_array([img])

        out_path = mrd2nifti.nifti_from_image_array(image_array, str(tmp_path))

        assert out_path.endswith("1_test_P.nii")

    def test_all_none_array_raises_value_error(self):
        image_array = np.full((1, 1), None, dtype=object)
        with pytest.raises(ValueError):
            mrd2nifti.nifti_from_image_array(image_array, "/outfolder")

    def test_creates_outfolder_if_missing(self, tmp_path):
        outfolder = tmp_path / "new_output_dir"
        img = make_full_image()
        image_array = build_image_array([img])
        assert not outfolder.exists()

        mrd2nifti.nifti_from_image_array(image_array, str(outfolder))

        assert outfolder.exists()
