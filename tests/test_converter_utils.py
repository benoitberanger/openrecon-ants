import h5py
import ismrmrd
import numpy as np
import pytest

from converter.utils import slice_pos, check_MRDfile

# ---------------------------------------------------------------------------
# slice_pos()
# ---------------------------------------------------------------------------
class FakeHeaderForSlicePos:
    def __init__(self, position, slice_dir):
        self.position = position
        self.slice_dir = slice_dir


class FakeImageForSlicePos:
    def __init__(self, position, slice_dir):
        self._head = FakeHeaderForSlicePos(position, slice_dir)

    def getHead(self):
        return self._head


class TestSlicePos:

    def test_projects_position_onto_slice_normal(self):
        img = FakeImageForSlicePos(position=(10.0, 20.0, 30.0), slice_dir=(0.0, 0.0, 1.0))
        assert slice_pos(img) == pytest.approx(30.0)

    def test_tilted_slice_normal(self):
        img = FakeImageForSlicePos(position=(1.0, 2.0, 3.0), slice_dir=(0.0, 0.6, 0.8))
        expected = 1.0 * 0.0 + 2.0 * 0.6 + 3.0 * 0.8
        assert slice_pos(img) == pytest.approx(expected)

    def test_negative_slice_dir_flips_sign(self):
        img_pos = FakeImageForSlicePos(position=(0.0, 0.0, 5.0), slice_dir=(0.0, 0.0, 1.0))
        img_neg = FakeImageForSlicePos(position=(0.0, 0.0, 5.0), slice_dir=(0.0, 0.0, -1.0))
        assert slice_pos(img_pos) == -slice_pos(img_neg)

    def test_works_with_real_ismrmrd_header(self):
        img = ismrmrd.Image.from_array(np.zeros((1, 2, 2), dtype=np.int16), transpose=False)
        head = img.getHead()
        head.position = (0.0, 0.0, 12.0)
        head.slice_dir = (0.0, 0.0, 1.0)
        img.setHead(head)
        assert slice_pos(img) == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# check_MRDfile()
# ---------------------------------------------------------------------------
def make_mrd_h5(path, groups=("dataset",), include_xml_in=None, corrupt_image_in=None):
    with h5py.File(str(path), 'w') as f:
        for group_name in groups:
            g = f.create_group(group_name)
            if include_xml_in == group_name:
                g.create_dataset('xml', data=b"<ismrmrdHeader></ismrmrdHeader>")

            img_group = g.create_group('image_0')
            img_group.create_dataset('data', data=np.zeros((1,), dtype=np.int16))
            if corrupt_image_in != group_name:
                img_group.create_dataset('header', data=np.zeros((1,), dtype=np.int8))
            img_group.create_dataset('attributes', data=b"")


class TestCheckMRDfile:

    def test_valid_file_returns_group_name(self, tmp_path):
        h5_path = tmp_path / "input.h5"
        make_mrd_h5(h5_path, groups=("dataset_test",))

        result = check_MRDfile(str(h5_path), in_group=None, out_folder=str(tmp_path / "out"))

        assert result == "dataset_test"

    def test_auto_selects_alphabetically_last_group_when_not_specified(self, tmp_path):
        h5_path = tmp_path / "input.h5"
        make_mrd_h5(h5_path, groups=("a_group", "z_group"))

        result = check_MRDfile(str(h5_path), in_group=None, out_folder=str(tmp_path / "out"))

        assert result == "z_group"

    def test_missing_group_returns_none(self, tmp_path):
        h5_path = tmp_path / "input.h5"
        make_mrd_h5(h5_path, groups=("dataset",))

        result = check_MRDfile(str(h5_path), in_group="does_not_exist", out_folder=str(tmp_path / "out"))

        assert result is None

    def test_malformed_image_subgroup_returns_none(self, tmp_path):
        h5_path = tmp_path / "input.h5"
        make_mrd_h5(h5_path, groups=("dataset",), corrupt_image_in="dataset")

        result = check_MRDfile(str(h5_path), in_group="dataset", out_folder=str(tmp_path / "out"))

        assert result is None

    def test_xml_dataset_is_not_treated_as_a_malformed_image_group(self, tmp_path):
        h5_path = tmp_path / "input.h5"
        make_mrd_h5(h5_path, groups=("dataset",), include_xml_in="dataset")

        result = check_MRDfile(str(h5_path), in_group="dataset", out_folder=str(tmp_path / "out"))

        assert result == "dataset"

    def test_creates_out_folder_if_missing(self, tmp_path):
        h5_path = tmp_path / "input.h5"
        make_mrd_h5(h5_path, groups=("dataset",))
        out_folder = tmp_path / "brand_new_output_dir"
        assert not out_folder.exists()

        check_MRDfile(str(h5_path), in_group="dataset", out_folder=str(out_folder))

        assert out_folder.exists()

    def test_default_out_folder_derived_from_filename(self, tmp_path):
        h5_path = tmp_path / "test.h5"
        make_mrd_h5(h5_path, groups=("dataset",))
        expected_out_folder = tmp_path / "test"

        check_MRDfile(str(h5_path), in_group="dataset", out_folder=None)

        assert expected_out_folder.exists()