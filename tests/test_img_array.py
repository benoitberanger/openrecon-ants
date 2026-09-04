import ismrmrd
import numpy as np
import pytest

from conftest import FakeImage
from utils import img_array


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------
@pytest.fixture
def single_image():
    """One image (slice=0, contrast=0, image_type=MAGNITUDE)"""
    return FakeImage(slice=0, contrast=0, image_type=ismrmrd.IMTYPE_MAGNITUDE)


@pytest.fixture
def multi_slice_images():
    """3 slices, 2 contrasts, for multi-dimensional array testing."""
    images = []
    for s in range(3):
        for c in range(2):
            images.append(FakeImage(slice=s, contrast=c, image_type=1))
    return images


# ---------------------------------------------------------------------------
# Tests for build_image_array()
# ---------------------------------------------------------------------------
class TestBuildImageArray:

    def test_single_image_shape(self, single_image):
        # Arrange : one image only 
        # (1 for all dimensions, except image_type)
        img_list = [single_image]

        # Act
        arr = img_array.build_image_array(img_list)

        # Assert
        assert arr.shape == (1, 1, 1, 1, 1, 1, 2, 1)
        assert arr[0, 0, 0, 0, 0, 0, 1, 0] == [single_image]
        assert arr[0, 0, 0, 0, 0, 0, 0, 0] is None

    def test_multi_slice_contrast_shape(self, multi_slice_images):
        arr = img_array.build_image_array(multi_slice_images)

        # 3 slices (0,1,2), 2 contrasts (0,1), image_type=1
        assert arr.shape[img_array.mrd_indexes.slice] == 3
        assert arr.shape[img_array.mrd_indexes.contrast] == 2
        assert arr.shape[img_array.mrd_indexes.image_type] == 2

    def test_empty_cells_are_none(self, multi_slice_images):
        # Arrange: adding an image to expend the array and create empty cells
        images = multi_slice_images + [FakeImage(slice=0, contrast=0, repetition=5, image_type=1)]
        
        # Act:
        arr = img_array.build_image_array(images)

        # Assert: testing some cells who should be empty
        assert arr[0, 0, 0, 0, 0, 0, 0, 0] is None
        assert arr[0, 0, 0, 0, 3, 0, 1, 0] is None
        assert arr[2, 1, 0, 0, 2, 0, 1, 0] is None

    def test_duplicate_key_appends_to_list(self):
        # Two images with same keys should be one the same cells
        # forming a list.
        img_a = FakeImage(slice=0, contrast=0, image_type=1)
        img_b = FakeImage(slice=0, contrast=0, image_type=1)
        arr = img_array.build_image_array([img_a, img_b])
        cell = arr[0, 0, 0, 0, 0, 0, 1, 0]
        assert cell == [img_a, img_b]


# ---------------------------------------------------------------------------
# Tests for validate_index()
# ---------------------------------------------------------------------------
class TestValidateIndex:

    def test_none_is_always_valid(self):
        assert img_array.validate_index(None, dim_size=5, dim_name="slice") is True

    @pytest.mark.parametrize("value", [0, 4, -1, -5])
    def test_int_in_range(self, value):
        assert img_array.validate_index(value, dim_size=5, dim_name="slice") is True

    @pytest.mark.parametrize("value", [5, -6, 10, 100])
    def test_int_out_of_range(self, value):
        assert img_array.validate_index(value, dim_size=5, dim_name="slice") is False

    def test_slice_selecting_at_least_one_element(self):
        assert img_array.validate_index(slice(0, 2), dim_size=5, dim_name="slice") is True

    def test_slice_selecting_nothing_is_invalid(self):
        assert img_array.validate_index(slice(10, 20), dim_size=5, dim_name="slice") is False


# ---------------------------------------------------------------------------
# Tests for get_subarray()
# ---------------------------------------------------------------------------
class TestGetSubarray:

    def test_select_single_slice(self, multi_slice_images):
        arr = img_array.build_image_array(multi_slice_images)
        sub = img_array.get_subarray(arr, img_slice=1)
        # dimension is preserved (size 1), not dropped
        assert sub.shape[img_array.mrd_indexes.slice] == 1
        assert sub.shape[img_array.mrd_indexes.contrast] == 2

    def test_get_subarray_negative_index_selects_last_element(self):
        images = [FakeImage(contrast=0), FakeImage(contrast=1), FakeImage(contrast=2)]
        arr = img_array.build_image_array(images)

        sub = img_array.get_subarray(arr, img_contrast=-1)
        assert sub.shape[img_array.mrd_indexes.contrast] == 1
        # the -1 selection should correspond to the max contrast, i.e. index 2
        full_sub = img_array.get_subarray(arr, img_contrast=2)
        assert sub[0, 0, 0, 0, 0, 0, ismrmrd.IMTYPE_MAGNITUDE, 0] == \
            full_sub[0, 0, 0, 0, 0, 0, ismrmrd.IMTYPE_MAGNITUDE, 0]
        
    def test_get_subarray_slice_argument(self):
        images = [FakeImage(contrast=c) for c in range(4)]
        arr = img_array.build_image_array(images)

        sub = img_array.get_subarray(arr, img_contrast=slice(1, 3))
        assert sub.shape[img_array.mrd_indexes.contrast] == 2

    def test_select_out_of_range_returns_empty_array(self, multi_slice_images):
        arr = img_array.build_image_array(multi_slice_images)
        sub = img_array.get_subarray(arr, img_slice=99)
        assert sub.size == 0

    def test_no_filter_returns_same_shape(self, multi_slice_images):
        arr = img_array.build_image_array(multi_slice_images)
        sub = img_array.get_subarray(arr)
        assert sub.shape == arr.shape


# ---------------------------------------------------------------------------
# Tests for get_type_magnitude / get_type_phase / get_contrast
# ---------------------------------------------------------------------------
class TestShortcuts:

    def test_get_type_magnitude(self):
        mag = FakeImage(slice=0, image_type=ismrmrd.IMTYPE_MAGNITUDE)
        phase = FakeImage(slice=0, image_type=ismrmrd.IMTYPE_PHASE)
        arr = img_array.build_image_array([mag, phase])

        sub = img_array.get_type_magnitude(arr)
        flat = img_array.flatten(sub)
        assert flat == [mag]

    def test_get_type_phase(self):
        mag = FakeImage(slice=0, image_type=ismrmrd.IMTYPE_MAGNITUDE)
        phase = FakeImage(slice=0, image_type=ismrmrd.IMTYPE_PHASE)
        arr = img_array.build_image_array([mag, phase])

        sub = img_array.get_type_phase(arr)
        flat = img_array.flatten(sub)
        assert flat == [phase]

    def test_get_contrast(self, multi_slice_images):
        arr = img_array.build_image_array(multi_slice_images)
        sub = img_array.get_contrast(arr, img_contrast=0)

        for img in img_array.flatten(sub):
            assert img.contrast == 0


# ---------------------------------------------------------------------------
# Tests for flatten()
# ---------------------------------------------------------------------------
class TestFlatten:

    def test_flatten_returns_all_images(self, multi_slice_images):
        arr = img_array.build_image_array(multi_slice_images)
        result = img_array.flatten(arr)
        assert len(result) == len(multi_slice_images)
        assert set(result) == set(multi_slice_images)

    def test_flatten_empty_array(self):
        arr = np.full((2, 2), None, dtype=object)
        assert img_array.flatten(arr) == []


# ---------------------------------------------------------------------------
# Tests for stack_images()
# ---------------------------------------------------------------------------
class TestStackImages:

    def test_stack_basic(self):
        images = [
            FakeImage(data=np.ones((1, 2, 3, 3))),
            FakeImage(data=np.zeros((1, 2, 3, 3))),
        ]
        data, head, meta = img_array.stack_images(images)

        assert data.shape == (2, 1, 2, 3, 3)  # [img, cha, z, y, x]
        assert len(head) == 2
        assert len(meta) == 2

    def test_stack_casts_dtype(self):
        images = [FakeImage(data=np.ones((1, 1, 2, 2), dtype=np.float64))]
        data, _, _ = img_array.stack_images(images, dtype=np.int16)
        assert data.dtype == np.int16

    def test_stack_images_preserves_order(self):
        images = [
            FakeImage(slice=0, data=np.zeros((1, 2, 3, 3))),
            FakeImage(slice=1, data=np.ones((1, 2, 3, 3))),
        ]
        data, _, _ = img_array.stack_images(images)
        
        assert np.all(data[0] == 0)
        assert np.all(data[1] == 1)
    
    def test_stack_empty_list_raises(self):
        with pytest.raises(ValueError):
            img_array.stack_images([])
