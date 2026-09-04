import ismrmrd
import numpy as np
import pytest

from utils.OutputSeries import OutputSeries

@pytest.fixture
def base_head_meta(make_image):
    """Deux images de référence (head + meta) pour construire une série."""
    img_a = make_image(slice=0, image_series_index=0)
    img_b = make_image(slice=1, image_series_index=0)
    head = [img_a.getHead(), img_b.getHead()]
    meta = [
        ismrmrd.Meta.deserialize(img_a.attribute_string),
        ismrmrd.Meta.deserialize(img_b.attribute_string),
    ]
    return head, meta

# ---------------------------------------------------------------------------
# Tests for OutputSeries.__init__()
# ---------------------------------------------------------------------------
def test_empty_output_series_has_length_zero():
    series = OutputSeries()
    assert len(series) == 0
    assert series.get() == []

# ---------------------------------------------------------------------------
# Tests for OutputSeries.add()
# ---------------------------------------------------------------------------
def test_add_returns_self_for_chaining(base_head_meta):
    head, meta = base_head_meta
    data = np.zeros((2, 1, 1, 4, 4), dtype=np.float32)
    series = OutputSeries()

    result = series.add(data, head, meta)
    assert result is series

def test_add_increments_series_count(base_head_meta):
    head, meta = base_head_meta
    data = np.zeros((2, 1, 1, 4, 4), dtype=np.float32)
    series = OutputSeries()

    series.add(data, head, meta)
    assert len(series) == 1

    series.add(data, head, meta)
    assert len(series) == 2

def test_add_mismatched_head_meta_length_raises(base_head_meta):
    head, meta = base_head_meta
    data = np.zeros((2, 1, 1, 4, 4), dtype=np.float32)
    series = OutputSeries()

    with pytest.raises(ValueError):
        series.add(data, head, meta[:1])
    
def test_add_mismatched_head_length_vs_data_shape_raises(base_head_meta):
    head, meta = base_head_meta
    # data.shape[0] == 3 but head/meta only have 2 elements
    data = np.zeros((3, 1, 1, 4, 4), dtype=np.float32)
    series = OutputSeries()

    with pytest.raises(ValueError):
        series.add(data, head, meta)

def test_no_series_added_after_failed_add(base_head_meta):
    head, meta = base_head_meta
    # data.shape[0] == 3 but head/meta only have 2 elements
    data = np.zeros((3, 1, 1, 4, 4), dtype=np.float32)
    series = OutputSeries()

    with pytest.raises(ValueError):
        series.add(data, head, meta)
    assert len(series) == 0

def test_add_sets_image_series_index_per_series(base_head_meta):
    head, meta = base_head_meta
    data = np.zeros((2, 1, 1, 4, 4), dtype=np.float32)
    series = OutputSeries()

    series.add(data, head, meta)  # series 0
    series.add(data, head, meta)  # series 1

    _, head0, _ = series.get()[0]
    _, head1, _ = series.get()[1]

    assert all(h.image_series_index == 0 for h in head0)
    assert all(h.image_series_index == 1 for h in head1)

def test_add_appends_process_history(base_head_meta):
    head, meta = base_head_meta
    data = np.zeros((2, 1, 1, 4, 4), dtype=np.float32)
    series = OutputSeries()

    series.add(data, head, meta, process_history="Test")
    _, _, m = series.get()[0]
    assert all(item["ImageProcessingHistory"] == ["Test"] for item in m)

def test_add_joins_sequence_description_list_with_underscore(base_head_meta):
    head, meta = base_head_meta
    data = np.zeros((2, 1, 1, 4, 4), dtype=np.float32)
    series = OutputSeries()

    series.add(data, head, meta, sequence_description=["Test", "List"])
    _, _, m = series.get()[0]
    assert all(item["SequenceDescriptionAdditional"] == "Test_List" for item in m)

def test_add_deep_copies_head_so_series_are_independent(base_head_meta):
    head, meta = base_head_meta
    data = np.zeros((2, 1, 1, 4, 4), dtype=np.float32)
    series = OutputSeries()

    series.add(data, head, meta)
    head[0].image_series_index = 999
    _, stored_head, _ = series.get()[0]
    assert stored_head[0].image_series_index != 999

# ---------------------------------------------------------------------------
# Tests for OutputSeries.get()
# ---------------------------------------------------------------------------

def test_get_returns_series_in_insertion_order(base_head_meta):
    data = np.zeros((2, 1, 1, 4, 4), dtype=np.float32)
    head, meta = base_head_meta
    series = OutputSeries()
    series.add(data, head, meta, sequence_description="First")
    series.add(data, head, meta, sequence_description="Second")

    result = series.get()
    assert len(result) == 2
    _, _, meta0 = result[0]
    _, _, meta1 = result[1]
    assert meta0[0]["SequenceDescriptionAdditional"] == "First"
    assert meta1[0]["SequenceDescriptionAdditional"] == "Second"
