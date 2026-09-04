import sys

import ismrmrd
import numpy as np
import pytest

from server.pipeline import Pipeline
from utils.img_array import build_image_array, flatten


@pytest.fixture
def dummy_app():
    import sys
    import types
 
    module = types.ModuleType("apps.app.dummy_app")
 
    def process_image(img_array, configJSON, metadata):
        from utils.img_array import flatten, stack_images
 
        images = flatten(img_array)
        data, head, meta = stack_images(images, dtype=np.float32)
        return [(data, head, meta)]
 
    module.process_image = process_image
    sys.modules["apps.app.dummy_app"] = module
 
    yield "dummy_app"
 
    del sys.modules["apps.app.dummy_app"]

@pytest.fixture
def pipeline_obj(fake_connection, dummy_app):
    return Pipeline(fake_connection, app_config=dummy_app, app_directory="app", save_nifti=False)



# ---------------------------------------------------------------------------
# test of pipeline.load_module()
# ---------------------------------------------------------------------------
class TestLoadModule:
    def test_import_error_leaves_module_none(self, fake_connection):
        pipeline = Pipeline(fake_connection, app_config="does_not_exist", app_directory="app", save_nifti=False)
        assert pipeline.module is None

    def test_success_sets_module(self, fake_connection, dummy_app):
        pipeline = Pipeline(fake_connection, app_config=dummy_app, app_directory="app", save_nifti=False)
        assert pipeline.module is sys.modules["apps.app.dummy_app"]
        assert hasattr(pipeline.module, "process_image")

    def test_wrong_app_directory_fails_to_import(self, fake_connection, dummy_app):
        pipeline = Pipeline(fake_connection, app_config=dummy_app, app_directory="wrong", save_nifti=False)
        assert pipeline.module is None


# ---------------------------------------------------------------------------
# test of pipeline.images_selector()
# ---------------------------------------------------------------------------

class TestImageSelector:
    def test_defaults_to_all_when_no_config(self, pipeline_obj, make_image):
            images = [make_image(contrast=c) for c in range(3)]
            arr = build_image_array(images)

            result = pipeline_obj.images_selector(arr, configJSON=None)

            images_list = flatten(result)
            assert len(images_list) == len(images)
            for i in range(len(images)):
                assert images_list[i] == images[i]

    def test_filter_magnitude_only(self, pipeline_obj, make_image):
        mag = make_image(image_type=ismrmrd.IMTYPE_MAGNITUDE)
        phase = make_image(image_type=ismrmrd.IMTYPE_PHASE)
        arr = build_image_array([mag, phase])

        result = pipeline_obj.images_selector(arr, configJSON={"parameters":{"ImageType": "Magnitude"}})

        assert flatten(result) == [mag]

    def test_first_echo(self, pipeline_obj, make_image):
        images = [make_image(contrast=c) for c in range(3)]
        arr = build_image_array(images)

        result = pipeline_obj.images_selector(arr, configJSON={"parameters":{"SelectEcho": "FirstEcho"}})

        assert flatten(result) == [images[0]]

    def test_last_echo(self, pipeline_obj, make_image):
        images = [make_image(contrast=c) for c in range(3)]
        arr = build_image_array(images)

        result = pipeline_obj.images_selector(arr, configJSON={"parameters":{"SelectEcho": "LastEcho"}})

        assert flatten(result) == [images[-1]]
    
    def test_unknown_value_falls_back_to_all(self, pipeline_obj, make_image):
        images = [make_image(contrast=c) for c in range(3)]
        arr = build_image_array(images)

        result = pipeline_obj.images_selector(arr, configJSON={"parameters":{"UnknownParameters"}})

        images_list = flatten(result)
        assert len(images_list) == len(images)
        for i in range(len(images)):
            assert images[i] == images_list[i]


# ---------------------------------------------------------------------------
# test of pipeline.send_volume_as_2Dslices()
# ---------------------------------------------------------------------------
class TestSendVolumeAs2DSlices:

    def test_sends_one_image_per_slice(self, pipeline_obj, make_header):
        data = np.zeros((3, 1, 2, 2), dtype=np.float32)
        head = [make_header() for _ in range(3)]
        meta = [ismrmrd.Meta() for _ in range(3)]

        pipeline_obj.send_volume_as_2Dslices(data, head, meta)

        assert len(pipeline_obj.connection.sent_images) == 3

    def test_image_series_index_offset_is_applied(self, pipeline_obj, make_header):
        pipeline_obj.max_image_series_index = 5
        data = np.zeros((1, 1, 2, 2), dtype=np.float32)
        head = [make_header(image_series_index=0)]
        meta = [ismrmrd.Meta()]

        pipeline_obj.send_volume_as_2Dslices(data, head, meta)

        sent_img = pipeline_obj.connection.sent_images[0]
        assert sent_img.getHead().image_series_index == 6

    def test_keep_image_geometry_flag_is_set_in_meta(self, pipeline_obj, make_header):
        data = np.zeros((2, 1, 2, 2), dtype=np.float32)
        head = [make_header() for _i in range(2)]
        meta = [ismrmrd.Meta() for _ in range(2)]

        pipeline_obj.send_volume_as_2Dslices(data, head, meta)

        assert len(pipeline_obj.connection.sent_images) == 2
        for sent in pipeline_obj.connection.sent_images:
            sent_meta = ismrmrd.Meta.deserialize(sent.attribute_string)
            assert sent_meta["Keep_image_geometry"] == "1"


# ---------------------------------------------------------------------------
# test of pipeline.run()
# ---------------------------------------------------------------------------
class TestRun:
    def test_empty_images_list_send_nothing(self, pipeline_obj):
        result = pipeline_obj.run([], configJSON=None, metadata=None)

        assert result == []
        assert pipeline_obj.connection.sent_images == []

    def test_no_module_sends_original_images_and_stop(self, pipeline_obj, make_image):
        pipeline_obj.module = None
        images = [make_image(slice=0, image_series_index=0)]

        result = pipeline_obj.run(images, configJSON=None, metadata=None)

        assert pipeline_obj.connection.sent_images == images
        assert result == []
    
    def test_save_original_first_by_default(self, pipeline_obj, make_image):
        images = [make_image(slice=0, image_series_index=0)]

        pipeline_obj.run(images, configJSON=None, metadata=None)

        assert pipeline_obj.connection.sent_images[0] == images[0]

    def test_save_original_false_does_not_send_original(self, pipeline_obj, make_image):
        images = [make_image(slice=0, image_series_index=0)]

        pipeline_obj.run(images, configJSON={"parameters":{"SaveOriginal": False}}, metadata=None)

        assert images[0] not in pipeline_obj.connection.sent_images

    def test_sends_one_slice_per_input_image(self, pipeline_obj, make_image):
        images = [
            make_image(slice=0, image_series_index=0),
            make_image(slice=1, image_series_index=0),
            make_image(slice=0, image_series_index=1)
        ]

        pipeline_obj.run(images, configJSON={"parameters":{"SaveOriginal": False}}, metadata=None)

        assert len(pipeline_obj.connection.sent_images) == 3
    
    def test_run_increments_image_series_index_for_processed_slice(self, pipeline_obj, make_image):
        images = [make_image(slice=0, image_series_index=5)]

        pipeline_obj.run(images, configJSON={"parameters":{"SaveOriginal": False}}, metadata=None)

        sent = pipeline_obj.connection.sent_images[0]
        assert sent.image_series_index == 11

    def test_max_image_series_index_tracks_the_highest_input_value(self, pipeline_obj, make_image):
        images = [
            make_image(slice=1, image_series_index=7),
            make_image(slice=0, image_series_index=2),
        ]

        pipeline_obj.run(images, configJSON={"parameters":{"SaveOriginal": False}}, metadata=None)

        assert pipeline_obj.max_image_series_index == 7
        offset = pipeline_obj.max_image_series_index + 1
        sent_indexes = sorted(img.image_series_index for img in pipeline_obj.connection.sent_images)
        assert sent_indexes == sorted([2 + offset, 7 + offset])
