import json
import sys
import types

import ismrmrd
import numpy as np
import pytest

from utils.img_array import build_image_array, flatten, get_subarray, stack_images
from server.connection import Connection
from server.server import Server


@pytest.mark.integration
class TestBuildImageArrayOnRealCapture:

    def test_every_captured_image_is_recovered(self, mrd_sample_dataset):
        _, images = mrd_sample_dataset

        arr = build_image_array(images)
        recovered = flatten(arr)

        assert len(recovered) == len(images)
        assert set(id(img) for img in recovered) == set(id(img) for img in images)

    def test_array_shape_matches_declared_dimensions(self, mrd_sample_dataset):
        _, images = mrd_sample_dataset
        arr = build_image_array(images)

        from utils.img_array import mrd_indexes
        for dim in mrd_indexes:
            observed_max = max(getattr(img, dim.name) for img in images)
            assert arr.shape[dim] >= observed_max + 1


@pytest.mark.integration
class TestImageTypeFilteringOnRealCapture:

    def test_magnitude_filter_returns_only_magnitude_images(self, mrd_sample_dataset):
        _, images = mrd_sample_dataset
        arr = build_image_array(images)

        sub = get_subarray(arr, img_image_type=ismrmrd.IMTYPE_MAGNITUDE)
        result = flatten(sub)

        magnitude_present = any(img.image_type == ismrmrd.IMTYPE_MAGNITUDE for img in images)
        if not magnitude_present:
            pytest.skip("Sample capture contains no magnitude images.")

        assert len(result) > 0
        assert all(img.image_type == ismrmrd.IMTYPE_MAGNITUDE for img in result)


@pytest.mark.integration
class TestStackImagesOnRealCapture:

    def test_stack_preserves_count_and_casts_dtype(self, mrd_sample_dataset):
        _, images = mrd_sample_dataset

        data, head, meta = stack_images(images, dtype=np.float32)

        assert data.shape[0] == len(images)
        assert data.dtype == np.float32
        assert len(head) == len(images)
        assert len(meta) == len(images)

    def test_metadata_header_parses_as_valid_mrd_xml(self, mrd_sample_dataset):
        xml_header, _ = mrd_sample_dataset

        parsed = ismrmrd.xsd.CreateFromDocument(xml_header)

        assert len(parsed.encoding) >= 1


# ---------------------------------------------------------------------------
# A real (tiny) app module, registered under app.<name> like a genuine
# OpenRecon processing module would be. This is not a mock: Pipeline loads
# it via importlib exactly as it would load a user-provided module.
# ---------------------------------------------------------------------------
@pytest.fixture
def dummy_app_module():
    module = types.ModuleType("apps.app.dummy_integration_app")

    def process_image(img_array, configJSON, metadata):
        from utils.img_array import flatten, stack_images

        images = flatten(img_array)
        data, head, meta = stack_images(images, dtype=np.float32)
        return [(data, head, meta)]

    module.process_image = process_image
    sys.modules["apps.app.dummy_integration_app"] = module

    yield "dummy_integration_app"

    del sys.modules["apps.app.dummy_integration_app"]


@pytest.fixture
def server_obj(dummy_app_module):
    # __init__ is bypassed on purpose: we don't want a real bound TCP
    # socket for this test, only the attributes handle() actually reads.
    s = Server.__new__(Server)
    s.debug = False
    s.app_config = dummy_app_module
    s.app_directory = "app"
    s.savedata = False
    s.saveFolder = ""
    s.save_nifti = False
    return s


MINIMAL_MRD_HEADER = (
    '<?xml version="1.0" encoding="utf-8"?>'
    '<ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD">'
    '<encoding>'
    '<encodedSpace>'
    '<matrixSize><x>4</x><y>4</y><z>1</z></matrixSize>'
    '<fieldOfView_mm><x>200</x><y>200</y><z>5</z></fieldOfView_mm>'
    '</encodedSpace>'
    '<reconSpace>'
    '<matrixSize><x>4</x><y>4</y><z>1</z></matrixSize>'
    '<fieldOfView_mm><x>200</x><y>200</y><z>5</z></fieldOfView_mm>'
    '</reconSpace>'
    '<trajectory>cartesian</trajectory>'
    '</encoding>'
    '</ismrmrdHeader>'
)


@pytest.mark.integration
class TestServerHandleEndToEnd:

    def test_full_round_trip_produces_original_and_processed_images(self, server_obj, socketpair, make_image
    ):
        client_sock, server_sock = socketpair
        client = Connection(client_sock, savedata=False)

        client.send_config_text("openrecon")
        client.send_metadata(MINIMAL_MRD_HEADER)
        client.send_text(json.dumps({"parameters": {"SaveOriginal": True}}))
        image = make_image(slice=0, image_series_index=0, shape=(1, 1, 4, 4))
        client.send_image(image)
        client.send_close()

        # --- server processes the handshake synchronously ----------------
        server_obj.handle(server_sock)

        # --- client reads back everything the server sent ----------------
        received = [item for item in client if item is not None]

        assert len(received) == 2
        assert all(isinstance(img, ismrmrd.Image) for img in received)

        original, processed = received
        assert original.image_series_index == 0
        assert processed.image_series_index != original.image_series_index

    def test_non_openrecon_config_drains_stream_without_processing(self, server_obj, socketpair
    ):
        client_sock, server_sock = socketpair
        client = Connection(client_sock, savedata=False)

        client.send_config_text("some_other_recon")
        client.send_metadata(MINIMAL_MRD_HEADER)
        client.send_close()

        server_obj.handle(server_sock)

        received = [item for item in client if item is not None]
        assert received == []