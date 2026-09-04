import json

import ismrmrd
import pytest

from server.connection import Connection
from server.server import Server


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


@pytest.fixture
def server_obj_invertcontrast(monkeypatch):
    """
    A real Server wired to the real 'invertContrast' example application
    (app/invertContrast.py), exactly as it runs via `python main.py`.

    The NIfTI debug export performed by invertContrast.process_image() is
    stubbed out: it is a debug/inspection side effect (writes files under
    ./test/data on every call) that is unrelated to the MRD contract this
    suite verifies, and would otherwise write to disk on every test run.
    """
    import apps.demo.invertContrast as invertcontrast_module
    monkeypatch.setattr(invertcontrast_module, "nifti_from_image_array", lambda *a, **kw: None)

    # __init__ is bypassed on purpose: we don't want a real bound TCP
    # socket for this test, only the attributes handle() actually reads.
    s = Server.__new__(Server)
    s.debug = False
    s.app_config = "invertContrast"
    s.app_directory = "demo"
    s.savedata = False
    s.saveFolder = ""
    s.save_nifti = False
    return s


def run_invertcontrast_session(server_obj, socketpair, images, config=None):
    """
    Drive one full client -> server -> client MRD session synchronously,
    and return every ismrmrd.Image the (fake) client received back.

    Parameters
    ----------
    server_obj : Server
        Server instance under test (e.g. server_obj_invertcontrast).
    socketpair : tuple(socket, socket)
        As returned by the `socketpair` fixture.
    images : list of ismrmrd.Image
        Images to send to the server, in order.
    config : dict or None
        Optional OpenRecon parameters, sent as
        `{"parameters": config}` in a single JSON text message.
    """
    client_sock, server_sock = socketpair
    client = Connection(client_sock, savedata=False)

    client.send_config_text("openrecon")
    client.send_metadata(MINIMAL_MRD_HEADER)
    if config is not None:
        client.send_text(json.dumps({"parameters": config}))

    for image in images:
        client.send_image(image)
    client.send_close()

    server_obj.handle(server_sock)

    return [item for item in client if item is not None]


@pytest.mark.integration
class TestInvertContrastFullPipeline:
    """
    Verifies the full round trip through the real template + the real
    invertContrast example: MRD in -> Server -> Pipeline -> app module
    -> Server -> MRD out, checking both pixel values and metadata.
    """

    def test_pixel_values_are_correctly_inverted(self, server_obj_invertcontrast, socketpair, make_image):
        # 0.0 and 4095.0 are chosen so the normalise()/invert maths in
        # invertContrast.py fall on exact integers (scale factor becomes
        # exactly 1.0), avoiding round-half-to-even ambiguity in the test.
        img_low = make_image(slice=0, image_series_index=0, image_type=ismrmrd.IMTYPE_MAGNITUDE, value=0.0)
        img_high = make_image(slice=1, image_series_index=0, image_type=ismrmrd.IMTYPE_MAGNITUDE, value=4095.0)

        received = run_invertcontrast_session(server_obj_invertcontrast, socketpair, [img_low, img_high])

        # 2 original images (SaveOriginal defaults to True) + 2 processed images
        assert len(received) == 4

        originals = [img for img in received if img.image_series_index == 0]
        processed = [img for img in received if img.image_series_index != 0]
        assert len(originals) == 2
        assert len(processed) == 2

        # Originals are untouched (still float32, still the values that were sent)
        original_values = sorted(float(img.data.flat[0]) for img in originals)
        assert original_values == pytest.approx([0.0, 4095.0])

        # Processed images: contrast is inverted, so low <-> high swap
        processed_by_slice = {img.slice: int(img.data.flat[0]) for img in processed}
        assert processed_by_slice[0] == 4095  # was 0.0 pre-inversion
        assert processed_by_slice[1] == 0     # was 4095.0 pre-inversion

    def test_processed_series_index_offset(self, server_obj_invertcontrast, socketpair, make_image):
        # Original images use image_series_index=5. The processed series
        # must not collide with it: offset = max(original indexes) + 1.
        images = [make_image(slice=0, image_series_index=5, value=10.0)]

        received = run_invertcontrast_session(server_obj_invertcontrast, socketpair, images)

        processed = [img for img in received if img.image_series_index != 5]
        assert len(processed) == 1
        assert processed[0].image_series_index == 6

    def test_processed_metadata_is_stamped(self, server_obj_invertcontrast, socketpair, make_image):
        images = [make_image(slice=0, image_series_index=0, value=100.0)]

        received = run_invertcontrast_session(server_obj_invertcontrast, socketpair, images)
        processed = [img for img in received if img.image_series_index != 0][0]

        meta = ismrmrd.Meta.deserialize(processed.attribute_string)
        assert meta["Keep_image_geometry"] == "1"
        assert meta["SequenceDescriptionAdditional"] == "invertcontrast"

        history = meta["ImageProcessingHistory"]
        if isinstance(history, str):
            history = [history]
        assert "INVERT" in history

    def test_save_original_false_returns_only_processed_images(self, server_obj_invertcontrast, socketpair, make_image):
        images = [make_image(slice=0, image_series_index=0, value=10.0)]

        received = run_invertcontrast_session(
            server_obj_invertcontrast, socketpair, images, config={"SaveOriginal": False}
        )

        assert len(received) == 1
        assert received[0].image_series_index != 0

    def test_multiple_image_types_produce_separate_output_series(self, server_obj_invertcontrast, socketpair, make_image):
        mag = make_image(slice=0, image_series_index=0, image_type=ismrmrd.IMTYPE_MAGNITUDE, value=1000.0)
        phase = make_image(slice=0, image_series_index=0, image_type=ismrmrd.IMTYPE_PHASE, value=2000.0)

        received = run_invertcontrast_session(
            server_obj_invertcontrast, socketpair, [mag, phase], config={"SaveOriginal": False}
        )

        # One output series per (image_type, series) pair encountered:
        # magnitude -> series 0, phase -> series 1, then offset by
        # (max original index + 1) = 1 -> final indexes 1 and 2.
        assert len(received) == 2
        assert sorted(img.image_series_index for img in received) == [1, 2]

    def test_image_type_filter_processes_only_selected_type(self, server_obj_invertcontrast, socketpair, make_image):
        mag = make_image(slice=0, image_series_index=0, image_type=ismrmrd.IMTYPE_MAGNITUDE, value=1000.0)
        phase = make_image(slice=0, image_series_index=0, image_type=ismrmrd.IMTYPE_PHASE, value=2000.0)

        received = run_invertcontrast_session(
            server_obj_invertcontrast, socketpair, [mag, phase], config={"ImageType": "Magnitude"}
        )

        # Both originals are still sent (SaveOriginal defaults to True and
        # is independent from the ImageType filter), but only the
        # magnitude image is actually processed.
        originals = [img for img in received if img.image_series_index == 0]
        processed = [img for img in received if img.image_series_index != 0]

        assert len(originals) == 2
        assert len(processed) == 1
        assert processed[0].image_type == ismrmrd.IMTYPE_MAGNITUDE

    def test_debug_mode_bypasses_processing_module(self, server_obj_invertcontrast, socketpair, make_image, monkeypatch):
        import apps.demo.invertContrast as invertcontrast_module

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("process_image() must not be called when Debug=True")

        monkeypatch.setattr(invertcontrast_module, "process_image", _fail_if_called)

        image = make_image(slice=0, image_series_index=0, value=42.0)

        received = run_invertcontrast_session(
            server_obj_invertcontrast, socketpair, [image], config={"Debug": True}
        )

        assert len(received) == 1
        assert received[0].image_series_index == 0
        assert float(received[0].data.flat[0]) == pytest.approx(42.0)