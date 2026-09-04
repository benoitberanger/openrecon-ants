import os
import socket

import ismrmrd
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# make_image: build a real ismrmrd.Image
#
# Use this whenever a test needs a *real* header/meta/serialization
# (Pipeline, OutputSeries, Connection round-trips, ...).
# ---------------------------------------------------------------------------
@pytest.fixture
def make_image():

    def _make(
        slice=0,
        contrast=0,
        average=0,
        phase=0,
        repetition=0,
        set=0,
        image_type=ismrmrd.IMTYPE_MAGNITUDE,
        image_series_index=0,
        shape=(1, 1, 4, 4),
        value=1.0,
        dtype=np.float32
    ):
        data = np.full(shape, value, dtype=dtype)
        img = ismrmrd.Image.from_array(data, transpose=False)

        img.slice = slice
        img.contrast = contrast
        img.average = average
        img.phase = phase
        img.repetition = repetition
        img.set = set
        img.image_type = image_type
        img.image_series_index = image_series_index

        meta = ismrmrd.Meta()
        img.attribute_string = meta.serialize()

        return img

    return _make


# ---------------------------------------------------------------------------
# FakeImage: minimal, attribute-only double for utils/img_array.py tests.
# ---------------------------------------------------------------------------
class FakeImage:
    def __init__(self, slice=0, contrast=0, average=0, phase=0,
                 repetition=0, set=0, image_type=ismrmrd.IMTYPE_MAGNITUDE,
                 image_series_index=0, data=None):
        self.slice = slice
        self.contrast = contrast
        self.average = average
        self.phase = phase
        self.repetition = repetition
        self.set = set
        self.image_type = image_type
        self.image_series_index = image_series_index
        self.data = data if data is not None else np.ones((1, 2, 3, 3), dtype=np.float32)
        self.attribute_string = "<?xml version='1.0' encoding='UTF-8'?><ismrmrdMeta></ismrmrdMeta>"

    def getHead(self):
        return {"slice": self.slice, "contrast": self.contrast}


# ---------------------------------------------------------------------------
# make_header: build a real ismrmrd.ImageHeader with specific geometry/index
# fields set.
# ---------------------------------------------------------------------------
@pytest.fixture
def make_header():

    def _make(
        image_series_index=0,
        matrix=(64, 64, 1),
        fov=(240.0, 240.0, 5.0),
        read_dir=(1.0, 0.0, -1.0),
        phase_dir=(-1.0, 1.0, 0.0),
        slice_dir=(0.0, -1.0, 1.0),
    ):
        tmp_img = ismrmrd.Image.from_array(np.zeros((1, 2, 2), dtype=np.int16), transpose=False)
        head = tmp_img.getHead()
        head.image_series_index = image_series_index
        head.matrix_size[:] = matrix
        head.field_of_view[:] = fov
        head.read_dir = read_dir
        head.phase_dir = phase_dir
        head.slice_dir = slice_dir
        return head

    return _make


# ---------------------------------------------------------------------------
# FakeConnection: minimal double used by Pipeline tests. Only implements
# what Pipeline actually calls on a connection (send_image, send_logging,
# send_close).
# ---------------------------------------------------------------------------
class FakeConnection:

    def __init__(self):
        self.sent_images = []
        self.logs = []
        self.closed = False

    def send_image(self, image_or_list):
        if isinstance(image_or_list, list):
            self.sent_images.extend(image_or_list)
        else:
            self.sent_images.append(image_or_list)

    def send_logging(self, level, message):
        self.logs.append((level, message))

    def send_close(self):
        self.closed = True


@pytest.fixture
def fake_connection():
    return FakeConnection()


# ---------------------------------------------------------------------------
# FakeSocket/RaisingSocket: fake the true system boundary (the TCP socket)
# so server.connection.Connection can be tested without a real network.
# ---------------------------------------------------------------------------
class FakeSocket:
    def __init__(self, incoming: bytes = b""):
        self._incoming = incoming
        self._pos = 0
        self.sent = bytearray()
        self.shutdown_called_with = None
        self.closed = False

    def recv(self, nbytes, flags=0):
        chunk = self._incoming[self._pos: self._pos + nbytes]
        if not (flags & socket.MSG_PEEK):
            self._pos += len(chunk)
        return chunk

    def send(self, data):
        data = bytes(data)
        self.sent.extend(data)
        return len(data)

    def shutdown(self, how):
        self.shutdown_called_with = how

    def close(self):
        self.closed = True


class RaisingSocket(FakeSocket):
    """FakeSocket emulating a connection error (ConnectionResetError)."""
    def recv(self, nbytes, flags=0):
        raise ConnectionResetError("connection reset by peer")


# ---------------------------------------------------------------------------
# Real MRD sample data, for integration tests only.
# Kept here so any test file can request it, but nothing in the fast unit
# suite depends on it. Tests that use this fixture must be marked
# @pytest.mark.integration.
# ---------------------------------------------------------------------------

#TO-DO: Change folder name
MRD_SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "data/MRD_in")

def _discover_mrd_samples():
    if not os.path.isdir(MRD_SAMPLE_DIR):
        return []
    return sorted(
        os.path.join(MRD_SAMPLE_DIR, f)
        for f in os.listdir(MRD_SAMPLE_DIR)
        if f.endswith(".h5")
    )
 
 
def pytest_generate_tests(metafunc):
    if "mrd_sample_path" in metafunc.fixturenames:
        samples = _discover_mrd_samples()
        metafunc.parametrize(
            "mrd_sample_path",
            samples,
            ids=[os.path.basename(p) for p in samples],
        )
 
 
@pytest.fixture
def mrd_sample_path(request):
    """
    Path to one real MRD (.h5) sample file.
    """
    return request.param
 
 
@pytest.fixture
def mrd_sample_dataset(mrd_sample_path):
    """
    Opens one real captured MRD file (mrd_sample_path is parametrized
    across every sample, see above) and yields (xml_header, images):
 
    - xml_header : str, the raw MRD XML header as stored in the file.
    - images     : list of ismrmrd.Image, every image found across every
                   'image_<series_index>' group in the file, in group then
                   index order (i.e. as connection.py originally wrote them
                   via dset.append_image("image_%d" % series_index, image)).
    """
    dset = ismrmrd.Dataset(mrd_sample_path, "dataset", create_if_needed=False)
    try:
        xml_header = dset.read_xml_header()
        if isinstance(xml_header, bytes):
            xml_header = xml_header.decode("utf-8")
 
        image_groups = sorted(g for g in dset.list() if g.startswith("image_"))
        if not image_groups:
            pytest.skip(f"{mrd_sample_path} contains no 'image_*' group to read images from.")
 
        images = []
        for group in image_groups:
            n = dset.number_of_images(group)
            for i in range(n):
                images.append(dset.read_image(group, i))
 
        if not images:
            pytest.skip(f"{mrd_sample_path} has image groups but no images inside them.")
 
        yield xml_header, images
    finally:
        dset.close()
 
 
@pytest.fixture
def socketpair():
    """
    A real connected pair of AF_UNIX sockets, standing in for the TCP
    socket between an MRD client and the server. Used by integration
    tests that exercise Connection/Server against real socket semantics
    (MSG_PEEK, MSG_WAITALL, partial reads) instead of FakeSocket.
    """
    client_sock, server_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    yield client_sock, server_sock
    for s in (client_sock, server_sock):
        try:
            s.close()
        except OSError:
            pass
