import ismrmrd
import numpy as np
import pytest

import server.server as server_app_module
from server import constants
from server.server import Server

@pytest.fixture
def server_obj():
    # __init__ not called so no real socket
    s = Server.__new__(Server)
    s.debug = False
    s.app_config = "does_not_exist"
    s.app_directory = "app"
    s.save_nifti = False
    return s


# ---------------------------------------------------------------------------
# handle_metadata()
# ---------------------------------------------------------------------------
class FakeMinimalConnection:
    def __init__(self, value, open=True):
        self.value = value
        self.open = open

    def __next__(self):
        return self.value


class TestHandleMetadata:

    def test_valid_xml_is_parsed(self, server_obj):
        xml = (
            '<ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD">'
            '<acquisitionSystemInformation><systemFieldStrength_T>7</systemFieldStrength_T>'
            '<systemVendor>Siemens</systemVendor><systemModel>MAGNETOM Terra.X</systemModel></acquisitionSystemInformation>'
            '</ismrmrdHeader>'
        )

        connection = FakeMinimalConnection(xml)

        result = server_obj.handle_metadata(connection)
        assert isinstance(result, ismrmrd.xsd.ismrmrdHeader)

    def test_invalid_xml_falls_back_to_raw_string(self, server_obj):
        connection = FakeMinimalConnection("this is not valid xml")
        result = server_obj.handle_metadata(connection)
        assert result == "this is not valid xml"
    
    def test_none_with_closed_connection_returns_none(self, server_obj):
        connection = FakeMinimalConnection(None, open=False)
        result = server_obj.handle_metadata(connection)
        assert result is None


# ---------------------------------------------------------------------------
# handleJSON()
# ---------------------------------------------------------------------------
class FakePeekableConnection:
    def __init__(self, peek_id, next_value=None):
        self.peek_id = peek_id
        self.next_value = next_value
        self.next_was_called = False
    
    def peek_mrd_message_identifier(self):
        return self.peek_id
    
    def __next__(self):
        self.next_was_called = True
        return self.next_value


class TestHandleJSON:
    """tests for handle JSON"""

    def test_parses_json_when_text_message_present(self, server_obj):
        connection = FakePeekableConnection(constants.MRD_MESSAGE_TEXT, next_value='{"TestParameters":"test"}')
        result = server_obj.handleJSON(connection)
        assert result == {"TestParameters":"test"}

    def test_no_text_message_return_none(self, server_obj):
        connection = FakePeekableConnection(constants.MRD_MESSAGE_CLOSE)
        result = server_obj.handleJSON(connection)

        assert result is None
        assert connection.next_was_called is False

    def test_malformed_json_return_none(self, server_obj):
        connection = FakePeekableConnection(constants.MRD_MESSAGE_TEXT, next_value="malformed json {{")
        result = server_obj.handleJSON(connection)

        assert result is None


# ---------------------------------------------------------------------------
# handle_image_stream()
# ---------------------------------------------------------------------------
class FakePipeline:
    def __init__(self, connection, app_config, app_directory, save_nifti):
        self.connection = connection
        self.app_config = app_config
        self.app_directory = app_directory
        self.save_nifti = save_nifti

    def run(self, images, configJSON, metadata):
        return []


class FakeStreamConnection:
    def __init__(self, items):
        self.items = items
        self.open = True
        self.sent_images = []
        self.logs = []
        self.close_sent = False
        self.shutdown_called = False

    def __iter__(self):
        yield from self.items
    
    def send_image(self, img):
        self.sent_images.append(img)

    def send_close(self):
        self.close_sent = True
    
    def shutdown_close(self):
        self.shutdown_called = True

    def send_logging(self, level, msg):
        self.logs.append((level, msg))


class RaisingStreamConnection(FakeStreamConnection):
    """Same as FakeStreamConnection, but raises partway through iteration
    to exercise the exception / shutdown_close path of handle_image_stream."""
    def __init__(self, items, fail_at):
        super().__init__(items)
        self.fail_at = fail_at
 
    def __iter__(self):
        for i, item in enumerate(self.items):
            if i == self.fail_at:
                raise RuntimeError("simulated failure mid-stream")
            yield item


class TestHandleImageStream:

    def test_debug_mode_send_back_original_images(self, server_obj, make_image, monkeypatch):
        monkeypatch.setattr(server_app_module, "Pipeline", FakePipeline)
        images = [make_image(), make_image()]
        connection = FakeStreamConnection(images + [None])
        server_obj.debug = True

        server_obj.handle_image_stream(connection, configJSON=None, metadata="METADATA")

        assert connection.sent_images == images

    def test_debug_mode_activated_via_json(self, server_obj, make_image, monkeypatch):
        monkeypatch.setattr(server_app_module, "Pipeline", FakePipeline)
        images = [make_image(), make_image()]
        connection = FakeStreamConnection(images + [None])
        
        server_obj.handle_image_stream(connection, configJSON={"parameters":{"Debug": True}}, metadata="METADATA")

        assert connection.sent_images == images

    def test_debug_mode_send_image_back_without_pipeline(self, server_obj, make_image, monkeypatch):
        monkeypatch.setattr(server_app_module, "Pipeline", None)
        images = [make_image(), make_image()]
        connection = FakeStreamConnection(images + [None])
        server_obj.handle_image_stream(connection, configJSON={"parameters":{"Debug": True}}, metadata="METADATA")

        assert connection.sent_images == images

    def test_exception_mid_stream_triggers_shutdown_not_normal_close(self, server_obj, make_image, monkeypatch):
        monkeypatch.setattr(server_app_module, "Pipeline", FakePipeline)
        images = [make_image(), make_image(), make_image()]
        connection = RaisingStreamConnection(images, fail_at=1)
 
        server_obj.handle_image_stream(connection, configJSON=None, metadata="METADATA")
 
        assert connection.shutdown_called is True
        assert any(level == "ERROR" for level, _ in connection.logs)
        assert connection.close_sent is True

    def test_raw_kspace_data_triggers_error_handling(self, server_obj, monkeypatch):
        monkeypatch.setattr(server_app_module, "Pipeline", FakePipeline)
        connection = FakeStreamConnection([ismrmrd.Acquisition()] + [None])

        server_obj.handle_image_stream(connection, configJSON=None, metadata="METADATA")

        assert connection.shutdown_called is True
        assert any(level == "ERROR" for level, _ in connection.logs)

    def test_unsupported_message_type_is_logged_as_error(self, server_obj, monkeypatch):
        monkeypatch.setattr(server_app_module, "Pipeline", FakePipeline)
        connection = FakeStreamConnection([object(), None])
 
        server_obj.handle_image_stream(connection, configJSON=None, metadata="METADATA")
 
        assert any(level == "ERROR" for level, _ in connection.logs)
        assert connection.close_sent is True
