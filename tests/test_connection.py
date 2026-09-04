import ismrmrd
import pytest

import server.constants as constants
from server.connection import Connection
from conftest import FakeSocket, RaisingSocket

# ---------------------------------------------------------------------------
# read() / peek()
# ---------------------------------------------------------------------------
class TestReadAndPeek:
    def test_read_consumes_bytes(self):
        connection = Connection(FakeSocket(incoming=b"hello world"), savedata=False)
        assert connection.read(5) == b"hello"
        assert connection.read(6) == b" world"

    def test_peek_does_not_consume_bytes(self):
        connection = Connection(FakeSocket(incoming=b"hello world"), savedata=False)
        assert connection.peek(5) == b"hello"
        assert connection.read(6) == b"hello "


# ---------------------------------------------------------------------------
# read_mrd_message_length() / read_mrd_message_identifier() /
# peek_mrd_message_identifier() / unknow_message_identifier()
# ---------------------------------------------------------------------------
class TestMRDMessageIdentifier :
    def test_read_mrd_message_length(self):
        incoming = constants.MrdMessageLength.pack(1234)
        connection = Connection(FakeSocket(incoming=incoming), savedata=False)
        assert connection.read_mrd_message_length() == 1234

    def test_unknow_message_identifier(self):
        with pytest.raises(StopIteration):
            Connection.unknown_message_identifier(9999)

    def test_read_mrd_message_identifier_normal_case(self):
        incoming = constants.MrdMessageIdentifier.pack(constants.MRD_MESSAGE_CLOSE)
        connection = Connection(FakeSocket(incoming=incoming), savedata=False)
        assert connection.read_mrd_message_identifier() == constants.MRD_MESSAGE_CLOSE

    def test_read_mrd_message_identifier_empty_socket_close_connection(self):
        connection = Connection(FakeSocket(incoming=b""), savedata=False)
        result = connection.read_mrd_message_identifier()
        assert result is None
        assert connection.open is False

    def test_read_mrd_message_identifier_connection_reset_close_connection(self):
        connection = Connection(RaisingSocket(), savedata=False)
        result = connection.read_mrd_message_identifier()
        assert result is None
        assert connection.open is False

    def test_peek_mrd_message_identifier_does_not_consume(self):
        incoming = constants.MrdMessageIdentifier.pack(constants.MRD_MESSAGE_TEXT)
        connection = Connection(FakeSocket(incoming=incoming), savedata=False)
        assert connection.peek_mrd_message_identifier() == constants.MRD_MESSAGE_TEXT
        assert connection.read_mrd_message_identifier() == constants.MRD_MESSAGE_TEXT


# ---------------------------------------------------------------------------
# read_config_file() / read_config_text() / read_metadata() / read_text()
# ---------------------------------------------------------------------------
class TestReadMRDMessage:
    def test_read_config_file_strips_null_padding(self):
        incoming = constants.MrdMessageConfigurationFile.pack(b"config_test.xml\0")
        conn = Connection(FakeSocket(incoming=incoming), savedata=False)
        assert conn.read_config_file() == "config_test.xml"

    def test_read_config_text(self):
        content = "some config text\0"
        incoming = constants.MrdMessageLength.pack(len(content.encode())) + content.encode()
        conn = Connection(FakeSocket(incoming=incoming), savedata=False)
        assert conn.read_config_text() == "some config text"

    def test_read_metadata(self):
        content = "<xml>fake header</xml>\0"
        incoming = constants.MrdMessageLength.pack(len(content.encode())) + content.encode()
        conn = Connection(FakeSocket(incoming=incoming), savedata=False)
        assert conn.read_metadata() == "<xml>fake header</xml>"

    def test_read_text(self):
        content = "test\0"
        incoming = constants.MrdMessageLength.pack(len(content.encode())) + content.encode()
        conn = Connection(FakeSocket(incoming=incoming), savedata=False)
        assert conn.read_text() == "test"


# ---------------------------------------------------------------------------
# send_config_file() / send_config_text() / send_metadata() / send_text()
# ---------------------------------------------------------------------------
class TestSendMRDMessage:
    def test_send_config_file_format(self):
        sock = FakeSocket()
        connection = Connection(sock, savedata=False)
        connection.send_config_file("config_test.xml")

        identifier = constants.MrdMessageIdentifier.unpack(bytes(sock.sent[:2]))[0]
        assert identifier == constants.MRD_MESSAGE_CONFIG_FILE

        payload = constants.MrdMessageConfigurationFile.unpack(bytes(sock.sent[2:]))[0]
        assert payload.split(b'\x00', 1)[0] == b"config_test.xml"

    def test_send_config_text(self):
        sock = FakeSocket()
        connection = Connection(sock, savedata=False)
        connection.send_config_text("some config text")

        identifier = constants.MrdMessageIdentifier.unpack(bytes(sock.sent[:2]))[0]
        assert identifier == constants.MRD_MESSAGE_CONFIG_TEXT

        length = constants.MrdMessageLength.unpack(bytes(sock.sent[2:6]))[0]
        assert length == len("some config text\0".encode())

        payload = bytes(sock.sent[6:])
        assert payload == b"some config text\0"

    def test_send_metadata(self):
        sock = FakeSocket()
        connection = Connection(sock, savedata=False)
        connection.send_metadata("<xml>fake header</xml>")

        identifier = constants.MrdMessageIdentifier.unpack(bytes(sock.sent[:2]))[0]
        assert identifier == constants.MRD_MESSAGE_METADATA_XML_TEXT

        length = constants.MrdMessageLength.unpack(bytes(sock.sent[2:6]))[0]
        assert length == len("<xml>fake header</xml>\0".encode())

        payload = bytes(sock.sent[6:])
        assert payload == b"<xml>fake header</xml>\0"

    def test_send_text(self):
        sock = FakeSocket()
        connection = Connection(sock, savedata=False)
        connection.send_text("test")

        identifier = constants.MrdMessageIdentifier.unpack(bytes(sock.sent[:2]))[0]
        assert identifier == constants.MRD_MESSAGE_TEXT

        length = constants.MrdMessageLength.unpack(bytes(sock.sent[2:6]))[0]
        assert length == len("test\0".encode())

        payload = bytes(sock.sent[6:])
        assert payload == b"test\0"


# ---------------------------------------------------------------------------
# shutdown_close() / read_close() / send_close()
# ---------------------------------------------------------------------------
class TestClose:
    def test_shutdown_close(self):
        sock = FakeSocket()
        connection = Connection(sock, savedata=False)
        connection.shutdown_close()

        assert sock.shutdown_called_with is not None
        assert sock.closed is True

    def test_shutdown_close_swallows_error_if_fails(self):
        class BrokenShutdownSocket(FakeSocket):
            def shutdown(self):
                raise OSError("socket already closed")

        sock = BrokenShutdownSocket()
        connection = Connection(sock, savedata=False)
        connection.shutdown_close()

        assert sock.closed is True

    def test_send_close(self):
        sock = FakeSocket()
        connection = Connection(sock, savedata=False)
        connection.send_close()

        identifier = constants.MrdMessageIdentifier.unpack(bytes(sock.sent))[0]
        assert identifier == constants.MRD_MESSAGE_CLOSE

    def test_read_close(self):
        sock = FakeSocket()
        connection = Connection(sock, savedata=False)
        connection.read_close()

        assert connection.open is False

# ---------------------------------------------------------------------------
# read_image() / send_image()
# ---------------------------------------------------------------------------
class TestReadSendImage:
    def test_send_one_image(self, make_image):
        images = make_image()
        sock = FakeSocket()
        connection = Connection(sock, savedata=False)
        connection.send_image(images)

        assert connection.sentImages == 1

    def test_send_image_list(self, make_image):
        images = [make_image(), make_image(), make_image()]
        sock = FakeSocket()
        connection = Connection(sock, savedata=False)
        connection.send_image(images)

        assert connection.sentImages == len(images)

    def test_send_empty_list_sends_nothing(self):
        sock = FakeSocket()
        connection = Connection(sock, savedata=False)
        connection.send_image([])

        assert bytes(sock.sent) == b""
        assert connection.sentImages == 0

    def test_read_image(self, make_image):
        images = make_image()
        out_sock = FakeSocket()
        out_connection = Connection(out_sock, savedata=False)
        out_connection.send_image(images)

        in_sock = FakeSocket(incoming=out_sock.sent)
        in_connection = Connection(in_sock, savedata=False)
        identifier = in_connection.read_mrd_message_identifier()
        assert identifier == constants.MRD_MESSAGE_ISMRMRD_IMAGE

        result = in_connection.read_image()
        assert in_connection.recvImages == 1
        assert result == images
