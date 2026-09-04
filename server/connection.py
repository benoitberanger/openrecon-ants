"""Reads/writes MRD protocol messages over the TCP socket."""

import ctypes
import datetime
import logging
import os
import random
import socket
import threading

import ismrmrd
import numpy as np

import server.constants as constants


class Connection:
    """
    Manages a single MRD client connection over a TCP socket.

    Handles the MRD message protocol by reading and sending different
    message type (config, metadata, images, text, close) according to
    the MRD wire format. Optionally saves all incoming data to an HDF5
    file for debugging purposes.

    Supported message types:

    - ``MRD_MESSAGE_CONFIG_FILE`` (1) : configuration filename
    - ``MRD_MESSAGE_CONFIG_TEXT`` (2) : configuration text contents
    - ``MRD_MESSAGE_METADATA_XML_TEXT`` (3) : MRD XML header
    - ``MRD_MESSAGE_CLOSE`` (4) : end of stream signal
    - ``MRD_MESSAGE_TEXT`` (5) : arbitrary text / logging
    - ``MRD_MESSAGE_ISMRMRD_IMAGE`` (1022) : image data

    Attributes
    ----------
    socket : socket.socket
        Underlying TCP socket for this connection.
    savedata : bool
        If True, incoming data is saved to an HDF5 file.
    savedataFile : str
        Explicit path for the HDF5 save file. If empty, a timestamped
        filename is generated automatically in savedataFolder.
    savedataFolder : str
        Directory where the HDF5 save file is created.
    savedataGroup : str
        HDF5 group name used to store incoming data. Default: 'dataset'.
    dset : ismrmrd.Dataset or None
        Open HDF5 dataset for saving, or None if not yet created.
    open : bool
        True while the connection is active. Set to False on
        MRD_MESSAGE_CLOSE or socket error.
    lock : threading.Lock
        Mutex protecting all socket send operations for thread safety.
    sentImages : int
        Running count of images sent over this connection.
    recvImages : int
        Running count of images received over this connection.
    handlers : dict
        Maps MRD message identifiers to their read handler methods.
    """

    def __init__(self, socket: int, savedata: bool, savedataFile: str = "", savedataFolder: str = "", savedataGroup: str = "dataset") -> None:
        """
        Initialise the connection and register message handlers.

        Parameters
        ----------
        socket : socket.socket
            Accepted TCP socket for this client connection.
        savedata : bool
            If True, save all incoming MRD data to an HDF5 file.
        savedataFile : str, optional
            Explicit HDF5 output path. If empty, a timestamped filename
            is generated in savedataFolder.
        savedataFolder : str, optional
            Directory for the auto-generated HDF5 file. Created if it
            does not exist.
        savedataGroup : str, optional
            HDF5 group name. Default is 'dataset'.
        """
        self.socket         = socket
        self.savedata       = savedata
        self.savedataFile   = savedataFile
        self.savedataFolder = savedataFolder
        self.savedataGroup  = savedataGroup
        self.dset           = None
        self.open           = True
        self.lock           = threading.Lock()
        self.sentImages     = 0
        self.recvImages     = 0
        self.handlers       = {
            constants.MRD_MESSAGE_CONFIG_FILE:         self.read_config_file,
            constants.MRD_MESSAGE_CONFIG_TEXT:         self.read_config_text,
            constants.MRD_MESSAGE_METADATA_XML_TEXT:   self.read_metadata,
            constants.MRD_MESSAGE_CLOSE:               self.read_close,
            constants.MRD_MESSAGE_TEXT:                self.read_text,
            constants.MRD_MESSAGE_ISMRMRD_IMAGE:       self.read_image
        }

    def __iter__(self):
        """
        Iterate over incoming MRD messages until the connection closes.

        Yields each decoded message object in order of arrival.
        Stops when self.open becomes False (MRD_MESSAGE_CLOSE received
        or socket error).
        """
        while self.open:
            yield self.next()

    def __next__(self):
        return self.next()
        
    def read(self, nbytes: int):
        """
        Read exactly nbytes from the socket.

        Parameters
        ----------
        nbytes : int
            Number of bytes to read.

        Returns
        -------
        bytes
            Raw bytes read from the socket.
        """
        return self.socket.recv(nbytes, socket.MSG_WAITALL)

    def peek(self, nbytes: int):
        """
        Read nbytes from the socket without consuming the buffer.

        Parameters
        ----------
        nbytes : int
            Number of bytes to peek.

        Returns
        -------
        bytes
            Raw bytes peeked from the socket buffer.
        """
        return self.socket.recv(nbytes, socket.MSG_PEEK)
    
    def read_mrd_message_length(self) -> int:
        """
        Read and unpack the 4-byte message length field.

        Returns
        -------
        int
            Length in bytes of the following variable-length message body.
        """
        length_bytes = self.read(constants.SIZEOF_MRD_MESSAGE_LENGTH)
        return constants.MrdMessageLength.unpack(length_bytes)[0]

    @staticmethod
    def unknown_message_identifier(identifier: int) -> None:
        """
        Handle an unrecognised MRD message identifier.

        Parameters
        ----------
        identifier : int
            The unrecognised message identifier received from the socket.

        Raises
        ------
        StopIteration
            Always raised to abort iteration on unknown message types.
        """
        logging.error("Received unknown message type: %d", identifier)
        raise StopIteration

    def read_mrd_message_identifier(self) -> int | None:
        """
        Read and unpack the 2-byte message identifier from the socket.

        Returns
        -------
        int
            MRD message identifier.
        None
            If the connection was closed or reset before a full
            identifier could be read.
        """
        try:
            identifier_bytes = self.read(constants.SIZEOF_MRD_MESSAGE_IDENTIFIER)
        except ConnectionResetError:
            logging.error("Connection closed unexpectedly")
            self.open = False
            return

        if (len(identifier_bytes) == 0):
            self.open = False
            return

        return constants.MrdMessageIdentifier.unpack(identifier_bytes)[0]
        
    def peek_mrd_message_identifier(self) -> int | None:
        """
        Peek at the next 2-byte message identifier without consuming it.

        Returns
        -------
        int
            MRD message identifier.
        None
            If the connection was closed or reset.
        """
        try:
            identifier_bytes = self.peek(constants.SIZEOF_MRD_MESSAGE_IDENTIFIER)
        except ConnectionResetError:
            logging.error("Connection closed unexpectedly")
            self.open = False
            return

        if (len(identifier_bytes) == 0):
            self.open = False
            return

        return constants.MrdMessageIdentifier.unpack(identifier_bytes)[0]
        

    def next(self):
        """
        Read the next MRD message and dispatch to the appropriate handler.
        """
        with self.lock:
            id = self.read_mrd_message_identifier()

            if (self.open == False):
                return
                
            handler = self.handlers.get(id, lambda: Connection.unknown_message_identifier(id))
            return handler()
    

    def send_logging(self, level: str, contents: str) -> None:
        """
        Send a log message to the client as an MRD_MESSAGE_TEXT.

        Formats the message as "<level> <contents>" before sending.

        Parameters
        ----------
        level : str
            Log level string, e.g. 'ERROR', 'WARNING', 'INFO'.
        contents : str
            Log message body, typically a traceback or status string.
        """
        try:
            formatted_contents = "%s %s" % (level, contents)
        except:
            logging.warning("Unsupported logging level: " + level)
            formatted_contents = contents

        self.send_text(formatted_contents)
    

    def shutdown_close(self) -> None:
        """
        Forcefully close the socket without sending MRD_MESSAGE_CLOSE.

        Used to signal failure to the client.
        """
        # Encapsulate shutdown in a try block because the socket may have
        # already been closed on the other side
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except:
            pass
        self.socket.close()
        logging.info("Socket closed")

    def create_save_file(self) -> None:
        """
        Create the HDF5 file for saving incoming MRD data.
        """
        if self.savedata is True:
            # Create savedata folder, if necessary
            if ((self.savedataFolder) and (not os.path.exists(self.savedataFolder))):
                os.makedirs(self.savedataFolder)
                logging.debug("Created folder " + self.savedataFolder + " to save incoming data")

            if (self.savedataFile):
                self.mrdFilePath = self.savedataFile
            else:
                self.mrdFilePath = os.path.join(self.savedataFolder, "MRD_input_" + datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S") + "_" + str(random.randint(0, 100)) + ".h5")
                # self.mrdFilePath = os.path.join(self.savedataFolder, "MRD_input_" + datetime.now().strftime("%Y-%m-%d-%H%M%S" + "_" + str(random.randint(0,100)) + ".h5"))

            # Create HDF5 file to store incoming MRD data
            logging.info("Incoming data will be saved to: '%s' in group '%s'", self.mrdFilePath, self.savedataGroup)
            self.dset = ismrmrd.Dataset(self.mrdFilePath, self.savedataGroup)
            self.dset._file.require_group(self.savedataGroup)

    # ----- MRD_MESSAGE_CONFIG_FILE (1) ------------------------------------------
    # This message contains the file name of a configuration file used for 
    # image reconstruction/post-processing.  The file must exist on the server.
    # Message consists of:
    #   ID               (   2 bytes, unsigned short)
    #   Config file name (1024 bytes, char          )
    def read_config_file(self) -> str:
        """
        Read an MRD_MESSAGE_CONFIG_FILE message (1).

        Returns
        -------
        str
            Configuration filename, stripped of null terminators.
        """
        logging.info("<-- Received MRD_MESSAGE_CONFIG_FILE (1)")
        config_file_bytes = self.read(constants.SIZEOF_MRD_MESSAGE_CONFIGURATION_FILE)
        config_file = constants.MrdMessageConfigurationFile.unpack(config_file_bytes)[0]
        config_file = config_file.split(b'\x00',1)[0].decode('utf-8')  # Strip off null terminators in fixed 1024 size

        return config_file
        
    def send_config_file(self, filename: str) -> None:
        """
        Send an MRD_MESSAGE_CONFIG_FILE message (1).

        Parameters
        ----------
        filename : str
            Configuration filename to send. Encoded and padded to
            1024 bytes per the MRD wire format.
        """
        with self.lock:
            logging.info("--> Sending MRD_MESSAGE_CONFIG_FILE (1)")
            self.socket.send(constants.MrdMessageIdentifier.pack(constants.MRD_MESSAGE_CONFIG_FILE))
            self.socket.send(constants.MrdMessageConfigurationFile.pack(filename.encode()))

        
    # ----- MRD_MESSAGE_CONFIG_TEXT (2) ------------------------------------------
    # This message contains the configuration information (text contents) used 
    # for image reconstruction/post-processing.  Text is null-terminated.
    # Message consists of:
    #   ID               (   2 bytes, unsigned short)
    #   Length           (   4 bytes, uint32_t      )
    #   Config text data (  variable, char          )
    def read_config_text(self) -> str:
        """
        Read an MRD_MESSAGE_CONFIG_TEXT message (2).

        Returns
        -------
        str
            Configuration text contents, stripped of null terminator.
        """
        logging.info("<-- Received MRD_MESSAGE_CONFIG_TEXT (2)")
        length = self.read_mrd_message_length()
        config = self.read(length)
        config = config.split(b'\x00',1)[0].decode('utf-8')  # Strip off null terminator

        return config

    def send_config_text(self, contents: str) -> None:
        """
        Send an MRD_MESSAGE_CONFIG_TEXT message (2).

        Parameters
        ----------
        contents : str
            Configuration text to send. A null terminator is appended
            automatically before encoding.
        """
        with self.lock:
            logging.info("--> Sending MRD_MESSAGE_CONFIG_TEXT (2)")
            self.socket.send(constants.MrdMessageIdentifier.pack(constants.MRD_MESSAGE_CONFIG_TEXT))
            contents_with_nul = '%s\0' % contents # Add null terminator
            self.socket.send(constants.MrdMessageLength.pack(len(contents_with_nul.encode())))
            self.socket.send(contents_with_nul.encode())
        

    # ----- MRD_MESSAGE_METADATA_XML_TEXT (3) ------------------------------------
    # This message contains the metadata for the entire dataset, formatted as
    # MRD XML flexible data header text.  Text is null-terminated.
    # Message consists of:
    #   ID               (   2 bytes, unsigned short)
    #   Length           (   4 bytes, uint32_t      )
    #   Text xml data    (  variable, char          )
    def read_metadata(self) -> str:
        """
        Read an MRD_MESSAGE_METADATA_XML_TEXT message (3).

        Contains the MRD XML flexible data header describing the full
        acquisition (encoding, system info, sequence parameters, etc.).

        Returns
        -------
        str
            MRD XML header string, stripped of null terminator.
        """
        logging.info("<-- Received MRD_MESSAGE_METADATA_XML_TEXT (3)")
        length = self.read_mrd_message_length()
        metadata = self.read(length)
        metadata = metadata.split(b'\x00',1)[0].decode('utf-8')  # Strip off null terminator

        return metadata

    def send_metadata(self, contents) -> None:
        """
        Send an MRD_MESSAGE_METADATA_XML_TEXT message (3).

        Parameters
        ----------
        contents : str or ismrmrd.xsd.ismrmrdHeader
            MRD XML header to send. A null terminator is appended
            automatically before encoding.
        """
        with self.lock:
            logging.info("--> Sending MRD_MESSAGE_METADATA_XML_TEXT (3)")
            self.socket.send(constants.MrdMessageIdentifier.pack(constants.MRD_MESSAGE_METADATA_XML_TEXT))
            contents_with_nul = '%s\0' % contents # Add null terminator
            self.socket.send(constants.MrdMessageLength.pack(len(contents_with_nul.encode())))
            self.socket.send(contents_with_nul.encode())


    # ----- MRD_MESSAGE_CLOSE (4) ------------------------------------------------
    def read_close(self) -> None:
        """
        Read an MRD_MESSAGE_CLOSE message (4).
        """

        logging.info("<-- Received MRD_MESSAGE_CLOSE (4)")
        logging.info("    Total received images:       %5d", self.recvImages)
        logging.info("------------------------------------------")

        if self.savedata is True:
            if self.dset is None:
                self.create_save_file()

            logging.debug("Closing file %s", self.dset._file.filename)
            self.dset.close()
            self.dset = None

        self.open = False
        return
        
    def send_close(self) -> None:
        """
        Send an MRD_MESSAGE_CLOSE message (4).

        Signals to the client that all data has been sent and the
        server is done processing.
        """

        with self.lock:
            logging.info("--> Sending MRD_MESSAGE_CLOSE (4)")
            logging.info("  Total sent images:  %5d", self.sentImages)
            self.socket.send(constants.MrdMessageIdentifier.pack(constants.MRD_MESSAGE_CLOSE))


    # ----- MRD_MESSAGE_TEXT (5) -------------------------------------------------
    # This message contains arbitrary text data.
    # Message consists of:
    #   ID               (   2 bytes, unsigned short)
    #   Length           (   4 bytes, uint32_t      )
    #   Text data        (  variable, char          )
    def read_text(self) -> str:
        """
        Read an MRD_MESSAGE_TEXT message (5).

        Returns
        -------
        str
            Decoded text content, stripped of null terminator.
        """
        logging.info("<-- Received MRD_MESSAGE_TEXT (5)")
        length = self.read_mrd_message_length()
        text = self.read(length)
        text = text.split(b'\x00',1)[0].decode('utf-8')  # Strip off null terminator
        return text
        
    def send_text(self, contents: str) -> None:
        """
        Send an MRD_MESSAGE_TEXT message (5).

        Parameters
        ----------
        contents : str
            Text to send. A null terminator is appended automatically.
        """
        with self.lock:
            logging.info("--> Sending MRD_MESSAGE_TEXT (5)")
            logging.info("    %s", contents)
            self.socket.send(constants.MrdMessageIdentifier.pack(constants.MRD_MESSAGE_TEXT))
            contents_with_nul = '%s\0' % contents # Add null terminator
            self.socket.send(constants.MrdMessageLength.pack(len(contents_with_nul.encode())))
            self.socket.send(contents_with_nul.encode())


    # ----- MRD_MESSAGE_ISMRMRD_IMAGE (1022) -----------------------------------
    # This message contains a single [x y z cha] image.
    # Message consists of:
    #   ID               (   2 bytes, unsigned short)
    #   Fixed header     ( 198 bytes, mixed         )
    #   Attribute length (   8 bytes, uint64_t      )
    #   Attribute data   (  variable, char          )
    #   Image data       (  variable, variable      )
    def read_image(self) -> ismrmrd.Image:
        """
        Read an MRD_MESSAGE_ISMRMRD_IMAGE message (1022).

        Reads and assembles the three parts of an MRD image message:
        fixed header, attribute string, and raw pixel data.

        If savedata is True, the image is appended to the HDF5 dataset
        under the group ``image_<image_series_index>``.

        Returns
        -------
        ismrmrd.Image
            Fully populated image object with header, attributes,
            and pixel data.
        """
        self.recvImages += 1
        logging.info("<-- Received MRD_MESSAGE_ISMRMRD_IMAGE (1022)")
        # return ismrmrd.Image.deserialize_from(self.read)

        # Explicit version of deserialize_from() for more verbose debugging
        # logging.debug("   Reading in %d bytes of image header", ctypes.sizeof(ismrmrd.ImageHeader))
        header_bytes = self.read(ctypes.sizeof(ismrmrd.ImageHeader))

        attribute_length_bytes = self.read(ctypes.sizeof(ctypes.c_uint64))
        attribute_length = ctypes.c_uint64.from_buffer_copy(attribute_length_bytes)
        # logging.debug("   Reading in %d bytes of attributes", attribute_length.value)

        attribute_bytes = self.read(attribute_length.value)
        # if (attribute_length.value > 25000):
        #     logging.debug("   Attributes (truncated): %s", attribute_bytes[0:24999].decode('utf-8'))
        # else:
        #     logging.debug("   Attributes: %s", attribute_bytes.decode('utf-8'))

        image = ismrmrd.Image(header_bytes, attribute_bytes.split(b'\x00',1)[0].decode('utf-8'))  # Strip off null terminator

        logging.info("    Image is size %d x %d x %d with %d channels of type %s", 
                     image.getHead().matrix_size[0], 
                     image.getHead().matrix_size[1], 
                     image.getHead().matrix_size[2], 
                     image.channels, 
                     image.data.dtype)
        def calculate_number_of_entries(nchannels, xs, ys, zs):
            return nchannels * xs * ys * zs

        nentries = calculate_number_of_entries(image.channels, *image.getHead().matrix_size)
        nbytes = nentries * image.data.dtype.itemsize

        logging.debug("Reading in %d bytes of image data", nbytes)
        data_bytes = self.read(nbytes)

        image.data.ravel()[:] = np.frombuffer(data_bytes, dtype=image.data.dtype)

        if self.savedata is True:
            if self.dset is None:
                self.create_save_file()
            self.dset.append_image("image_%d" % image.image_series_index, image)

        return image
        
    def send_image(self, images: ismrmrd.Image | list[ismrmrd.Image]) -> None:
        """
        Send one or more MRD_MESSAGE_ISMRMRD_IMAGE messages (1022).

        Parameters
        ----------
        images : ismrmrd.Image or list of ismrmrd.Image
            Image or images to send.
        """
        with self.lock:
            if not isinstance(images, list):
                images = [images]
            
            logging.info("--> Sending MRD_MESSAGE_ISMRMRD_IMAGE (1022) (%d images)", len(images))
            
            if len(images) == 0:
                return

            for image in images:
                if image is None:
                    continue

                self.sentImages += 1
                self.socket.send(constants.MrdMessageIdentifier.pack(constants.MRD_MESSAGE_ISMRMRD_IMAGE))
                image.serialize_into(self.socket.send)

            # Explicit version of serialize_into() for more verbose debugging
            # self.socket.send(image.getHead())
            # self.socket.send(constants.MrdMessageAttribLength.pack(len(image.attribute_string)))
            # self.socket.send(bytes(image.attribute_string, 'utf-8'))
            # self.socket.send(bytes(image.data))
