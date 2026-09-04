"""Manages the connection lifecycle and dispatches incoming MRD data to the pipeline."""

import gc
import json
import logging
import os
import signal
import socket
import traceback

import ismrmrd

from utils.utils import check_OR_arguments
from utils.memory import log_memory, log_memory_delta
from server.debug import send_back_debug
from server.pipeline import Pipeline
from server.connection import Connection
import server.constants as constants

class Server:
    """
    MRD image processing server

    Manages the connection lifecycle and dispatches incoming MRD data
    to the appropriate handler. Currently supports image data only
    (raw k-space and waveform data are not supported).

    Attributes
    ----------
    connection : Connection
        Active MRD connection used to receive and send data.
    app_config : str
        Name of the application module to load in the pipeline.
        (e.g. 'invertcontrast').
    app_directory : str
        Python package directory containing the application module.
        (e.g. 'app').
    debug : bool
        If True, images are sent back unmodified with diagnostic info
        logged for each image. No processing is performed.
    """

    def __init__(self, port: int, address: str, app_config: str, app_directory: str, savedata: bool, savedataFolder: str, saveNifti: bool, debug: bool) -> None:
        """
        Initialise and bind the server socket.

        Parameters
        ----------
        port : int
            TCP port to listen on.
        address : str
            IP address to bind to.
        app_config : str
            Name of the application module to load in the pipeline.
        app_directory : str
            Python package containing the application module.
        savedata : bool
            If True, save incoming MRD data to disk.
        savedataFolder : str
            Path to save the incoming MRD data to disk.
        saveNifti : bool
            If True, convert to Nifti and save output MRD data to disk.
        debug : bool
            If True, enable debug mode at startup.
        """
        logging.info(f"Starting server and listening for data at {address}:{port}")

        self.app_config = app_config
        self.app_directory = app_directory
        self.savedata = savedata
        self.saveFolder = savedataFolder
        self.save_nifti = saveNifti
        self.debug = debug

        if not debug:
            self.check_app_files(app_config, app_directory)

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((address, port))


    @staticmethod
    def check_app_files(app_config: str, app_directory: str) -> None:
        """
        Verify that the processing module file exist.

        Parameters
        ----------
        app_config : str
            Name of the application module (without .py extension).
        app_directory : str
            Python package directory containing the application module.

        Raises
        ------
        FileNotFoundError
            If the <app_config>.py file does not exist.
        """

        module_path = os.path.join('apps/', app_directory, app_config + '.py')
        if not os.path.isfile(module_path):
            raise FileNotFoundError(f"Processing module not found: '{module_path}'")


    def serve(self) -> None:
        """
        Start the server main loop and accept incoming connections.

        Listens indefinitely for incoming TCP connections and calls
        handle() for each one.
        """

        logging.debug("Serving... ")
        self.socket.listen(0)

        while True:
            try:
                signal.siginterrupt(signal.SIGTERM, True)
                signal.siginterrupt(signal.SIGINT, True)
            except AttributeError:
                # signal.siginterrupt is not available in Windows
                pass

            sock, (remote_addr, remote_port) = self.socket.accept()

            logging.info(f"Accepting connection from: {remote_addr}:{remote_port}")

            self.handle(sock)


    def handle_metadata(self, connection: Connection) -> ismrmrd.xsd.ismrmrdHeader | str | None :
        """
        Receive and parse the MRD XML header from the connection.

        The MRD header is the first message expected after the config.
        If the connection is closed before any header is received, 
        returns None.

        Parameters
        ----------
        connection : Connection
            Active MRD connection.

        Returns
        -------
        ismrmrd.xsd.ismrmrdHeader or str or None
            Parsed MRD header object if the XML is valid.
            Raw XML string if parsing failed.
            None if the connection was closed before any data arrived.
        """
        metadata_xml = next(connection)

         # Break if no MRD header was received before a close message (e.g. Gadgetron dependency query)
        if ((metadata_xml is None) & (connection.open is False)):
            logging.info("Connection closed without an MRD header received")
            return
        
        logging.info("XML Metadata: %s", metadata_xml)
        try:
            metadata = ismrmrd.xsd.CreateFromDocument(metadata_xml)
            if (metadata.acquisitionSystemInformation.systemFieldStrength_T != None):
                logging.info("Data is from a %s %s at %1.1fT", metadata.acquisitionSystemInformation.systemVendor, metadata.acquisitionSystemInformation.systemModel, metadata.acquisitionSystemInformation.systemFieldStrength_T)
        except:
            logging.warning("Metadata is not a valid MRD XML structure.  Passing on metadata as text")
            metadata = metadata_xml

        return metadata


    def handleJSON(self, connection: Connection) -> dict | None:
        """
        Receive and parse an optional JSON configuration message.

        Parameters
        ----------
        connection : Connection
            Active MRD connection

        Returns
        -------
        dict or None
            Parsed JSON configuration dict if a text message was received
            and successfully parsed, otherwise None.
        """
        if connection.peek_mrd_message_identifier() == constants.MRD_MESSAGE_TEXT:
            configAdditionalText = next(connection)
            logging.info(f"Received additional config text: {configAdditionalText}")
            try:
                configAdditional = json.loads(configAdditionalText)
            except Exception as e:
                logging.error(f"Failed to parse as JSON: {e}")
                return None
        else:
            return None
        
        return configAdditional


    def handle_image_stream(self, connection: Connection, configJSON: dict | None, metadata: ismrmrd.xsd.ismrmrdHeader | str) -> None:
        """
        Receive a stream of MRD images, process them via the pipeline,
        and send results back over the connection.

        Images are accumulated into a group until the connection is
        closed or a null item is received, then forwarded to the pipeline.

        Debug mode can be activated at instantiation or at runtime via
        the 'Debug' key in configJSON. In debug mode each image is sent
        back unmodified with its metadata logged, therefore the pipeline 
        is not called.

        Parameters
        ----------
        connection : Connection
            Active MRD connection.
        configJSON : dict or None
            JSON configuration sent by the client. May be None if no
            configuration was provided.
        metadata : ismrmrd.xsd.ismrmrdHeader or str
            MRD formatted header describing the acquisition. May be a
            raw string if header conversion failed upstream.
        """

        try:
            logging.info("Incoming dataset contains %d encodings", len(metadata.encoding))
            logging.info("First encoding is of type '%s', with a matrix size of (%s x %s x %s) and a field of view of (%s x %s x %s)mm^3", 
                metadata.encoding[0].trajectory, 
                metadata.encoding[0].encodedSpace.matrixSize.x, 
                metadata.encoding[0].encodedSpace.matrixSize.y, 
                metadata.encoding[0].encodedSpace.matrixSize.z, 
                metadata.encoding[0].encodedSpace.fieldOfView_mm.x, 
                metadata.encoding[0].encodedSpace.fieldOfView_mm.y, 
                metadata.encoding[0].encodedSpace.fieldOfView_mm.z)

        except:
            logging.info("Improperly formatted metadata: \n%s", metadata)

        # Check if the debug mode is enabled via JSON
        if (not self.debug) :
            self.debug = check_OR_arguments(configJSON, arg_name='Debug', arg_type=bool, arg_default=False)
        
        # Initialize the pipeline (only required without the debug mode)
        pipeline = None
        if not self.debug:
            pipeline = Pipeline(connection, self.app_config, self.app_directory, self.save_nifti)

        # Continuously parse incoming data parsed from MRD messages
        imgGroup = []
        mem_start = log_memory("handle_image_stream", "Beginning")
        try:
            for item in connection:

                # When the connection is closed, all images have been received
                if not connection.open :
                    logging.info("Exit because connection closed. All images have been received")
                    break

                # ----------------------------------------------------------
                # Raw k-space data messages
                # ----------------------------------------------------------
                if isinstance(item, ismrmrd.Acquisition):
                    logging.error("Raw k-space data is not supported by this module")
                    raise Exception("Raw k-space data is not supported by this module")

                # ----------------------------------------------------------
                # Image data messages
                # ----------------------------------------------------------
                elif isinstance(item, ismrmrd.Image):
                    # If the debug mode is activated, send back original images
                    # with image infos displayed in the log
                    if self.debug:
                        send_back_debug(item, connection)
                    else:
                        imgGroup.append(item)
                        # Log every 50 images to avoid spaming
                        if len(imgGroup) % 50 == 0:
                            log_memory_delta("handle_image_stream", f"{len(imgGroup)} images accumulated", mem_start)

                elif item is None:
                    logging.info("Exit because null item received")
                    break

                else:
                    raise Exception("Unsupported data type %s", type(item).__name__)

            # Process images data
            if imgGroup:
                log_memory_delta("handle_image_stream", f"All item received — {len(imgGroup)} images", mem_start)
                logging.info("---------- PROCESSING IMAGES ----------")
                pipeline.run(imgGroup, configJSON, metadata)
                del imgGroup
                gc.collect()
                log_memory_delta("handle_image_stream", "After send", mem_start)
            
        except Exception as e:
            logging.error(traceback.format_exc())
            connection.send_logging("ERROR", traceback.format_exc())
            
            # Close connection without sending MRD_MESSAGE_CLOSE message to signal failure
            connection.shutdown_close()

        finally:
            try:
                connection.send_close()
            except:
                logging.error("Failed to send close message!")
            log_memory_delta("handle_image_stream", "End", mem_start)


    def handle(self, sock: int)-> None:
        """
        Handle a single client connection and dispatch data to the
        appropriate handler.

        Reads the connection handshake in order:

        1. **Config** : first message, identifies the processing mode.
           Only ``"openrecon"`` triggers image processing, any other 
           config value causes the connection to be drained and
           closed without processing.
        2. **Metadata** : MRD XML header
        3. **JSON config** : optional additional parameters selected by the user thanks to the UI.

        Then dispatches traffic to handle_image_stream().

        Parameters
        ----------
        sock : socket.socket
            Accepted client socket.
        """

        try:
            connection = Connection(sock, savedata=self.savedata, savedataFolder=self.saveFolder)

            # First message is the config (file or text)
            # With OpenRecon it should be "openrecon"
            config = next(connection)

            if ((config is None) & (connection.open is False)):
                logging.info("Connection closed without any data received")
                return
            logging.info(f"Received config: {config}")
            
            # Second messages is the metadata
            metadata = self.handle_metadata(connection)
            if not metadata:
                return
            
            # Additional config parameters passed through a JSON text message
            configJSON = self.handleJSON(connection)

            # If the config is openrecon load the app config
            # Else do nothing with the data
            if (config == "openrecon"):
                self.handle_image_stream(connection, configJSON, metadata)
            else :
                logging.info(f"No openrecon config requested : {config}")
                try:
                    for msg in connection:
                        if (not connection.open) or (msg is None):
                            break
                finally:
                    connection.send_close()

        except Exception as e:
            logging.exception(e)
        
        finally:
            connection.shutdown_close()

            if connection.savedata is True:
                self.close_and_renameSaveFile(connection)


    def close_and_renameSaveFile(self, connection: Connection) -> None :
        """
        Close the MRD save file and optionally rename it using the protocol name.

        If the connection was configured to auto-generate a save filename
        (i.e. savedataFile was not explicitly set), attempts to rename the
        file by replacing the ``MRD_input_`` prefix with the protocol name
        read from the MRD XML header.

        Parameters
        ----------
        connection : Connection
            Active or recently closed MRD connection. The following
            attributes are used:

            - ``connection.dset`` : HDF5 dataset handle to close.
            - ``connection.savedataFile`` : explicit save path if set, empty string if auto-generated.
            - ``connection.mrdFilePath`` : current path of the save file, updated in-place if renamed.
            - ``connection.savedataGroup`` : HDF5 group name used to read the XML header for the protocol name.
        """
        try:
            connection.dset.close()
        except:
            pass

        # --- Rename only if the filename was auto-generated ------------------
        if (connection.savedataFile != ""):
            if connection.mrdFilePath is not None:
                logging.info("Incoming data saved at %s", connection.mrdFilePath)
            return
        
        try:
            # Ensure ismrmrd package has a context manager
            if not (hasattr(ismrmrd.Dataset, '__enter__') and hasattr(ismrmrd.Dataset, '__exit__')):
                raise Exception("Current ismrmrd Python package does not support context manager as required by this code.  Please update to 1.14.1 or newer")

            # Rename the saved file to use the protocol name
            with ismrmrd.Dataset(connection.mrdFilePath, connection.savedataGroup, False) as dset:
                groups = dset.list()

                if ('xml' not in groups):
                    return
                xml_header = dset.read_xml_header()
                xml_header = xml_header.decode("utf-8")
                mrdHead = ismrmrd.xsd.CreateFromDocument(xml_header)
                protocol = mrdHead.measurementInformation.protocolName

            if (protocol ):
                newFilePath = connection.mrdFilePath.replace("MRD_input_", protocol + "_")
                os.rename(connection.mrdFilePath, newFilePath)
                connection.mrdFilePath = newFilePath
                logging.info("Save file renamed to %s", newFilePath)
        
        except Exception as e:
            logging.debug(f"Could not rename save file : {e}")
        
        if connection.mrdFilePath is not None:
            logging.info("Incoming data saved at %s", connection.mrdFilePath)
