from dataclasses import dataclass
import argparse
import datetime
import logging
import multiprocessing
import os
import socket
import sys
import time

import h5py
import ismrmrd

from server.connection import Connection

defaults = {
    'filename':           '',
    'in_group':           '',
    'address':            'localhost',
    'port':               9002,
    'outfile':            None,
    'out_group':          str(datetime.datetime.now()),
    'config_json':        None,
    'send_waveforms':     False,
    'verbose':            False,
    'logfile':            '',
}


@dataclass
class DatasetInfo:
    """
    Summary of an inspected MRD HDF5 dataset

    Attributes
    ----------
    in_group : str
        Resolved HDF5 group name containing the MRD data.
    has_raw : bool
        True if the group contains raw k-space acquisitions.
    has_image : bool
        True if the group contains reconstructed images.
    has_waveforms : bool
        True if the group contains waveforms data.
    """
    in_group:      str
    hasRaw:       bool
    hasImage:     bool
    hasWaveforms: bool


def inspect_dataset(filename: str, in_group: str) -> DatasetInfo:
    """
    Open an MRD HDF5 file and determine what data it contains.
    If in_group is empty and the file contains exactly one group,
    that group is selected automatically.

    Parameters
    ----------
    filename : str
        Path to the input HDF5 file.
    in_group : str
        HDF5 group to inspect. May be empty for auto-selection.

    Returns
    -------
    DatasetInfo
        Resolved group name and flags indicating which data types
        are present (raw, image, waveforms).
    """
    with h5py.File(args.filename, 'r') as dset:
        if not dset:
            logging.error(f"Not a valid dataset: {filename}")
            return
        dsetNames = dset.keys()
        logging.info(f"File {filename} contains {len(dset.keys())} groups:")
        print(" ", "\n  ".join(dsetNames))

        if not in_group:
            if len(dset.keys()) == 1:
                in_group = list(dset.keys())[0]
            else:
                logging.error("Input group not specified and multiple groups are present")
                sys.exit(1)

        if in_group not in dset:
            logging.error(f"Could not find group {in_group}")
            sys.exit(1)

        group = dset.get(in_group)

        logging.info(f"Reading data from group '{in_group}' in file '{filename}'")

        # ----- Determine type of data stored --------------------------------------
        # Raw data is stored as:
        #   /group/config      text of recon config parameters (optional)
        #   /group/xml         text of ISMRMRD flexible data header
        #   /group/data        array of IsmsmrdAcquisition data + header
        #   /group/waveforms   array of waveform (e.g. PMU) data

        # Image data is stored as:
        #   /group/config              text of recon config parameters (optional)
        #   /group/xml                 text of ISMRMRD flexible data header (optional)
        #   /group/image_0/data        array of IsmrmrdImage data
        #   /group/image_0/header      array of ImageHeader
        #   /group/image_0/attributes  text of image MetaAttributes
        hasRaw   = 'data' in group
        hasImage = any(key.startswith(("image_", "images_")) for key in group.keys())
        hasWaveforms = 'waveforms' in group

    if (not hasRaw and not hasImage):
        logging.error("File does not contain properly formatted MRD raw or image data")
        sys.exit(1)
    
    info = DatasetInfo(in_group, hasRaw, hasImage, hasWaveforms)
    logging.info("Dataset info: %s", info)
    return info


def load_validate_json(config_json: str) -> str | None:
    """
    Load the JSON config file if it exists.

    Parameters
    ----------
    config_json : str
        Path to the JSON config file.

    Returns
    -------
    str or None
        File contents as a string, or None if the path is empty
        or the file does not exist.
    """
    if not config_json:
        return None
    if not os.path.exists(config_json):
        logging.info(f"JSON config file not found : {config_json}")
        return None

    logging.info(f"Found JSON config file : {config_json}")
    fid = open(config_json, 'r')
    ConfigJSONText = fid.read()
    fid.close()
    
    return ConfigJSONText

#### SENDERS ##################################################################

def send_MRD_Metadata(connection: Connection, dset: ismrmrd.Dataset):
    """
    Read and send the MRD XML header from the dataset.
    If no XML header is found in the dataset, a 'dummy header' is sent
    and a warning is logged.

    Parameters
    ----------
    connection : Connection
        Active MRD connection.
    dset : ismrmrd.Dataset
        Open MRD dataset.

    Returns
    -------
    str
        The XML header string that was sent. Returned so it can be
        written to the output file after the session completes.
    """
    if 'xml' in dset.list():
        xml_header = dset.read_xml_header()
        xml_header = xml_header.decode("utf-8")
    else:
        logging.warning("Could not find MRD metadata xml in file")
        xml_header = "Dummy XML header"
    connection.send_metadata(xml_header)
    return xml_header


def send_additional_config(connection: Connection, dset: ismrmrd.Dataset, config_json: str, filename: str) -> None:
    """
    Send the additional JSON configuration to the server.
    The local JSON file (config_json) takes priority over any
    configAdditional embedded in the MRD dataset. If neither is
    available, nothing is sent.

    Parameters
    ----------
    connection : Connection
        Active outgoing MRD connection.
    dset : ismrmrd.Dataset
        Open MRD dataset, checked for embedded configAdditional.
    config_json : str
        Path to the local JSON config file. Takes priority over
        any config embedded in the dataset.
    filename : str
        Input filename, used in warning messages only.
    """
    ConfigJSONText = load_validate_json(config_json)

    groups = dset.list()
    if ConfigJSONText is not None:
        if ('configAdditional' in groups):
            logging.warning(f"configAdditional found in file {filename}, but is overridden by local file {config_json}!")

        connection.send_text(ConfigJSONText)

    elif ('configAdditional' in groups):
        configAdditionalText = dset._dataset['configAdditional'][0]
        configAdditionalText = configAdditionalText.decode("utf-8")
        connection.send_text(configAdditionalText)

    else:
        # No additional config in local .json file or in MRD file
        logging.info("No additional config to send")


def send_waveforms_data(connection: Connection, dset: ismrmrd.Dataset) -> None:
    """
    Send all waveforms data from the dataset.

    Parameters
    ----------
    connection : Connection
        Active outgoing MRD connection.
    dset : ismrmrd.Dataset
        Open MRD dataset.
    """
    logging.info(f"Sending {dset.number_of_waveforms()} waveform data")

    for idx in range(0, dset.number_of_waveforms()):
        wav = dset.read_waveform(idx)
        try:
            connection.send_waveform(wav)
        except:
            logging.error('Failed to send waveform %d -- aborting!' % idx)
            break


def send_raw_data(connection: Connection, dset: ismrmrd.Dataset) -> None:
    """
    Send all raw k-space acquisitions from the dataset.

    Parameters
    ----------
    connection : Connection
        Active outgoing MRD connection.
    dset : ismrmrd.Dataset
        Open MRD dataset.
    """
    logging.info(f"Sending {dset.number_of_acquisitions()} raw data readouts")

    for idx in range(dset.number_of_acquisitions()):
        acq = dset.read_acquisition(idx)
        try:
            connection.send_acquisition(acq)
        except:
            logging.error('Failed to send acquisition %d -- aborting!' % idx)
            break


def send_image_data(connection: Connection, dset: ismrmrd.Dataset, in_group: str) -> None:
    """
    Send all reconstructed images from the dataset.

    Iterates over all image groups (keys starting with 'image_' or
    'images_') and sends every image within each group.

    Parameters
    ----------
    connection : Connection
        Active outgoing MRD connection.
    dset : ismrmrd.Dataset
        Open MRD dataset.
    in_group : str
        HDF5 group name, used for log messages only.
    """
    logging.info("Starting image data session")
    image_groups = [key for key in dset.list() if (key.startswith('image_') or key.startswith('images_'))]

    for group in image_groups:
        logging.info("Reading images from '/" + in_group + "/" + group + "'")

        for imgNum in range(0, dset.number_of_images(group)):
            image = dset.read_image(group, imgNum)

            if not isinstance(image.attribute_string, str):
                image.attribute_string = image.attribute_string.decode('utf-8')

            logging.debug("Sending image %d of %d", imgNum, dset.number_of_images(group)-1)
            try:
                connection.send_image(image)
            except:
                logging.error('Failed to send image %d -- aborting!' % imgNum)
                break

#### Connection ###############################################################

def connect_to_server(address, port: int) -> socket.socket:
    """
    Open a TCP connection to the MRD server with retries.
    Attempts to connect up to 5 times, waiting 1 second 
    between each attempt.

    Parameters
    ----------
    address : str
        Server hostname or IP address.
    port : int
        Server TCP port.

    Returns
    -------
    socket.socket
        Connected socket.
    """
    # Spawn a thread to connect and handle incoming data
    logging.info(f"Connecting to MRD server at {address}:{port}")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    maxAttempts = 5

    for attempt in range(0, maxAttempts):
        try:
            sock.connect((address, port))
            return sock
        except socket.error as error:
            logging.warning("Failed to connect (%d/%d): %s" % (attempt+1, maxAttempts, error))
            time.sleep(1)

    sock.close()
    logging.error("... Aborting")
    sys.exit(1)


def connection_receive_loop(sock: int, outfile: str, outgroup: str, verbose: bool, logfile: str, recvImages):
    """
    Start a Connection instance, in a separate process, to receive incoming MRD data.

    Parameters
    ----------
    sock : socket.socket
        Connected TCP socket.
    outfile : str
        Path to the HDF5 output file where received data is saved.
    outgroup : str
        HDF5 group name for output data.
    verbose : bool
        If True, log at DEBUG level.
    logfile : str
        Path to a log file. If empty, logs to stdout only.
    recvImages : multiprocessing.Value
        Shared integer counter updated with the number of received images
        when the loop exits.
    """

    if verbose:
        verbosity = logging.DEBUG
    else:
        verbosity = logging.INFO

    if logfile:
        logging.basicConfig(filename=logfile, format='%(asctime)s - %(message)s', level=verbosity)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    else:
        logging.basicConfig(format='%(asctime)s - %(message)s', level=verbosity)

    incoming_connection = Connection(sock, True, outfile, "", outgroup)

    try:
        for msg in incoming_connection:
            if msg is None:
                break
    finally:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except:
            pass
        sock.close()
        logging.debug("Socket closed (reader)")

        # Dataset may not be closed properly if a close message is not received
        try:
            incoming_connection.dset.close()
        except:
            pass

    recvImages.value = incoming_connection.recvImages

###############################################################################

def main(args):

    # --- Output file ---------------------------------------------------------
    if args.outfile is None:
        base, ext = os.path.splitext(args.filename)
        args.outfile = base + '_results' + ext
        logging.info("Output file not specified -- writing results to %s", args.outfile)

    # ----- Validate dataset --------------------------------------------------
    info = inspect_dataset(args.filename, args.in_group)

    # ----- Open connection to server -----------------------------------------
    sock = connect_to_server(args.address, args.port)
    recvImages    = multiprocessing.Value('i', 0)

    process = multiprocessing.Process(
        target=connection_receive_loop, 
        args=(sock, args.outfile, args.out_group, args.verbose, args.logfile, recvImages),
        daemon = True)
    process.start()

    # This connection is only used for outgoing data.  It should not be used for
    # writing to the HDF5 file as multi-threading issues can occur
    connection = Connection(sock, False)

    # --- Check ismrmrd context manager support -------------------------------
    if not (hasattr(ismrmrd.Dataset, '__enter__') and hasattr(ismrmrd.Dataset, '__exit__')):
        raise Exception(
            "Current ismrmrd Python package does not support context manager as required by this code. " 
            "Please update to 1.14.1 or newer")


    # --- Send data -----------------------------------------------------------
    with ismrmrd.Dataset(args.filename, info.in_group, create_if_needed=False) as dset:
        logging.info("Sending config: openrecon")
        connection.send_config_file('openrecon')

        xml_header = send_MRD_Metadata(connection, dset)

        send_additional_config(connection, dset, args.config_json, args.filename)

        # TO-DO: Interleave waveform and other data so they arrive chronologically
        if args.send_waveforms and info.hasWaveforms:
            send_waveforms_data(connection, dset)

        if info.hasRaw:
            send_raw_data(connection, dset)

        if info.hasImage:
            send_image_data(connection, dset, info.in_group)

    # --- Close and wait ------------------------------------------------------
    try:
        connection.send_close()
    except:
        logging.error('Failed to send close message!')

    logging.debug("Waiting for threads to finish")
    process.join()

    sock.close()
    logging.info("Socket closed (writer)")

    # --- Save XML header to output file --------------------------------------
    # Save a copy of the MRD XML header now that the connection thread is 
    # finished with the file
    logging.debug("Writing MRD metadata to file")
    dset = ismrmrd.Dataset(args.outfile, args.out_group)
    dset.write_xml_header(bytes(xml_header, 'utf-8'))
    dset.close()

    logging.info("---------------------- Summary ----------------------")
    logging.info("Sent %5d images        |  Received %5d images", connection.sentImages, recvImages.value)
    logging.info("Results written to %s", args.outfile)
    logging.info("Session complete")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example client for MRD streaming format',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('filename',                                        help='Input file')
    parser.add_argument('-a', '--address',                                 help='Address (hostname) of MRD server')
    parser.add_argument('-p', '--port',               type=int,            help='Port')
    parser.add_argument('-o', '--outfile',                                 help='Output file')
    parser.add_argument('-g', '--in-group',                                help='Input data group')
    parser.add_argument('-G', '--out-group',                               help='Output group name')
    parser.add_argument('-c', '--config-json',                             help='JSON file with the config')
    parser.add_argument('-w', '--send-waveforms',     action='store_true', help='Send waveform (physio) data')
    parser.add_argument('-v', '--verbose',            action='store_true', help='Verbose mode')
    parser.add_argument('-l', '--logfile',            type=str,            help='Path to log file')

    parser.set_defaults(**defaults)

    args = parser.parse_args()

    format_log = 'CLIENT : %(levelname)8s: %(message)s'

    if args.logfile:
        print("Logging to file: ", args.logfile)
        logging.basicConfig(filename=args.logfile, format=format_log, level=logging.WARNING)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    else:
        print("No logfile provided")
        logging.basicConfig(format=format_log, level=logging.WARNING)

    if args.verbose:
        logging.root.setLevel(logging.DEBUG)
    else:
        logging.root.setLevel(logging.INFO)

    main(args)