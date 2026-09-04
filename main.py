import argparse
import logging
import os
import signal
import sys

from server.server import Server

defaults = {
    'host':             '0.0.0.0',
    'port':             9002,
    'config':           'invertContrast',
    'dirname':          'demo',
    'savedataFolder':   '/tmp/share/saved_data'
}

def main(args: argparse.Namespace):
    """Launch the server"""
    server = Server(args.port, args.host, args.config, args.dirname, args.savedata, args.savedataFolder, args.saveNifti, args.debug)
    
    if args.debug:
        logging.info("Server mode : DEBUG")

    # Trap signal interrupts (e.g. ctrl+c, SIGTERM) and gracefully stop
    def handle_signals(signum, frame) -> None:
        print("Received signal interrupt -- stopping server")
        server.socket.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signals)
    signal.signal(signal.SIGINT,  handle_signals)

    # Allow ctrl+c in Windows
    if sys.platform == 'win32':
        signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    # Start server
    server.serve()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example server for openRecon')

    def dir_path(input_dir: str) -> bool:
        actual_dir = os.path.join('apps/', input_dir)
        if os.path.basename(actual_dir) != input_dir:
            raise ValueError(f"Not a valid path : {input_dir} must not be a nested path")
        
        if not os.path.isdir(actual_dir):
            raise argparse.ArgumentTypeError(f"Not a valid path : {actual_dir}")
        
        return input_dir

    parser.add_argument('-p', '--port',            type=int,            help='Port')
    parser.add_argument('-H', '--host',            type=str,            help='Host')
    parser.add_argument('-c', '--config',          type=str,            help='config module')
    parser.add_argument('-d', '--dirname',         type=dir_path,       help='Application directory name')
    parser.add_argument('-v', '--verbose',         action='store_true', help='Verbose output.')
    parser.add_argument('-l', '--logfile',         type=str,            help='Path to log file')
    parser.add_argument('-s', '--savedata',        action='store_true', help='Save incoming data')
    parser.add_argument('-S', '--savedataFolder',  type=str,            help='Folder to save incoming data')
    parser.add_argument('-n', '--saveNifti',       action='store_true', help='Convert to Nifti and save output data')
    parser.add_argument('-D', '--debug',           action='store_true', help='Debug mode: send back the original images and log all info about them')


    parser.set_defaults(**defaults)

    args = parser.parse_args()

    if args.verbose:
        logLevel = logging.DEBUG
    else:
        logLevel = logging.INFO

    format_log='SERVER : %(levelname)8s: %(message)s'

    if args.logfile:
        print("Logging to file:", args.logfile)

        # Get full path to the log file
        absLogPath = os.path.abspath(args.logfile)
        if not os.path.exists(os.path.dirname(absLogPath)):
            os.makedirs(os.path.dirname(absLogPath))

        logging.basicConfig(
            format=format_log,
            level=logLevel,
            handlers=[
                logging.FileHandler(args.logfile),
                logging.StreamHandler(sys.stdout)
            ],
            force=True
        )
    else:
        print("No logfile provided")
        logging.basicConfig(
            format=format_log,
            level=logLevel,
            handlers=[
                logging.StreamHandler(sys.stdout)
            ],
            force=True
        )

    main(args)
