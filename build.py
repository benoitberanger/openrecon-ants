import argparse
import logging
import os
import glob
import shutil
import pprint
import sys
import subprocess
import re
import json
import base64
# import urllib

import jsonschema


def print_section(name: str) -> None:
    """Display section name"""
    DEBUG_LINE = '#'*40
    print('')
    print(DEBUG_LINE)
    print(f'# {name}')
    print(DEBUG_LINE)


def check_docker_version() -> bool:
    """
    Verify that the installed Docker version is compatible with Siemens OpenRecon.
    (Siemens OpenRecon supports Docker versions up to (but not including) 25)

    Returns
    -------
    bool
        True if the Docker version is compatible, False otherwise.
    """
    logger = logging.getLogger()

    result = subprocess.run(['docker', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    version_output = result.stdout.strip()
    logger.debug(version_output)
    pattern = r"Docker version (\d+\.\d+\.\d+),"
    matches = re.findall(pattern, version_output)
    if not len(matches):
        logger.critical('Could not find Docker version')
        return False
    docker_version = matches[0].split('.')[0] # 20.21.22 -> 20
    maximum_docker_version = 25
    if int(docker_version) >= maximum_docker_version:
        logger.info(f'Docker version {docker_version} is too high. Siemens allows maximum version: {maximum_docker_version}')
        return False
    logger.info(f'Docker version {docker_version}<={maximum_docker_version} is ok')
    return True


def check_dependencies(dependencies_name: str) -> None:
    """
    Check if the specified dependencies is installed in the system.

    Parameters
    ----------
    dependencies_name: str
        e.g. docker, git, zip
    """
    logger = logging.getLogger()

    path = shutil.which(dependencies_name)
    if path:
        logger.info(f'`{dependencies_name}` is installed')
    else:
        logger.critical(f'`{dependencies_name}` does not seem to be present in the system')
        sys.exit(1)


def check_target_dir(target_path: str) -> dict:
    """
    Check that the application directory contains all required files.

    Expected files:
        - application.Dockerfile            : Final dockerfile
        - <process_name>_json_ui.json       : OpenRecon UI parameter definition
        - OpenReconSchema_*.json            : JSON schema for validation
        - <process_name>.py (process file)  : main processing script
    
    Parameters
    ----------
    target_path : str
        Absolute path to the application directory.

    Returns
    -------
    dict with the following structure::

        {
            'name': {
                'process': str,   # e.g. 'invertcontrast'
                'schema':  str,   # e.g. 'OpenReconSchema_1.1.0'
            },
            'path': {
                'process': str,   # path to invertcontrast.py
                'ui_json': str,   # path to invertcontrast_json_ui.json
                'schema':  str,   # path to OpenReconSchema_1.1.0.json
            }
        }
    """
    logger = logging.getLogger()

    # Dockerfile to find
    dockerfile_path = os.path.join(target_path, 'application.Dockerfile')
    if os.path.exists(dockerfile_path):
        logger.info(f'Found application.Dockerfile : {dockerfile_path}')
    else:
        logger.error(f'application.Dockerfile not found : {dockerfile_path}')
        sys.exit(1)

    # JSON files to find
    json_ui_pattern = '*_json_ui.json'
    schema_pattern  = 'OpenReconSchema_*.json'
    json_ui_list = glob.glob(os.path.join(target_path, json_ui_pattern))
    schema_list  = glob.glob(os.path.join(target_path, schema_pattern ))

    if len(json_ui_list) == 1:
        logger.info(f'Found JSON UI file : {json_ui_list[0]}')
    else:
        logger.error(f'Found {len(json_ui_list)}/1 JSON UI file with pattern {json_ui_pattern} in {target_path}')
        sys.exit(1)

    if len(schema_list) == 1:
        logger.info(f'Found OpenReconSchema file : {schema_list[0]}')
    else:
        logger.error(f'Found {len(schema_list)}/1 OpenReconSchema file with pattern {schema_pattern} in {target_path}')
        sys.exit(1)

    process_name = os.path.splitext( os.path.basename(json_ui_list[0]) )[0].replace('_json_ui', '')
    schema_name  = os.path.splitext( os.path.basename( schema_list[0]) )[0]

    # fetch the .py process
    process_path = os.path.join(target_path, f'{process_name}.py')
    if os.path.exists(process_path):
        logger.info(f'Found .py process file : {process_path}')
       
    else:
        logger.error(f'.py process not found : {process_path}')
        sys.exit(1)

    # create a dict to save the necessary paths and names
    target_data = {
        'name' : {
            'process': process_name,
            'schema' :  schema_name,
        },
        'path': {
            'process' : process_path,
            'ui_json' : json_ui_list[0],
            'schema'  :  schema_list[0],
        }
    }
    pprint.pprint(target_data, sort_dicts=False)

    return target_data


def build_base_image(dockerfile_path: str) -> None:
    """
    Build the base Docker image ``python-openrecon-server``.

    The base image contains all ISMRMRD Python dependencies and serves
    as the starting point for the application-specific image.

    Parameters
    ----------
    dockerfile_path : str
        Path to the MRD.Dockerfile.
    """
    logger = logging.getLogger()

    result = subprocess.run(['docker', 'images', 'python-openrecon-server'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    output = result.stdout.strip()
    if 'python-openrecon-server' in output:
        logger.info('docker image `python-openrecon-server` already built')
        return

    # build docker image for python-openrecon-server
    # this image is the starting point, that will be refined later
    logger.info('building docker image `python-openrecon-server`')
    subprocess.run(['docker', 'build', '--tag', 'python-openrecon-server', '--file', dockerfile_path, './'], check=True)


def check_json_format(json_content: dict, target_data: dict) -> bool:
    """
    Validate JSON object for the application against the JSON schema reference

    Parameters
    ----------
    json_content : any
        Parsed JSON UI content.
    target_data : dict
        Target directory data as returned by check_target_dir().

    Returns
    -------
    bool
        True if the JSON is valid against the schema, False otherwise.
    """
    logger = logging.getLogger()

    logger.info(f"load JSON Schema : {target_data['path']['schema']}")
    with open(file=target_data['path']['schema'], mode='r') as fid:
        schema_content = json.load(fp=fid)

    validator = jsonschema.Draft7Validator(schema=schema_content)

    errors = list(validator.iter_errors(instance=json_content))
    if errors:
        logger.error('our Json vs. Schema errors :')
        for error in errors:
                logger.error(error)
        return False
    
    logger.info(f'No error in out JSON compared against the Schema')
    return True


def prepare_infos(json_content, build_path: str) -> dict:
    """
    Prepare output files names (Dockerfile, .tar, .zip)

    Parameters
    ----------
    json_content : any
        Parsed JSON UI content.
    build_path : str
        Absolute path to the build output directory.

    Returns
    -------
    dict with the following structure::

        {
            'info': {
                'version': str,
                'vendor':  str,
                'name':    str,
            },
            'name': {
                'docker': str,   # e.g. 'OpenRecon_ICM_InvertContrast:V1.0.0'
                'base':   str,   # e.g. 'OpenRecon_ICM_InvertContrast_V1.0.0'
            },
            'path': {
                'docker': str,   # path to generated .Dockerfile
                'tar':    str,   # path to output .tar
                'pdf':    str,   # path to output .pdf
                'zip':    str,   # path to output .zip
            }
        }
    """
    logger = logging.getLogger()

    # prep build dir
    if os.path.exists(build_path):
        logger.info(f'`build` dir found : {build_path}')
    else:
        if not os.path.exists("builds"):
            os.mkdir("builds")
        os.mkdir(build_path)
        logger.info(f'`build` dir created : {build_path}')
    
    version = json_content['general']['version']
    vendor  = json_content['general']['vendor' ]
    name    = json_content['general']['id'     ]

    # other file/path
    base_name = f'OpenRecon_{vendor}_{name}_V{version}'

    build_data = {
        'info': {
            'version'   : version,
            'vendor'    : vendor,
            'name'      : name
        },
        'name': {
            'docker': f'OpenRecon_{vendor}_{name}:V{version}'.lower(),
            'base'  : f'OpenRecon_{vendor}_{name}_V{version}'
        },
        'path': {
            'docker' : os.path.join(build_path, f"{base_name}.Dockerfile"),
            'tar'    : os.path.join(build_path, f"{base_name}.tar"),
            'pdf'    : os.path.join(build_path, f"{base_name}.pdf"),
            'zip'    : os.path.join(build_path, f"{base_name}.zip")
        }
    }

    pprint.pprint(build_data, sort_dicts=False)

    return build_data


def write_dockerfile(json_content, cmdline: str, docker_path: str, build_data: dict) -> None:
    """
    Generate the final application Dockerfile for OpenRecon.

    Copies the base application Dockerfile and appends two directives:

    - ``CMD`` — the server launch command line.
    - ``LABEL`` — the OpenRecon metadata label, containing the full
      JSON UI content encoded as Base64 (required by Siemens OpenRecon).

    Parameters
    ----------
    json_content : any
        Parsed JSON UI content to embed in the LABEL directive.
    cmdline : str
        Shell command used to start the MRD server inside the container.
    docker_path : str
        Path to the source ``application.Dockerfile`` in the app directory.
    build_data : dict
        Dict containing the path for the generated Dockerfile in the build 
        directory, and the name and version of the application.
    """
    logger = logging.getLogger()

    # encoded the json content in base 64
    encoded_json_content = base64.b64encode((json.dumps(obj=json_content,indent=2)).encode('utf-8')).decode('utf-8')

    # write the Dockerfile content
    logger.info(f"Write `build` Dockerfile : {build_data['path']['docker']}")
    dockerfile_content = [
        f'',
        f'# Version',
        f'LABEL "{build_data['info']['name']}_version"="{build_data['info']['version']}"',
        f'',
        f'# CMD line',
        f'CMD [ "/bin/bash", "-c", "/usr/sbin/ldconfig && {cmdline}" ] ',
        f'',
        f'# mandatory for OpenRecon (see OR documentation)',
        f'LABEL "com.siemens-healthineers.magneticresonance.openrecon.metadata:1.1.0"="{encoded_json_content}"',
        f'',
    ]
    dockerfile_content = "\n".join(dockerfile_content)

    shutil.copy(docker_path, build_data['path']['docker'])

    with open(file=build_data['path']['docker'], mode='a') as fid:
        fid.writelines(dockerfile_content)


def create_pdf(file_path: str, lines_of_text: list[str]) -> None:
    """
    Generate a minimal pdf with information about the app

    Parameters
    ----------
    file_path : str
        Output path for the generated PDF file.
    lines_of_text : list[str]
        Lines of text to write in the PDF file.
    """
    pdf_header = b'%PDF-1.4\n'
    
    objects = []
    objects.append(b'1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n') # Object 1: Catalog
    objects.append(b'2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n') # Object 2: Pages
    objects.append(b'3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n') # Object 3: Page
    
    # Object 4: Page content
    content_stream = "BT /F1 24 Tf 100 750 Td" 
    for line in lines_of_text:
        content_stream += f" ({line}) Tj 0 -30 Td"
    content_stream += " ET"
    objects.append(f'4 0 obj\n<< /Length {len(content_stream)} >>\nstream\n{content_stream}\nendstream\nendobj\n'.encode())
    objects.append(b'5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n') # Object 5: Font
    
    pdf_body = b''.join(objects)
    
    # Cross-reference table
    xref_offset = len(pdf_header)
    xref = b'xref\n0 6\n0000000000 65535 f \n'
    xref_entry_offsets = [xref_offset]
    for obj in objects:
        xref_entry_offsets.append(xref_entry_offsets[-1] + len(obj))
    for offset in xref_entry_offsets:
        xref += f'{offset:010} 00000 n \n'.encode()
    
    trailer = f'trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref_entry_offsets[-1]}\n%%EOF'.encode()
    
    # Write
    with open(file_path, 'wb') as f:
        f.write(pdf_header)
        f.write(pdf_body)
        f.write(xref)
        f.write(trailer)


def packaging_OR_image(build_data: dict, build_path: str, pdf_file: str | None) -> None:
    """
    Export the OpenRecon Docker image and package it as a .zip file.

    Performs three steps in order:
    1. Save the Docker image to a ``.tar`` archive via ``docker save``.
    2. Generate a ``.pdf`` documentation file with vendor/name/version info.
    3. Bundle the ``.tar`` and ``.pdf`` into a single ``.zip`` file.

    Parameters
    ----------
    build_data: dict
        Build data as returned by prepare_infos().
    build_path : str
        Absolute path to the build output directory.
    pdf_file: str or None
        Path the a pdf file.
    """
    logger = logging.getLogger()

    cwd = os.getcwd()
    # build_path = os.path.join(cwd, 'builds')

    # save docker image in a .tar
    logger.info(f"(1/2) saving image `{build_data['name']['docker']}` in a .tar {build_data['path']['tar']}")
    subprocess.run(['docker', 'save', '-o', build_data['path']['tar'], build_data['name']['docker']], check=True)
    logger.info(f"(2/2) saving image DONE")

    if pdf_file:
        pdf_path = os.path.join(cwd, pdf_file)

    if (not pdf_file) or (not os.path.exists(pdf_path)):
        logging.info(f"PDF file {pdf_file} not found. Generation of a minimal PDF.")
        pdf_path = build_data['path']['pdf']
        # generate PDF
        lines = [
            f'vendor={build_data['info']['vendor']}',
            f'name={build_data['info']['name']}',
            f'version={build_data['info']['version']}',
        ]
        logger.info(f"write PDF file : {build_data['path']['pdf']}")
        create_pdf(file_path=build_data['path']['pdf'], lines_of_text=lines)

    # save everything in a ZIP file
    logger.info(f"(1/2) zip all files : {build_data['path']['zip']}")
    subprocess.run(['zip', '-j', build_data['name']['base']+'.zip', build_data['path']['tar'], pdf_path], check=True, cwd=build_path)
    logger.info(f"(2/2) zip all files DONE")


def main(args: argparse.Namespace):

    #############
    ### setup ###
    #############

    logger = logging.getLogger()

    print_section('START')
    logger.info(f'Start of {os.path.basename(__file__)}')
    logger.warning('Untill the BUILD part, there is a "skip if already done" feature')
    cwd = os.getcwd()
    logger.info(f'Current working directory : {cwd}')

    # check if all system programs are here
    print_section('SYSTEM DEPENDENCIES')
    check_dependencies('zip')
    check_dependencies('git')
    check_dependencies('docker')
    if not check_docker_version():
        sys.exit(1)

    # check if the necessary file are present in the target dir
    target_path = os.path.join(cwd, 'apps/', args.dirname)
    print_section(f'Check `target` dir and its content : {target_path}')
    target_data = check_target_dir(target_path)


    #############
    ### build ###
    #############

    # Build base docker image with ISMRD
    print_section('BUILD SERVER')
    dockerfile_path = os.path.join(cwd, 'MRD.Dockerfile')
    build_base_image(dockerfile_path)

    print_section('BUILD')
    logger.warning('From now on, all steps will not have a "skip if already done" feature')

    # prep build dir
    build_path = os.path.join(cwd, 'builds', args.dirname)

    # load JSON UI
    logger.info(f"load UI JSON content : {target_data['path']['ui_json']}")
    with open(target_data['path']['ui_json'], 'r') as fid:
        json_content = json.load(fid)
    
    # Check if our updated JSON is ok
    if not check_json_format(json_content, target_data):
        sys.exit(1)

    # prepare infos 
    # prepare the commande line for the Dockerfile
    if args.debug:
        cmdline  = f'exec python3 main.py -v --debug -H=0.0.0.0 -p=9002 -l=/tmp/python-openrecon-server.log --config={target_data['name']['process']} --dirname={args.dirname} --savedata --savedataFolder /tmp/or'
    else:
        cmdline  = f'exec python3 main.py -v -H=0.0.0.0 -p=9002 -l=/tmp/python-openrecon-server.log --config={target_data['name']['process']} --dirname={args.dirname}'

    build_data = prepare_infos(json_content, build_path)

    # Write the Dockerfile
    app_docker_path = os.path.join(target_path, 'application.Dockerfile')
    write_dockerfile(json_content, cmdline, app_docker_path, build_data)

    # build docker image
    logger.info(f"building docker image `{build_data['name']['docker']}` from Docker file {build_data['path']['docker']}")
    subprocess.run(['docker', 'build', '--tag', build_data['name']['docker'], '--file', build_data['path']['docker'], cwd], check=True)

    # generate a pdf documentation and save the docker image and its doc in a .zip
    if not args.nopackage : 
        packaging_OR_image(build_data, build_path, args.pdf_file)

    # END
    print_section('All done !')
    sys.exit(0)


if __name__ == '__main__':

    # setup logging
    logging.basicConfig(
        level=logging.DEBUG,
        format=f"%(levelname)8s:%(funcName)15s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog            = 'build',
        description     = 'Build OpenRecon app',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )

    def dir_path(input_dir: str) -> bool:
        actual_dir = os.path.join('apps/', input_dir)
        if os.path.basename(actual_dir) != input_dir:
            raise ValueError(f"Not a valid path : {input_dir} must not be a nested path")
        
        if not os.path.isdir(actual_dir):
            raise argparse.ArgumentTypeError(f"Not a valid path : {actual_dir}")
        
        return input_dir

    parser.add_argument(
        '--dirname',
        type    = dir_path,
        help    = 'Application directory name in apps/ directory. ex: `demo`, `echo_sum`',
        default = 'demo'
    )
    parser.add_argument('-p', '--pdf-file', type=str,              help='PDF file to put in the app .zip file. If None, generate a minimal one.', default=None)
    parser.add_argument('-D', '--debug',    action='store_true',   help='Build the server in debug mode')
    parser.add_argument('--nopackage',      action='store_true',   help='Do not save the docker image in a .zip file')

    args = parser.parse_args()

    main(args)
