# intenend for python3

# external modules
import jsonschema

# builtin modules
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
import urllib


def print_section(name: str) -> None:
    DEBUG_LINE = '#'*40
    print('')
    print(DEBUG_LINE)
    print(f'# {name}')
    print(DEBUG_LINE)


def check_zip() -> None:
    logger = logging.getLogger()

    # zip ?
    path_git = shutil.which('zip')
    if path_git:
        logger.info('`zip` is installed')
    else:
        logger.critical('`zip` does not seem to be present in the system')
        sys.exit(1)


def check_git() -> None:
    logger = logging.getLogger()

    # git ?
    path_git = shutil.which('git')
    if path_git:
        logger.info('`git` is installed')
    else:
        logger.critical('`git` does not seem to be present in the system')
        sys.exit(1)


def check_docker() -> None:
    logger = logging.getLogger()

    # docker ?
    path_docker = shutil.which('docker')
    if path_docker:
        logger.info('`docker` is installed')
    else:
        logger.critical('`Docker` does not seem to be present in the system')
        sys.exit(1)

    # docker version ok ?
    result = subprocess.run(['docker', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    version_output = result.stdout.strip()
    logger.debug(version_output)
    pattern = r"Docker version (\d+\.\d+\.\d+),"
    matches = re.findall(pattern, version_output)
    if not len(matches):
        logger.critical('Could not find Docker version')
        sys.exit(1)
    docker_version = matches[0].split('.')[0] # 20.21.22 -> 20
    maximum_docker_version = 25
    if int(docker_version) >= maximum_docker_version:
        logger.critical(f'Docker version {docker_version} is too high. Maximum is {maximum_docker_version}')
        sys.exit(1)
    logger.info(f'Docker version {docker_version}<{maximum_docker_version} is ok')


def clone_server(repo_path: str) -> None:
    logger = logging.getLogger()

    # python-ismrmrd-server : this has the client / server functionality
    git_adress = 'https://github.com/kspaceKelvin/python-ismrmrd-server'
    if os.path.exists(repo_path):
        logger.info(f'Found `python-ismrmrd-server` dir, do not clone it again : {repo_path}')
    else:
        logger.info('`python-ismrmrd-server` not found, cloning it...')
        subprocess.run(['git', 'clone', git_adress], check=True)


def build_server(repo_dockerfile_path: str) -> None:
        logger = logging.getLogger()

        result = subprocess.run(['docker', 'images', 'python-ismrmrd-server'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        output = result.stdout.strip()
        if 'python-ismrmrd-server' in output:
            logger.info('docker image `python-ismrmrd-server` already built')
            return

        # build docker image for python-ismrmrd-server
        # this image is the starting point, that will be refined latter
        logger.info('building docker image `python-ismrmrd-server`')
        subprocess.run(['docker', 'build', '--tag', 'python-ismrmrd-server', '--file', repo_dockerfile_path, './'], check=True)


def check_target_dir(target_path: str) -> dict:
    logger = logging.getLogger()

    # files to find
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
        

def create_pdf(file_path: str, lines_of_text: list[str]) -> None:
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


def main(args: argparse.Namespace):

    #############
    ### setup ###
    #############

    # setup logging
    logging.basicConfig(
        level=logging.DEBUG,
        format=f"%(levelname)8s:%(funcName)15s: %(message)s",
    )
    logger = logging.getLogger()

    print_section('START')
    logger.info(f'Start of {os.path.basename(__file__)}')
    logger.warning('Untill the BUILD part, there is a "skip if already done" feature')
    cwd = os.getcwd()
    logger.info(f'Current working directory : {cwd}')


    # check if all system programs are here
    print_section('SYSTEM DEPENDENCIES')
    check_zip()
    check_git()
    check_docker()

    # python-ismrmrd-server : clone & build docker image
    print_section('CLONE & BUILD SERVER')
    repo_path            = os.path.join(cwd, 'python-ismrmrd-server')
    repo_dockerfile_path = os.path.join(repo_path, 'docker', 'Dockerfile')
    clone_server(repo_path)
    build_server(repo_dockerfile_path)

    # target dir
    target_path = os.path.join(cwd, args.dirname)
    print_section(f'Check "target" dir and its content : {target_path}')
    target_data = check_target_dir(target_path)

    #############
    ### build ###
    #############

    print_section('BUILD')
    logger.warning('From now on, all steps will not have a "skip if already done" feature')

    # download the DeepLearning weights
    weights_name = 'synthstrip.1.pt'
    weights_url = 'https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/requirements/synthstrip.1.pt'
    weights_path = os.path.join(cwd, 'app', weights_name)
    if os.path.exists(weights_path):
        logger.info(f'Weigths `{weights_name}` in `app` dir : {weights_path}')
    else:
        logger.info(f'Download weigths `{weights_name}` in `app` dir : {weights_path}')
        urllib.request.urlretrieve(weights_url, weights_path)

    # prep build dir
    build_path = os.path.join(cwd, 'build')
    if os.path.exists(build_path):
        logger.info(f'`build` dir found : {build_path}')
    else:
        os.mkdir(build_path)
        logger.info(f'`build` dir created : {build_path}')

    # prep some paths
    build_data = {
        'name': {
            'docker': '',
            'base'  : '',
        },
        'path': {
            'process' : target_data['path']['process'].replace(target_path, build_path),
            'ui_json' : target_data['path']['ui_json'].replace(target_path, build_path),
            'schema'  : target_data['path']['schema' ].replace(target_path, build_path),
            'pdf'     : '',
        }
    }
    pprint.pprint(build_data, sort_dicts=False)

    # copy files in the `build` dir
    to_copy = [
        # [src dst]
        [target_data['path']['process'], build_data['path']['process']                ],
        [target_data['path']['ui_json'], build_data['path']['ui_json']                ],
        [target_data['path']['schema' ], build_data['path']['schema' ]                ],
        [weights_path                  , weights_path.replace(target_path, build_path)],
    ]
    for src_dst in to_copy:
        logger.info(f'copy : SRC={src_dst[0]} DST={src_dst[1]}')
        shutil.copy(src=src_dst[0],dst=src_dst[1])

    # load JSON UI
    logger.info(f'load UI JSON content : {build_data['path']['ui_json']}')
    with open(build_data['path']['ui_json'], 'r') as fid:
        json_content = json.load(fid)

    # prep info
    cmdline  = f'CMD [ "python3", "/opt/code/python-ismrmrd-server/main.py", "-v", "-H=0.0.0.0", "-p=9002", "-l=/tmp/python-ismrmrd-server.log", "--defaultConfig={target_data['name']['process']}"]'
    version                         = json_content['general']['version']
    vendor                          = json_content['general']['vendor' ]
    name                            = json_content['general']['id'     ]

    # other file/path
    build_data['name']['docker'] = f'OpenRecon_{vendor}_{name}:V{version}'.lower()
    build_data['name']['base'  ] = f'OpenRecon_{vendor}_{name}_V{version}'
    build_data['path']['docker'] = os.path.join(build_path, f'{build_data['name']['base']}.Dockerfile')
    build_data['path']['tar'   ] = os.path.join(build_path, f'{build_data['name']['base']}.tar')
    build_data['path']['zip'   ] = os.path.join(build_path, f'{build_data['name']['base']}.zip')
    build_data['path']['pdf'   ] = os.path.join(build_path, f'{build_data['name']['base']}.pdf')
    pprint.pprint(build_data, sort_dicts=False)

    # load JSON Schema, to check if our updated JSON is ok
    logger.info(f'load JSON Schema : {build_data['path']['schema']}')
    with open(file=build_data['path']['schema'], mode='r') as fid:
        schema_content = json.load(fp=fid)
    validator = jsonschema.Draft7Validator(schema=schema_content)
    errors = list(validator.iter_errors(instance=json_content))
    if errors:
        logger.error('our Json vs. Schema errors :')
        for error in errors:
                logger.error(error)
        sys.exit(1)
    logger.info(f'No error in out JSON compared against the Schema')

    # write the updated json in the `build` dir
    encoded_json_content = base64.b64encode((json.dumps(obj=json_content,indent=2)).encode('utf-8')).decode('utf-8')
    
    # write the Dockerfile content
    logger.info(f'Write `build` Dockerfile : {build_data['path']['docker']}')
    with open(file=build_data['path']['docker'], mode='w') as fid:
        fid.writelines([
            '# import python-ismrmrd-server as starting point \n',
            f'FROM python-ismrmrd-server \n',
            '\n'])
        fid.writelines([
            '# mandatory for OpenRecon (see OR documentation) \n',
            f'LABEL "com.siemens-healthineers.magneticresonance.openrecon.metadata:1.1.0"="{encoded_json_content}" \n',
            '\n'])
        fid.writelines([
            '# copy the .py module \n',
            f'COPY {os.path.relpath(target_data['path']['process'], cwd)}  /opt/code/python-ismrmrd-server \n',
            '\n'])
        fid.writelines([
            '# copy the .pt file (weights) \n',
            f'COPY {os.path.relpath(weights_path, cwd)}  /opt/code/python-ismrmrd-server \n',
            '\n'])
        fid.writelines([
            '# Python modules \n',
            'RUN apt-get update && apt-get install -y gcc \n',
            'RUN pip3 --no-cache-dir install antspyx torch surfa \n',
            '\n'])
        fid.writelines([
            '# new CMD line \n',
            f'{cmdline} \n',
            '\n'])
        
    # build docker image
    logger.info(f'building docker image `{build_data['name']['docker']}` from Docker file {build_data['path']['docker']}')
    subprocess.run(['docker', 'build', '--tag', build_data['name']['docker'], '--file', build_data['path']['docker'], cwd], check=True)

    # save docker image in a .tar
    logger.info(f'(1/2) saving image `{build_data['name']['docker']}` in a .tar {build_data['path']['tar']}')
    subprocess.run(['docker', 'save', '-o', build_data['path']['tar'], build_data['name']['docker']], check=True)
    logger.info(f'(2/2) saving image DONE')

    # generate PDF
    lines = [
        f'vendor={vendor}',
        f'name={name}',
        f'version={version}',
    ]
    logger.info(f'write PDF file : {build_data['path']['pdf']}')
    create_pdf(file_path=build_data['path']['pdf'], lines_of_text=lines)

    # save everything in a ZIP file
    logger.info(f'(1/2) zip all files : {build_data['path']['zip']}')
    subprocess.run(['zip', build_data['name']['base']+'.zip', build_data['name']['base']+'.tar', build_data['name']['base']+'.pdf'], check=True, cwd=build_path)
    logger.info(f'(2/2) zip all files DONE')

    # END
    print_section('All done !')
    sys.exit(0)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog            = 'build',
        description     = 'Build OpenRecon app',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )

    def dir_path(input_dir):
        if os.path.isdir(input_dir):
            return input_dir
        else:
            raise argparse.ArgumentTypeError(f"Not a valid path : {input_dir}")
    parser.add_argument(
        '--dirname',
        type    = dir_path,
        help    = 'Application directory name. ex: `demo-i2i`, `app`',
        default = 'app'
    )

    args = parser.parse_args()

    main(args)
