# ----- 1. First stage to build ismrmrd and siemens_to_ismrmrd -----
FROM python:3.12.13-slim AS mrd_converter

ARG  DEBIAN_FRONTEND=noninteractive
ENV  TZ=America/Chicago

WORKDIR /opt/code

RUN apt-get update && apt-get install -y git cmake g++ libhdf5-dev libxml2-dev libxslt1-dev libboost-all-dev libfftw3-dev libpugixml-dev

# ISMRMRD library
RUN git clone https://github.com/ismrmrd/ismrmrd.git && \
    cd ismrmrd && \
    git checkout 2886c1f && \
    mkdir build && cd build && \
    cmake ../ && \
    make -j $(nproc) && \
    make install

# siemens_to_ismrmrd converter
# RUN git clone https://github.com/ismrmrd/siemens_to_ismrmrd.git && \
#     cd siemens_to_ismrmrd && \
#     git checkout v1.2.13 && \
#     mkdir build && cd build && \
#     cmake ../ && \
#     make -j $(nproc) && \
#     make install

# Create archive of ISMRMRD libraries (including symlinks) for second stage
RUN cd /usr/local/lib && tar -czvf libismrmrd.tar.gz libismrmrd*


# ----- 2. Create a devcontainer without all of the build dependencies of MRD -----
FROM python:3.12.13-slim AS python-or-devcontainer

LABEL org.opencontainers.image.description="Python OpenRecon Server"
LABEL org.opencontainers.image.url="https://github.com/benoitberanger/python-openrecon-server"

WORKDIR /opt/code

# Copy ISMRMRD files from last stage
COPY --from=mrd_converter /usr/local/include/ismrmrd        /usr/local/include/ismrmrd/
COPY --from=mrd_converter /usr/local/share/ismrmrd          /usr/local/share/ismrmrd/
COPY --from=mrd_converter /usr/local/bin/ismrmrd*           /usr/local/bin/
COPY --from=mrd_converter /usr/local/lib/libismrmrd.tar.gz  /usr/local/lib/
RUN cd /usr/local/lib && tar -zxvf libismrmrd.tar.gz && rm libismrmrd.tar.gz && ldconfig

# Copy siemens_to_ismrmrd from last stage
# COPY --from=mrd_converter /usr/local/bin/siemens_to_ismrmrd  /usr/local/bin/siemens_to_ismrmrd

# Add dependencies for siemens_to_ismrmrd
# RUN apt-get update && apt-get install --no-install-recommends -y libxslt1.1 libhdf5-103 libboost-program-options1.74.0 libpugixml1v5 git dos2unix nano
RUN apt-get update && apt-get install --no-install-recommends -y git dos2unix nano

# Tell nano to remember its position from the last time it opened a file
RUN echo "set positionlog" > ~/.nanorc

# Python MRD library
RUN pip3 install h5py==3.16.0 ismrmrd==1.14.2

# RUN cd /opt/code && \
#     git clone https://github.com/ismrmrd/ismrmrd-python-tools.git && \
#     cd /opt/code/ismrmrd-python-tools && \
#     pip3 install --no-cache-dir .

# matplotlib is used by rgb.py and provides various visualization tools including colormaps
# pydicom is used by dicom2mrd.py to parse DICOM data
RUN pip3 install --no-cache-dir numpy==2.4.4 pydicom==3.0.2 psutil==7.2.2 nibabel==5.4.2

# Cleanup files not required after installation
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /root/.cache/pip


# ----- 3. Copy deployed code into the devcontainer for deployment -----
FROM python-or-devcontainer AS python-or-runtime

# If doing local development, use this section to copy local code into Docker
# image. From the python-openrecon-server folder, uncomment the following lines
# below and run the command:
#    docker build --no-cache -t fire-python-custom -f docker/Dockerfile ./
WORKDIR /opt/code/python-openrecon-server

ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Throw an explicit error if docker build is run from the folder *containing*
# python-openrecon-server instead of within it (i.e. old method)
RUN if [ -d /python-openrecon-server ]; then echo "docker build should be run inside of python-openrecon-server instead of one directory up"; exit 1; fi
