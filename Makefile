# --- Parameters for run docker -----------------------------------------------
OPENRECON_IMAGE=openrecon_icm_invertcontrast
VERSION=v1.0.4

# --- Default parameters for server (main.py) ---------------------------------
CONFIG      ?= invertContrast
APP_DIR     ?= demo
LOGFILE		=test/test.log

# --- Parameters for local test -----------------------------------------------
IN_DIR=data/in
OUT_DIR=data/out


DATASET_NAME=test
# DATASET_NAME=bold_CMRR_10meas_mp_2026-05-20-164300_68
# DATASET_NAME=bold_CMRR_sbref_10meas_mp_2026-05-20-163940_94 (fucked up)
# DATASET_NAME=gre_1e_m__dis2dANDunfiltered_2026-05-20-161554_93
# DATASET_NAME=gre_1e_m_2026-05-20-134749_57
# DATASET_NAME=gre_1e_m_uncomb_2026-05-20-165810_25
# DATASET_NAME=gre_1e_mp_2026-05-20-135006_62
# DATASET_NAME=gre_2e_mp__dis2dANDunfiltered__3measMultiSerOFF_2026-05-20-162625_34
# DATASET_NAME=gre_2e_mp__dis2dANDunfiltered__3measMultiSerON_2026-05-20-162545_85
# DATASET_NAME=gre_2e_mp_3measMultiSerOFF_2026-05-20-135215_73
# DATASET_NAME=gre_2e_mp_3measMultiSerON_2026-05-20-135134_91
# DATASET_NAME=gre_2e_mp_2026-05-20-135043_54
# DATASET_NAME=gre_2e_mp_uncomb_2026-05-20-165830_28
# DATASET_NAME=gre_b0map_2026-05-20-163554_21
# DATASET_NAME=icubeSSFP_8pc_mp_2026-05-20-163334_63
# DATASET_NAME=mp2rage_classicT1map_2026-05-20-165019_43
# DATASET_NAME=QSM_EPI_1_iso_6echoes_P3_2_2026-05-20-165553_16

all: build run

## ----------------------------------------------------------------------
## Build / packaging OpenRecon / Run Docker
## ----------------------------------------------------------------------

build:
	@echo "Building Server Docker Image"
	python build.py

build-nopackage:
	@echo "Building Server Docker Image without packaging step"
	python build.py --nopackage

run:
	@echo "Starting the server in docker"
	docker run -p 9002:9002 -t ${OPENRECON_IMAGE}:${VERSION}

## ----------------------------------------------------------------------
## Local execution (server / client)
## ----------------------------------------------------------------------

server:
	@echo "Starting server locally"
	python main.py -v --config ${CONFIG} --dirname $(APP_DIR) $(if $(LOGFILE),-l $(LOGFILE),)

client:
	@echo "Starting Client"
	rm -f ${OUT_DIR}/*dcm ${OUT_DIR}/*h5
	python client.py -c openrecon.json -o ${OUT_DIR}/OR_${DATASET_NAME}.h5 ${IN_DIR}/${DATASET_NAME}.h5
	python -m converter.mrd2dicom -o ${OUT_DIR}/ ${OUT_DIR}/OR_${DATASET_NAME}.h5

view:
	mrview ${OUT_DIR} -mode 2

## ----------------------------------------------------------------------
## Tests
## ----------------------------------------------------------------------
 
test:
	pytest -v -rs -m "not integration"

## ----------------------------------------------------------------------
## Codespell
## ----------------------------------------------------------------------

codespell:
	codespell -L TE,nd,cach

## ----------------------------------------------------------------------
## Cleaning
## ----------------------------------------------------------------------

clean:
	rm -f ${OUT_DIR}/*dcm ${OUT_DIR}/*h5
	rm -fr builds/

fclean: clean
	rm -rf .pytest_cache
	docker system prune -a -f

re: fclean all


.PHONY: all server client debug view mrd2dicom build build-nopackage test codespell clean fclean re
