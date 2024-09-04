import ismrmrd
import os
import logging
import traceback
import numpy as np
import numpy.fft as fft
import xml.dom.minidom
import base64
import mrdhelper
import constants
from time import perf_counter

import ants


# Folder for debug output files
debugFolder = "/tmp/share/debug"

def process(connection, config, metadata):
    logging.info("Config: \n%s", config)

    # Metadata should be MRD formatted header, but may be a string
    # if it failed conversion earlier
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

    # Continuously parse incoming data parsed from MRD messages
    currentSeries = 0
    imgGroup = []
    try:
        for item in connection:
            # ----------------------------------------------------------
            # Raw k-space data messages
            # ----------------------------------------------------------
            if isinstance(item, ismrmrd.Acquisition):
                raise Exception("Raw k-space data is not supported by this module")

            # ----------------------------------------------------------
            # Image data messages
            # ----------------------------------------------------------
            elif isinstance(item, ismrmrd.Image):
                # When this criteria is met, run process_group() on the accumulated
                # data, which returns images that are sent back to the client.
                # e.g. when the series number changes:
                if item.image_series_index != currentSeries:
                    logging.info("Processing a group of images because series index changed to %d", item.image_series_index)
                    currentSeries = item.image_series_index
                    image = process_image(imgGroup, connection, config, metadata)
                    connection.send_image(image)
                    imgGroup = []

                # Only process magnitude images -- send phase images back without modification (fallback for images with unknown type)
                if (item.image_type is ismrmrd.IMTYPE_MAGNITUDE) or (item.image_type == 0):
                    imgGroup.append(item)
                else:
                    tmpMeta = ismrmrd.Meta.deserialize(item.attribute_string)
                    tmpMeta['Keep_image_geometry']    = 1
                    item.attribute_string = tmpMeta.serialize()

                    connection.send_image(item)
                    continue

            elif item is None:
                break

            else:
                raise Exception("Unsupported data type %s", type(item).__name__)

        # Process any remaining groups of image data.  This can 
        # happen if the trigger condition for these groups are not met.
        # This is also a fallback for handling image data, as the last
        # image in a series is typically not separately flagged.
        if len(imgGroup) > 0:
            logging.info("Processing a group of images (untriggered)")
            image = process_image(imgGroup, connection, config, metadata)
            connection.send_image(image)
            imgGroup = []

    except Exception as e:
        logging.error(traceback.format_exc())
        connection.send_logging(constants.MRD_LOGGING_ERROR, traceback.format_exc())
        
        # Close connection without sending MRD_MESSAGE_CLOSE message to signal failure
        connection.shutdown_close()

    finally:
        try:
            connection.send_close()
        except:
            logging.error("Failed to send close message!")

def process_image(images, connection, config, metadata):
    
    if len(images) == 0:
        return []

    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    logging.debug("Processing data with %d images of type %s", len(images), ismrmrd.get_dtype_from_data_type(images[0].data_type))

    # OR parameter : Save Original Images
    saveoriginalimages = True
    if ('parameters' in config) and ('SaveOriginalImages' in config['parameters']):
        logging.debug(f"type of config['parameters']['SaveOriginalImages'] is {type(config['parameters']['SaveOriginalImages'])}")

        if type(config['parameters']['SaveOriginalImages']) is str:
            if   config['parameters']['SaveOriginalImages'].lower() == 'true' :
                saveoriginalimages = True
            elif config['parameters']['SaveOriginalImages'].lower() == 'false':
                saveoriginalimages = False

        elif type(config['parameters']['SaveOriginalImages']) is bool:
            saveoriginalimages = config['parameters']['SaveOriginalImages']
        
    else:
        logging.warning("config['parameters']['SaveOriginalImages'] NOT FOUND !!")

    logging.info(f'saveoriginalimages = {saveoriginalimages}')

    if saveoriginalimages:
        images_ORIG = images.copy()

    # OR parameter : Config
    ANTsConfig = 'N4Dn'
    if ('parameters' in config) and ('config' in config['parameters']):
        ANTsConfig =  config['parameters']['ANTsConfig']
    else:
        logging.warning("config['parameters']['ANTsConfig'] NOT FOUND !!")

    logging.info(f'ANTsConfig = {ANTsConfig}')

    # Extract image data into a 5D array of size [img cha z y x]
    data = np.stack([img.data                              for img in images])
    head = [img.getHead()                                  for img in images]
    meta = [ismrmrd.Meta.deserialize(img.attribute_string) for img in images]

    # Display MetaAttributes for first image
    logging.debug("MetaAttributes[0]: %s", ismrmrd.Meta.serialize(meta[0]))

    # Optional serialization of ICE MiniHeader
    if 'IceMiniHead' in meta[0]:
        logging.debug("IceMiniHead[0]: %s", base64.b64decode(meta[0]['IceMiniHead']).decode('utf-8'))


    info = {
        'image_series_index_offset': 0,
        'ImageProcessingHistory': [],
        'SequenceDescriptionAdditional': '',
    }

    images_out = []
    if saveoriginalimages:
        images_out += images_ORIG
        info['image_series_index_offset'] += 1

    data_3d = get3Darray(data)
    ants_image_in = ants.from_numpy(data_3d)

    if ANTsConfig == 'N4':
        ants_image_n4 = ants.n4_bias_field_correction(ants_image_in, verbose=True)
        info['ImageProcessingHistory'].append('ANTs::N4BiasFieldCorrection')
        info['SequenceDescriptionAdditional'] += 'ANTsN4BiasFieldCorrection'
        images_n4 = createMRDImage(ants_image_n4, head, meta, metadata, info)
        images_out += images_n4

    elif ANTsConfig == 'Dn':
        ants_image_dn = ants.denoise_image(ants_image_in, v=1, r=2)
        info['ImageProcessingHistory'       ].append('ANTs::DenoiseImage')
        info['SequenceDescriptionAdditional'] += 'ANTsDenoiseImage'
        images_dn = createMRDImage(ants_image_dn, head, meta, metadata, info)
        images_out += images_dn

    elif ANTsConfig == 'N4Dn':
        ants_image_n4 = ants.n4_bias_field_correction(ants_image_in, verbose=True)
        info['ImageProcessingHistory'].append('ANTs::N4BiasFieldCorrection')
        info['SequenceDescriptionAdditional'] += 'ANTsN4BiasFieldCorrection'
        if saveoriginalimages:
            images_n4 = createMRDImage(ants_image_n4, head, meta, metadata, info)
            images_out += images_n4
            info['image_series_index_offset'] += 1
        ants_image_dn_n4 = ants.denoise_image(ants_image_n4, v=1, r=2)
        info['ImageProcessingHistory'].append('ANTs::DenoiseImage')
        info['SequenceDescriptionAdditional'] += '_ANTsDenoiseImage'
        images_dn_n4 = createMRDImage(ants_image_dn_n4, head, meta, metadata, info)
        images_out += images_dn_n4

    elif ANTsConfig == 'DnN4':
        ants_image_dn = ants.denoise_image(ants_image_in, v=1, r=2)
        info['ImageProcessingHistory'].append('ANTs::DenoiseImage')
        info['SequenceDescriptionAdditional'] += 'ANTsDenoiseImage'
        if saveoriginalimages:
            images_dn = createMRDImage(ants_image_dn, head, meta, metadata, info)
            images_out += images_dn
            info['image_series_index_offset'] += 1
        ants_image_n4_dn = ants.n4_bias_field_correction(ants_image_dn, verbose=True)
        info['ImageProcessingHistory'].append('ANTs::N4BiasFieldCorrection')
        info['SequenceDescriptionAdditional'] += '_ANTsN4BiasFieldCorrection'
        images_n4_dn = createMRDImage(ants_image_n4_dn, head, meta, metadata, info)
        images_out += images_n4_dn

    return images_out


def get3Darray(data) -> np.array:

    # Reformat data to [y x z cha img], i.e. [row col] for the first two dimensions
    data = data.transpose((3, 4, 2, 1, 0))

    logging.debug("Original image data is size %s" % (data.shape,))
    np.save(debugFolder + "/" + "imgOrig.npy", data)

    data_5d = data.astype(np.float64)

    # Reformat data from [y x z cha img] to [y x img]
    data_3d = data_5d[:,:,0,0,:]
    
    return data_3d


def createMRDImage(data_3d, head, meta, metadata, info):

    # Reformat data from [y x img] to [y x z cha img]
    data = data_3d.numpy()[:,:,np.newaxis,np.newaxis,:].astype(np.int16)

    # Re-slice back into 2D images
    imagesOut = [None] * data.shape[-1]
    for iImg in range(data.shape[-1]):
        imagesOut[iImg] = ismrmrd.Image.from_array(data[...,iImg].transpose((3, 2, 0, 1)), transpose=False)

        # Create a copy of the original fixed header and update the data_type
        # (we changed it to int16 from all other types)
        oldHeader = head[iImg]
        oldHeader.data_type = imagesOut[iImg].data_type

        # Set the image_type to match the data_type for complex data
        if (imagesOut[iImg].data_type == ismrmrd.DATATYPE_CXFLOAT) or (imagesOut[iImg].data_type == ismrmrd.DATATYPE_CXDOUBLE):
            oldHeader.image_type = ismrmrd.IMTYPE_COMPLEX

        # Increment series number when flag detected (i.e. follow ICE logic for splitting series)
        # if mrdhelper.get_meta_value(meta[iImg], 'IceMiniHead') is not None:
        #     if mrdhelper.extract_minihead_bool_param(base64.b64decode(meta[iImg]['IceMiniHead']).decode('utf-8'), 'BIsSeriesEnd') is True:
        #         currentSeries += 1

        oldHeader.image_series_index += info['image_series_index_offset']

        imagesOut[iImg].setHead(oldHeader)

        # Determine max value (12 or 16 bit)
        BitsStored = 12
        if (mrdhelper.get_userParameterLong_value(metadata, "BitsStored") is not None):
            BitsStored = mrdhelper.get_userParameterLong_value(metadata, "BitsStored")
        maxVal = 2**BitsStored - 1

        # Create a copy of the original ISMRMRD Meta attributes and update
        tmpMeta = meta[iImg]
        tmpMeta['DataRole']                       = 'Image'
        tmpMeta['ImageProcessingHistory']         = info['ImageProcessingHistory']
        tmpMeta['WindowCenter']                   = str((maxVal+1)/2)
        tmpMeta['WindowWidth']                    = str((maxVal+1))
        tmpMeta['SequenceDescriptionAdditional']  = info['SequenceDescriptionAdditional']
        tmpMeta['Keep_image_geometry']            = 1

        # # Add image orientation directions to MetaAttributes if not already present
        # if tmpMeta.get('ImageRowDir') is None:
        #     tmpMeta['ImageRowDir'] = ["{:.18f}".format(oldHeader.read_dir[0]), "{:.18f}".format(oldHeader.read_dir[1]), "{:.18f}".format(oldHeader.read_dir[2])]

        # if tmpMeta.get('ImageColumnDir') is None:
        #     tmpMeta['ImageColumnDir'] = ["{:.18f}".format(oldHeader.phase_dir[0]), "{:.18f}".format(oldHeader.phase_dir[1]), "{:.18f}".format(oldHeader.phase_dir[2])]

        metaXml = tmpMeta.serialize()
        logging.debug("Image MetaAttributes: %s", xml.dom.minidom.parseString(metaXml).toprettyxml())
        logging.debug("Image data has %d elements", imagesOut[iImg].data.size)

        imagesOut[iImg].attribute_string = metaXml

    return imagesOut