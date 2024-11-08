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
import pydicom

import ants

# for synthstrip
import torch
import torch.nn as nn
import surfa as sf


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

    # OR parameter : Apply In Brain Mask
    applyinbrainmask = True
    if ('parameters' in config) and ('ApplyInBrainMask' in config['parameters']):
        logging.debug(f"type of config['parameters']['ApplyInBrainMask'] is {type(config['parameters']['ApplyInBrainMask'])}")

        if type(config['parameters']['ApplyInBrainMask']) is str:
            if   config['parameters']['ApplyInBrainMask'].lower() == 'true' :
                applyinbrainmask = True
            elif config['parameters']['ApplyInBrainMask'].lower() == 'false':
                applyinbrainmask = False

        elif type(config['parameters']['ApplyInBrainMask']) is bool:
            applyinbrainmask = config['parameters']['ApplyInBrainMask']
        
    else:
        logging.warning("config['parameters']['ApplyInBrainMask'] NOT FOUND !!")

    logging.info(f'applyinbrainmask = {applyinbrainmask}')

    # OR parameter : ANTsConfig
    ANTsConfig = 'DnN4'
    if ('parameters' in config) and ('ANTsConfig' in config['parameters']):
        ANTsConfig =  config['parameters']['ANTsConfig']
    else:
        logging.warning("config['parameters']['ANTsConfig'] NOT FOUND !!")

    logging.info(f'ANTsConfig = {ANTsConfig}')


    # Extract image data into a 5D array of size [img cha z y x]
    data = np.stack([img.data                              for img in images])
    logging.info(f'MRD supposed organization : [img cha z y x]')
    logging.info(f'MRD data shape : {data.shape}')
    head = [img.getHead()                                  for img in images]
    meta = [ismrmrd.Meta.deserialize(img.attribute_string) for img in images]

    logging.warning(f'MRD SequenceDescription : {meta[0]['SequenceDescription']}')

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

    dicomDset = pydicom.dataset.Dataset.from_json(base64.b64decode(meta[0]['DicomJson']))
    ImageOrientationPatient = np.array(dicomDset.ImageOrientationPatient)
    logging.info(f'ImageOrientationPatient : {ImageOrientationPatient}')

    matrix = np.array(head[0].matrix_size[:]) 
    fov    = np.array(head[0].field_of_view[:])
    voxelsize = fov/matrix
    logging.info(f'MRD computed maxtrix [x y z] : {matrix   }')
    logging.info(f'MRD computed fov     [x y z] : {fov      }')
    logging.info(f'MRD computed voxel   [x y z] : {voxelsize}')

    read_dir  = np.array(images[0].read_dir )
    phase_dir = np.array(images[0].phase_dir)
    slice_dir = np.array(images[0].slice_dir)
    logging.info(f'MRD read_dir  [x y z] : {read_dir }')
    logging.info(f'MRD phase_dir [x y z] : {phase_dir}')
    logging.info(f'MRD slice_dir [x y z] : {slice_dir}')

    if applyinbrainmask:

        # affine = compute_nifti_affine(fov, matrix, direction_cosines)
        # affine = np.diag([*voxelsize,  1])
        affine = compute_nifti_affine(images[0] ,matrix, fov) # this is the good one !!!
        # affine = compute_nifti_affine_from_dicom(meta[0])

        # synthstrip
        logging.info(f'Affine for Synthstrip masking :')
        print(affine)
        data_iyx = data[:,0,0,:,:] # [img cha z y x] -> [img y x]
        data_xyi = data_iyx.transpose((2,1,0)) # [img y x] -> [x y img]
        logging.info(f'Shape of Synthstrip input : {data_xyi.shape}')
        brainmask = synthstrip(data_xyi, voxelsize, affine)
        logging.info(f'Shape of Synthstrip output : {brainmask.shape}')
        brainmask = brainmask.transpose((1,0,2)) # [x y img] -> [y x img]
        logging.info(f'Shape of mask (before ANTs) : {brainmask.shape}')
        # !!! ants N4 has a weird conversion==unstable mask system
        # workaround : cast the input nparray from bool to float
        ants_mask = ants.from_numpy(np.array(brainmask,dtype=np.float32))

    data_3d = get3Darray(data)
    logging.info(f'ANTs input data shape : {data_3d.shape}')
    ants_image_in = ants.from_numpy(data_3d)
    images_out = []
    
    if saveoriginalimages:
        images_ORIG = createMRDImage(ants_image_in, head, meta, metadata, info)
        images_out += images_ORIG
        info['image_series_index_offset'] += 1
        if applyinbrainmask:
            info['ImageProcessingHistory'].append('SynthstripMask')
            images_mask = createMRDImage(ants_mask, head, meta, metadata, info)
            images_out += images_mask
            info['image_series_index_offset'] += 1
            info['SequenceDescriptionAdditional'] += 'Masked_'

    if ANTsConfig == 'N4':
        if applyinbrainmask:
            ants_image_n4 = ants.n4_bias_field_correction(ants_image_in, verbose=True, mask=ants_mask)
            info['ImageProcessingHistory'].append('ANTs::N4BiasFieldCorrection@SynthstripMask')
        else:
            ants_image_n4 = ants.n4_bias_field_correction(ants_image_in, verbose=True)
            info['ImageProcessingHistory'].append('ANTs::N4BiasFieldCorrection')
        if saveoriginalimages:
            info['SequenceDescriptionAdditional'] += 'N4'
        images_n4 = createMRDImage(ants_image_n4, head, meta, metadata, info)
        images_out += images_n4

    elif ANTsConfig == 'Dn':
        if applyinbrainmask:
            ants_image_dn = ants.denoise_image(ants_image_in, v=1, r=2, mask=ants_mask)
            info['ImageProcessingHistory'].append('ANTs::DenoiseImage@SynthstripMask')
        else:
            ants_image_dn = ants.denoise_image(ants_image_in, v=1, r=2)
            info['ImageProcessingHistory'].append('ANTs::DenoiseImage')
        if saveoriginalimages:
            info['SequenceDescriptionAdditional'] += 'Dn'
        images_dn = createMRDImage(ants_image_dn, head, meta, metadata, info)
        images_out += images_dn

    elif ANTsConfig == 'N4Dn':
        if applyinbrainmask:
            ants_image_n4 = ants.n4_bias_field_correction(ants_image_in, verbose=True, mask=ants_mask)
            info['ImageProcessingHistory'].append('ANTs::N4BiasFieldCorrection@SynthstripMask')
        else:
            ants_image_n4 = ants.n4_bias_field_correction(ants_image_in, verbose=True)
            info['ImageProcessingHistory'].append('ANTs::N4BiasFieldCorrection')
        if saveoriginalimages:
            info['SequenceDescriptionAdditional'] += 'N4'
            images_n4 = createMRDImage(ants_image_n4, head, meta, metadata, info)
            images_out += images_n4
            info['image_series_index_offset'] += 1
        if applyinbrainmask:
            ants_image_dn_n4 = ants.denoise_image(ants_image_n4, v=1, r=2, mask=ants_mask)
            info['ImageProcessingHistory'].append('ANTs::DenoiseImage@SynthstripMask')
        else:
            ants_image_dn_n4 = ants.denoise_image(ants_image_n4, v=1, r=2)
            info['ImageProcessingHistory'].append('ANTs::DenoiseImage')
        if saveoriginalimages:
            info['SequenceDescriptionAdditional'] += '_Dn'
        images_dn_n4 = createMRDImage(ants_image_dn_n4, head, meta, metadata, info)
        images_out += images_dn_n4

    elif ANTsConfig == 'DnN4':
        if applyinbrainmask:
            ants_image_dn = ants.denoise_image(ants_image_in, v=1, r=2, mask=ants_mask)
            info['ImageProcessingHistory'].append('ANTs::DenoiseImage@SynthstripMask')
        else:
            ants_image_dn = ants.denoise_image(ants_image_in, v=1, r=2)
            info['ImageProcessingHistory'].append('ANTs::DenoiseImage')
        if saveoriginalimages:
            info['SequenceDescriptionAdditional'] += 'Dn'
            images_dn = createMRDImage(ants_image_dn, head, meta, metadata, info)
            images_out += images_dn
            info['image_series_index_offset'] += 1
        if applyinbrainmask:
            ants_image_n4_dn = ants.n4_bias_field_correction(ants_image_dn, verbose=True, mask=ants_mask)
            info['ImageProcessingHistory'].append('ANTs::N4BiasFieldCorrection@SynthstripMask')
        else:
            ants_image_n4_dn = ants.n4_bias_field_correction(ants_image_dn, verbose=True)
            info['ImageProcessingHistory'].append('ANTs::N4BiasFieldCorrection')
        if saveoriginalimages:
            info['SequenceDescriptionAdditional'] += '_N4'
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


def createMRDImage(ants_image, head, meta, metadata, info):

    # Reformat data from [y x img] to [y x z cha img]
    data = ants_image.numpy()[:,:,np.newaxis,np.newaxis,:].astype(np.int16)

    # Re-slice back into 2D images
    imagesOut = [None] * data.shape[-1]
    for iImg in range(data.shape[-1]):

        # Create new MRD instance for the inverted image
        # Transpose from convenience shape of [y x z cha] to MRD Image shape of [cha z y x]
        # from_array() should be called with 'transpose=False' to avoid warnings, and when called
        # with this option, can take input as: [cha z y x], [z y x], or [y x]
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

        # # Determine max value (12 or 16 bit)
        # BitsStored = 12
        # if (mrdhelper.get_userParameterLong_value(metadata, "BitsStored") is not None):
        #     BitsStored = mrdhelper.get_userParameterLong_value(metadata, "BitsStored")
        # maxVal = 2**BitsStored - 1

        # Create a copy of the original ISMRMRD Meta attributes and update
        tmpMeta = meta[iImg]
        tmpMeta['DataRole']                       = 'Image'
        if len(info['ImageProcessingHistory']) > 0:
            tmpMeta['ImageProcessingHistory']         = info['ImageProcessingHistory']
        # tmpMeta['WindowCenter']                   = str((maxVal+1)/2)
        # tmpMeta['WindowWidth']                    = str((maxVal+1))
        if len(info['SequenceDescriptionAdditional']) > 0:
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

def synthstrip(data, voxelsize, affine):
    # necessary for speed gains (I think)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    device = torch.device('cpu')
    device_name = 'CPU'

    def extend_sdt(sdt, border=1):
        """Extend SynthStrip's narrow-band signed distance transform (SDT).

        Recompute the positive outer part of the SDT estimated by SynthStrip, for
        borders that likely exceed the 4-5 mm band. Keeps the negative inner part
        intact and only computes the outer part where needed to save time.

        Parameters
        ----------
        sdt : sf.Volume
            Narrow-band signed distance transform estimated by SynthStrip.
        border : float, optional
            Mask border threshold in millimeters.

        Returns
        -------
        sdt : sf.Volume
            Extended SDT.

        """
        if border < int(sdt.max()):
            return sdt

        # Find bounding box.
        mask = sdt < 1
        keep = np.nonzero(mask)
        low = np.min(keep, axis=-1)
        upp = np.max(keep, axis=-1)

        # Add requested border.
        gap = int(border + 0.5)
        low = (max(i - gap, 0) for i in low)
        upp = (min(i + gap, d - 1) for i, d in zip(upp, mask.shape))

        # Compute EDT within bounding box. Keep interior values.
        ind = tuple(slice(a, b + 1) for a, b in zip(low, upp))
        out = np.full_like(sdt, fill_value=100)
        out[ind] = sf.Volume(mask[ind]).distance()
        out[keep] = sdt[keep]

        return sdt.new(out)

    # configure model
    print(f'Configuring model on the {device_name}')

    class StripModel(nn.Module):

        def __init__(self,
                    nb_features=16,
                    nb_levels=7,
                    feat_mult=2,
                    max_features=64,
                    nb_conv_per_level=2,
                    max_pool=2,
                    return_mask=False):

            super().__init__()

            # dimensionality
            ndims = 3

            # build feature list automatically
            if isinstance(nb_features, int):
                if nb_levels is None:
                    raise ValueError('must provide unet nb_levels if nb_features is an integer')
                feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
                feats = np.clip(feats, 1, max_features)
                nb_features = [
                    np.repeat(feats[:-1], nb_conv_per_level),
                    np.repeat(np.flip(feats), nb_conv_per_level)
                ]
            elif nb_levels is not None:
                raise ValueError('cannot use nb_levels if nb_features is not an integer')

            # extract any surplus (full resolution) decoder convolutions
            enc_nf, dec_nf = nb_features
            nb_dec_convs = len(enc_nf)
            final_convs = dec_nf[nb_dec_convs:]
            dec_nf = dec_nf[:nb_dec_convs]
            self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

            if isinstance(max_pool, int):
                max_pool = [max_pool] * self.nb_levels

            # cache downsampling / upsampling operations
            MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
            self.pooling = [MaxPooling(s) for s in max_pool]
            self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

            # configure encoder (down-sampling path)
            prev_nf = 1
            encoder_nfs = [prev_nf]
            self.encoder = nn.ModuleList()
            for level in range(self.nb_levels - 1):
                convs = nn.ModuleList()
                for conv in range(nb_conv_per_level):
                    nf = enc_nf[level * nb_conv_per_level + conv]
                    convs.append(ConvBlock(ndims, prev_nf, nf))
                    prev_nf = nf
                self.encoder.append(convs)
                encoder_nfs.append(prev_nf)

            # configure decoder (up-sampling path)
            encoder_nfs = np.flip(encoder_nfs)
            self.decoder = nn.ModuleList()
            for level in range(self.nb_levels - 1):
                convs = nn.ModuleList()
                for conv in range(nb_conv_per_level):
                    nf = dec_nf[level * nb_conv_per_level + conv]
                    convs.append(ConvBlock(ndims, prev_nf, nf))
                    prev_nf = nf
                self.decoder.append(convs)
                if level < (self.nb_levels - 1):
                    prev_nf += encoder_nfs[level]

            # now we take care of any remaining convolutions
            self.remaining = nn.ModuleList()
            for num, nf in enumerate(final_convs):
                self.remaining.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf

            # final convolutions
            if return_mask:
                self.remaining.append(ConvBlock(ndims, prev_nf, 2, activation=None))
                self.remaining.append(nn.Softmax(dim=1))
            else:
                self.remaining.append(ConvBlock(ndims, prev_nf, 1, activation=None))

        def forward(self, x):

            # encoder forward pass
            x_history = [x]
            for level, convs in enumerate(self.encoder):
                for conv in convs:
                    x = conv(x)
                x_history.append(x)
                x = self.pooling[level](x)

            # decoder forward pass with upsampling and concatenation
            for level, convs in enumerate(self.decoder):
                for conv in convs:
                    x = conv(x)
                if level < (self.nb_levels - 1):
                    x = self.upsampling[level](x)
                    x = torch.cat([x, x_history.pop()], dim=1)

            # remaining convs at full resolution
            for conv in self.remaining:
                x = conv(x)

            return x

    class ConvBlock(nn.Module):
        """
        Specific convolutional block followed by leakyrelu for unet.
        """

        def __init__(self, ndims, in_channels, out_channels, stride=1, activation='leaky'):
            super().__init__()

            Conv = getattr(nn, 'Conv%dd' % ndims)
            self.conv = Conv(in_channels, out_channels, 3, stride, 1)
            if activation == 'leaky':
                self.activation = nn.LeakyReLU(0.2)
            elif activation == None:
                self.activation = None
            else:
                raise ValueError(f'Unknown activation: {activation}')

        def forward(self, x):
            out = self.conv(x)
            if self.activation is not None:
                out = self.activation(out)
            return out

    with torch.no_grad():
        model = StripModel()
        model.to(device)
        model.eval()

    version = '1'
    print(f'Running SynthStrip model version {version}')

    modelfile = os.path.join(os.path.dirname(os.path.realpath(__file__)), f'synthstrip.{version}.pt')
    checkpoint = torch.load(modelfile, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # adaptation : rebuild the nifti info
    geom = sf.ImageGeometry(data.shape, voxsize=voxelsize, vox2world=affine)
    image = sf.Volume(data, geometry=geom)

    # loop over frames (try not to keep too much data in memory)
    print(f'Processing frame (of {image.nframes}):', end=' ', flush=True)
    dist = []
    mask = []
    border = 1 # default arg from the CLI
    for f in [0]:
    # for f in range(image.nframes):
        print(f + 1, end=' ', flush=True)
        frame = image.new(image.framed_data[..., f])

        # conform, fit to shape with factors of 64
        conformed = frame.conform(voxsize=1.0, dtype='float32', method='nearest', orientation='LIA').crop_to_bbox()
        target_shape = np.clip(np.ceil(np.array(conformed.shape[:3]) / 64).astype(int) * 64, 192, 320)
        conformed = conformed.reshape(target_shape)

        # normalize
        conformed -= conformed.min()
        conformed = (conformed / conformed.percentile(99)).clip(0, 1)

        # predict the sdt
        with torch.no_grad():
            input_tensor = torch.from_numpy(conformed.data[np.newaxis, np.newaxis]).to(device)
            sdt = model(input_tensor).cpu().numpy().squeeze()

        # extend the sdt if needed, unconform
        sdt = extend_sdt(conformed.new(sdt), border=border)
        sdt = sdt.resample_like(image, fill=100)
        dist.append(sdt)

        # extract mask, find largest CC to be safe
        mask.append((sdt < border).connected_component_mask(k=1, fill=True))

    # combine frames and end line
    dist = sf.stack(dist)
    mask = sf.stack(mask)
    print('done')

    return mask.data

# def compute_nifti_affine(fov, matrix_size, direction_cosines, offset=[0, 0, 0]):
#     voxel_size = [fov[i] / matrix_size[i] for i in range(3)]
    
#     # Scale direction cosines by voxel size
#     rotation_scaling_matrix = np.array([
#         [voxel_size[0] * direction_cosines['readout'][i] for i in range(3)],
#         [voxel_size[1] * direction_cosines['phase'][i] for i in range(3)],
#         [voxel_size[2] * direction_cosines['slice'][i] for i in range(3)]
#     ])
    
#     # Build affine matrix
#     affine = np.eye(4)
#     affine[:3, :3] = rotation_scaling_matrix
#     affine[:3, 3] = offset
    
#     return affine

def compute_nifti_affine(image_header, matrix_size, field_of_view):
    # Extract necessary fields
    # matrix_size   = image_header['matrix_size']        # [nx, ny, nz]
    # field_of_view = image_header['field_of_view']      # [fov_x, fov_y, fov_z]
    position      = image_header.position           # [px, py, pz]
    read_dir      = image_header.read_dir          # [rx, ry, rz]
    phase_dir     = image_header.phase_dir          # [px, py, pz]
    slice_dir     = image_header.slice_dir          # [sx, sy, sz]

    # Convert from LPS to RAS
    position_ras = [-position[0], -position[1], position[2]]
    read_dir_ras = [-read_dir[0], -read_dir[1], read_dir[2]]
    phase_dir_ras = [-phase_dir[0], -phase_dir[1], phase_dir[2]]
    slice_dir_ras = [-slice_dir[0], -slice_dir[1], slice_dir[2]]

    # Compute voxel size
    voxel_size = field_of_view / matrix_size
    
    # Construct rotation-scaling matrix
    rotation_scaling_matrix = np.column_stack([
        voxel_size[0] * np.array(read_dir_ras),
        voxel_size[1] * np.array(phase_dir_ras),
        voxel_size[2] * np.array(slice_dir_ras)
    ])
    
    # Construct affine matrix
    affine = np.eye(4)
    affine[:3, :3] = rotation_scaling_matrix
    affine[:3, 3] = position_ras  # Set translation
    
    # # Construct rotation-scaling matrix
    # rotation_scaling_matrix = np.column_stack([
    #     voxel_size[0] * np.array(read_dir),
    #     voxel_size[1] * np.array(phase_dir),
    #     voxel_size[2] * np.array(slice_dir)
    # ])
    
    # # Construct affine matrix
    # affine = np.eye(4)
    # affine[:3, :3] = rotation_scaling_matrix
    # affine[:3, 3] = position  # Set translation
    
    return affine

def compute_nifti_affine_from_meta(image_header, meta_attributes):
    matrix_size   = np.array([meta_attributes.encoding[0].reconSpace.matrixSize    .x, meta_attributes.encoding[0].reconSpace.matrixSize    .y, meta_attributes.encoding[0].reconSpace.matrixSize    .z])
    field_of_view = np.array([meta_attributes.encoding[0].reconSpace.fieldOfView_mm.x, meta_attributes.encoding[0].reconSpace.fieldOfView_mm.y, meta_attributes.encoding[0].reconSpace.fieldOfView_mm.z])
    
    dicomDset = pydicom.dataset.Dataset.from_json(base64.b64decode(meta[0]['DicomJson']))
    
    # Extract necessary fields from the MRD Image Header
    position_lps = image_header.position            # [px, py, pz] in LPS
    read_dir_lps = meta_attributes.ImageRowDir      # [rx, ry, rz]
    phase_dir_lps = meta_attributes.ImageColumnDir  # [px, py, pz]
    
    # Compute slice_dir as cross-product of read_dir and phase_dir
    slice_dir_lps = np.cross(read_dir_lps, phase_dir_lps)
    
    # Convert position and directions from LPS to RAS
    position_ras = [-position_lps[0], -position_lps[1], position_lps[2]]
    read_dir_ras = [-read_dir_lps[0], -read_dir_lps[1], read_dir_lps[2]]
    phase_dir_ras = [-phase_dir_lps[0], -phase_dir_lps[1], phase_dir_lps[2]]
    slice_dir_ras = [-slice_dir_lps[0], -slice_dir_lps[1], slice_dir_lps[2]]
    
    # Compute voxel sizes
    voxel_size = [fov / size for fov, size in zip(field_of_view, matrix_size)]
    
    # Construct rotation-scaling matrix in RAS coordinates
    rotation_scaling_matrix = np.column_stack([
        voxel_size[0] * np.array(read_dir_ras),
        voxel_size[1] * np.array(phase_dir_ras),
        voxel_size[2] * np.array(slice_dir_ras)
    ])
    
    # Construct affine matrix
    affine = np.eye(4)
    affine[:3, :3] = rotation_scaling_matrix
    affine[:3, 3] = position_ras  # Set translation in RAS coordinates
    
    return affine

def compute_nifti_affine_from_dicom(meta):

    dicomDset = pydicom.dataset.Dataset.from_json(base64.b64decode(meta['DicomJson']))
    
    # Extract row and column direction cosines from ImageOrientationPatient
    row_cosine = np.array(dicomDset.ImageOrientationPatient[:3])      # First 3 values
    column_cosine = np.array(dicomDset.ImageOrientationPatient[3:])   # Last 3 values
    
    # Compute slice direction as the cross product of row and column cosines
    slice_cosine = np.cross(row_cosine, column_cosine)
    
    # Compute voxel size scaling for each direction
    voxel_size_x, voxel_size_y = np.array(dicomDset.PixelSpacing)
    voxel_size_z = dicomDset.SliceThickness
    
    # Scale the directional cosines by the voxel sizes
    row_cosine_scaled = row_cosine * voxel_size_x
    column_cosine_scaled = column_cosine * voxel_size_y
    slice_cosine_scaled = slice_cosine * voxel_size_z
    
    # Create rotation-scaling matrix
    rotation_scaling_matrix = np.column_stack([row_cosine_scaled, column_cosine_scaled, slice_cosine_scaled])
    
    # Construct the full affine matrix in LPS
    lps_affine = np.eye(4)
    lps_affine[:3, :3] = rotation_scaling_matrix
    lps_affine[:3, 3] = dicomDset.ImagePositionPatient # Translation component
    
    # LPS to RAS transformation matrix
    # lps_to_ras = np.diag([-1, -1, 1, 1])
    lps_to_ras = np.diag([1, 1, 1, 1])
    
    # Convert the LPS affine to RAS affine
    ras_affine = lps_to_ras @ lps_affine

    return ras_affine