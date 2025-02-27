import ismrmrd
import os
import logging
import traceback
import numpy as np
import numpy.fft as fft
import xml.dom.minidom
import base64
#import mrdhelper
import constants
from time import perf_counter
#import pydicom

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


def synthstrip(data, voxelsize, affine):
    # necessary for speed gains (I think)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    device = torch.device('cpu')
    device_name = 'CPU' # force CPU usage, disable GPU

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
    print(f'modelfile={modelfile}')
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


def compute_nifti_affine(image_header, voxel_size):

    # Extract necessary fields
    position      = image_header.position
    read_dir      = image_header.read_dir
    phase_dir     = image_header.phase_dir
    slice_dir     = image_header.slice_dir 

    # Convert from LPS to RAS
    position_ras  = [ -position[0],  -position[1],  position[2]]
    read_dir_ras  = [ -read_dir[0],  -read_dir[1],  read_dir[2]]
    phase_dir_ras = [-phase_dir[0], -phase_dir[1], phase_dir[2]]
    slice_dir_ras = [-slice_dir[0], -slice_dir[1], slice_dir[2]]

    # Construct rotation-scaling matrix
    rotation_scaling_matrix = np.column_stack([
        voxel_size[0] * np.array( read_dir_ras),
        voxel_size[1] * np.array(phase_dir_ras),
        voxel_size[2] * np.array(slice_dir_ras)
    ])
    
    # Construct affine matrix
    affine = np.eye(4)
    affine[:3, :3] = rotation_scaling_matrix
    affine[:3,  3] = position_ras
    
    return affine


class ImageFactory:

    def __init__(self) -> None:
        self.image_series_index_offset    : int       =  0
        self.ImageProcessingHistory       : list[str] = []
        self.SequenceDescriptionAdditional: list[str] = []
        self.mrdHeader                    : list[ismrmrd.ImageHeader]
        self.mrdMeta                      : list[ismrmrd.Meta]

    @staticmethod
    def MRD5Dto3D(data_mrd5D: np.array) -> np.array:

        # Reformat data to [y x z cha img], i.e. [row col] for the first two dimensions
        data_mrd5D = data_mrd5D.transpose((3, 4, 2, 1, 0))

        logging.debug("Original image data is size %s" % (data_mrd5D.shape,))

        data_5d = data_mrd5D.astype(np.float64)

        # Reformat data from [y x z cha img] to [y x img]
        data_3d = data_5d[:,:,0,0,:]
        
        return data_3d
    
    def ANTsImageToMRD(self, ants_image: ants.ants_image.ANTsImage, history: str|list[str] = '', seq_descrip_add: str = '') -> list[ismrmrd.Image]:

        if   type(history) is list:
            self.ImageProcessingHistory += history
        elif type(history) is str and len(history)>0:
            self.ImageProcessingHistory.append(history)
        else:
            TypeError('bad `history` type')

        if len(seq_descrip_add)>0:
            self.image_series_index_offset += 1
            self.SequenceDescriptionAdditional.append(seq_descrip_add)

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
            oldHeader = self.mrdHeader[iImg]
            oldHeader.data_type = imagesOut[iImg].data_type

            # Set the image_type to match the data_type for complex data
            if (imagesOut[iImg].data_type == ismrmrd.DATATYPE_CXFLOAT) or (imagesOut[iImg].data_type == ismrmrd.DATATYPE_CXDOUBLE):
                oldHeader.image_type = ismrmrd.IMTYPE_COMPLEX

            oldHeader.image_series_index += self.image_series_index_offset

            imagesOut[iImg].setHead(oldHeader)

            # Create a copy of the original ISMRMRD Meta attributes and update
            tmpMeta = self.mrdMeta[iImg]
            tmpMeta['DataRole']                       = 'Image'
            if len(self.ImageProcessingHistory       ) > 0: tmpMeta['ImageProcessingHistory'       ] = self.ImageProcessingHistory
            if len(self.SequenceDescriptionAdditional) > 0: tmpMeta['SequenceDescriptionAdditional'] = '_'.join(self.SequenceDescriptionAdditional)
            tmpMeta['Keep_image_geometry']            = 1

            metaXml = tmpMeta.serialize()
            logging.debug("Image MetaAttributes: %s", xml.dom.minidom.parseString(metaXml).toprettyxml())
            logging.debug("Image data has %d elements", imagesOut[iImg].data.size)

            imagesOut[iImg].attribute_string = metaXml

        logging.info(f'ImageFactory: {self.image_series_index_offset=}')
        logging.info(f'ImageFactory: {self.ImageProcessingHistory=}')
        logging.info(f'ImageFactory: {self.SequenceDescriptionAdditional=}')

        return imagesOut

def check_OR_arguments(config, arg_name: str, arg_type: type, arg_default: any) -> any:
    arg_value = arg_default

    if ('parameters' in config) and (arg_name in config['parameters']):
        logging.info(f"found config['parameters']['{arg_name}'] : type={type(config['parameters'][arg_name])} content={config['parameters'][arg_name]}")
        arg_value =  config['parameters'][arg_name]
    else:
        logging.warning(f"config['parameters']['{arg_name}'] NOT FOUND !!")

    # in OR, the config only provides strings, so need to cast to the correct type
    if arg_type is str:
        pass
    elif arg_type is bool:
        if type(arg_value) is not bool:
            if   arg_value == 'True' : arg_value = True
            elif arg_value == 'False': arg_value = False
            else: raise ValueError(f"{arg_name} is detected as `str` but is not 'True' or 'False' ! Cannot cast it to `bool`")
    elif arg_type is int:
        if type(arg_value) is not int:
            arg_value = int(arg_value)
    elif arg_type is float:
        if type(arg_value) is not float:
            arg_value = float(arg_value)
    else:
        raise TypeError('wrong type in the config)')

    logging.info(f'{arg_name} = {arg_value}')
    return arg_value

    
def process_image(images, connection, config, metadata):
    
    if len(images) == 0:
        return []

    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    logging.debug("Processing data with %d images of type %s", len(images), ismrmrd.get_dtype_from_data_type(images[0].data_type))

    # OR parameters
    BrainMaskConfig    = check_OR_arguments(config, 'BrainMaskConfig'   , str , 'ApplyInBrainMask')
    ANTsConfig         = check_OR_arguments(config, 'ANTsConfig'        , str , 'N4Dn'            )
    SaveOriginalImages = check_OR_arguments(config, 'SaveOriginalImages', bool, True              )

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

    # Diagnostic info
    matrix    = np.array(head[0].matrix_size  [:]) 
    fov       = np.array(head[0].field_of_view[:])
    voxelsize = fov/matrix
    read_dir  = np.array(images[0].read_dir )
    phase_dir = np.array(images[0].phase_dir)
    slice_dir = np.array(images[0].slice_dir)
    logging.info(f'MRD computed maxtrix [x y z] : {matrix   }')
    logging.info(f'MRD computed fov     [x y z] : {fov      }')
    logging.info(f'MRD computed voxel   [x y z] : {voxelsize}')
    logging.info(f'MRD read_dir         [x y z] : {read_dir }')
    logging.info(f'MRD phase_dir        [x y z] : {phase_dir}')
    logging.info(f'MRD slice_dir        [x y z] : {slice_dir}')

    imgfactory = ImageFactory()
    imgfactory.mrdHeader = head
    imgfactory.mrdMeta   = meta

    data_3d = imgfactory.MRD5Dto3D(data_mrd5D=data)
    logging.info(f'ANTs input data shape : {data_3d.shape}')
    ants_image_in = ants.from_numpy(data_3d)

    masking_args  = {}
    masking_label = ''
    if BrainMaskConfig != 'None':

        affine = compute_nifti_affine(images[0], voxelsize)

        # synthstrip
        logging.info(f'Affine for Synthstrip masking :')
        print(affine)
        data_iyx = data[:,0,0,:,:] # [img cha z y x] -> [img y x]
        data_xyi = data_iyx.transpose((2,1,0)) # [img y x] -> [x y img]
        logging.info(f'Shape of Synthstrip input : {data_xyi.shape}')
        brainmask = synthstrip(data_xyi, voxelsize, affine)
        brainmask = brainmask.transpose((1,0,2)) # [x y img] -> [y x img]
        logging.info(f'Shape of mask (before ANTs) : {brainmask.shape}')

        # !!! ants N4 has a weird conversion==unstable mask system
        # workaround : cast the input nparray from bool to float
        ants_mask = ants.from_numpy(np.array(brainmask,dtype=np.float32))
        masking_args['mask'] = ants_mask
        masking_label = '@SynthstripMask'

    # default configuration, just copy original images
    images_out = imgfactory.ANTsImageToMRD(ants_image_in) # !!! still need to "Keep_image_geometry"

    if not SaveOriginalImages:

        if BrainMaskConfig == 'SkullStripping':
            ants_image_in = ants.mask_image(ants_image_in, ants_mask)

        if ANTsConfig == 'None':
            images_out       = imgfactory.ANTsImageToMRD(ants_image_in, history=masking_label)
        
        elif ANTsConfig == 'N4':
            ants_image_n4    = ants.n4_bias_field_correction(ants_image_in, verbose=True, **masking_args)
            images_out       = imgfactory.ANTsImageToMRD(ants_image_n4, history='ANTsN4BiasFieldCorrection'+masking_label)

        elif ANTsConfig == 'Dn':
            ants_image_dn    = ants.denoise_image(ants_image_in, v=1, **masking_args)
            images_out       = imgfactory.ANTsImageToMRD(ants_image_dn, history='ANTsDenoiseImage'+masking_label)

        elif ANTsConfig == 'N4Dn':
            ants_image_n4    = ants.n4_bias_field_correction(ants_image_in, verbose=True, **masking_args)
            ants_image_n4_dn = ants.denoise_image(ants_image_n4, v=1, **masking_args)
            images_out       = imgfactory.ANTsImageToMRD(ants_image_n4_dn, history=['ANTsN4BiasFieldCorrection'+masking_label, 'ANTsDenoiseImage'+masking_label])

        elif ANTsConfig == 'DnN4':
            ants_image_dn    = ants.denoise_image(ants_image_in, v=1, **masking_args)
            ants_image_dn_n4 = ants.n4_bias_field_correction(ants_image_dn, verbose=True, **masking_args)
            images_out       = imgfactory.ANTsImageToMRD(ants_image_dn_n4, history=['ANTsDenoiseImage'+masking_label, 'ANTsN4BiasFieldCorrection'+masking_label])

    else:

        if   BrainMaskConfig == 'ApplyInBrainMask':
            images_out      += imgfactory.ANTsImageToMRD(ants_mask, history='SynthstripMask', seq_descrip_add='Brainmask')
        elif BrainMaskConfig == 'SkullStripping':
            images_out      += imgfactory.ANTsImageToMRD(ants_mask, history='SynthstripMask', seq_descrip_add='Brainmask')
            ants_image_in    = ants.mask_image(ants_image_in, ants_mask)
            imgfactory.SequenceDescriptionAdditional.pop()
            images_out      += imgfactory.ANTsImageToMRD(ants_image_in, history='Synthstripped', seq_descrip_add='SS')

        if ANTsConfig == 'None':
            pass
        
        elif ANTsConfig == 'N4':
            ants_image_n4    = ants.n4_bias_field_correction(ants_image_in, verbose=True, **masking_args)
            images_out      += imgfactory.ANTsImageToMRD(ants_image_n4, history='ANTsN4BiasFieldCorrection'+masking_label, seq_descrip_add='N4')

        elif ANTsConfig == 'Dn':
            ants_image_dn    = ants.denoise_image(ants_image_in, v=1, **masking_args)
            images_out      += imgfactory.ANTsImageToMRD(ants_image_dn, history='ANTsDenoiseImage'+masking_label, seq_descrip_add='Dn')

        elif ANTsConfig == 'N4Dn':
            ants_image_n4    = ants.n4_bias_field_correction(ants_image_in, verbose=True, **masking_args)
            images_out      += imgfactory.ANTsImageToMRD(ants_image_n4, history='ANTsN4BiasFieldCorrection'+masking_label, seq_descrip_add='N4')
            ants_image_n4_dn = ants.denoise_image(ants_image_n4, v=1, **masking_args)
            images_out      += imgfactory.ANTsImageToMRD(ants_image_n4_dn, history='ANTsDenoiseImage'+masking_label, seq_descrip_add='Dn')

        elif ANTsConfig == 'DnN4':
            ants_image_dn    = ants.denoise_image(ants_image_in, v=1, **masking_args)
            images_out      += imgfactory.ANTsImageToMRD(ants_image_dn, history='ANTsDenoiseImage'+masking_label, seq_descrip_add='Dn')
            ants_image_dn_n4 = ants.n4_bias_field_correction(ants_image_dn, verbose=True, **masking_args)
            images_out      += imgfactory.ANTsImageToMRD(ants_image_dn_n4, history='ANTsN4BiasFieldCorrection'+masking_label, seq_descrip_add='N4')

    return images_out
