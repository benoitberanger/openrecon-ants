"""Debug mode functions : logging image metadata and sending images back unmodified."""

import logging

import ismrmrd

from server.connection import Connection

# Folder for debug output files
debugFolder = "/tmp/share/debug"

def display_info_images(image: ismrmrd.Image) -> None:
    """
    Log the metadata fields of a single MRD image.
    
    Intended for debug mode only. Logs the geometric orientation,
    loop counters, image type, and all FIRST_IN_* / LAST_IN_* flags.

    The following fields are available but currently commented out:
    version, data_type, flags, measurement_uid, matrix_size, field_of_view,
    channels, position, patient_table_position, acquisition_time_stamp,
    physiology_time_stamp, user_int, user_float, attribute_string_len.
    Uncomment the relevant lines to include them in the log output.

    Parameters
    ----------
    image : ismrmrd.Image
        MRD image whose metadata is to be logged.
    
    """

    logging.info('-------------------------------------')
    logging.info('-------------IMAGE INFO--------------')
    logging.info('-------------------------------------')

    image_type = ['0', 
                  'MRD_IMTYPE_MAGNITUDE', 
                  'MRD_IMTYPE_PHASE', 
                  'MRD_IMTYPE_REAL', 
                  'MRD_IMTYPE_IMAG', 
                  'MRD_IMTYPE_COMPLEX',
                  'MRD_IMTYPE_RGB']
    
    # logging.info(f'version                  : {image.version}')
    # logging.info(f'data_type                : {image.data_type}')
    # logging.info(f'flags                    : {image.flags}')
    # logging.info(f'measurement_uid          : {image.measurement_uid}')
    # logging.info(f'matrix_size              : {image.matrix_size}')
    # logging.info(f'field_of_view            : {image.field_of_view}')
    # logging.info(f'channels                 : {image.channels}')
    # logging.info(f'position                 : L={image.position[0]} mm, P={image.position[1]} mm, S={image.position[2]} mm')
    logging.info(f'read_dir                 : {image.read_dir[0], image.read_dir[1], image.read_dir[2]}')
    logging.info(f'phase_dir                : {image.phase_dir[0], image.phase_dir[1], image.phase_dir[2]}')
    logging.info(f'slice_dir                : {image.slice_dir[0], image.slice_dir[1], image.slice_dir[2]}')
    # logging.info(f'patient_table_position   : {image.patient_table_position}')
    logging.info(f'average                  : {image.average}')
    logging.info(f'slice                    : {image.slice}')
    logging.info(f'contrast                 : {image.contrast}')
    logging.info(f'phase                    : {image.phase}')
    logging.info(f'repetition               : {image.repetition}')
    logging.info(f'set                      : {image.set}')
    # logging.info(f'aquisition_time_stamp    : {image.aquisition_time_stamp}')
    # logging.info(f'physiology_time_stamp    : {image.physiology_time_stamp}')
    logging.info(f'image_type               : {image_type[image.image_type]}')
    logging.info(f'image_index              : {image.image_index}')
    logging.info(f'image_series_index       : {image.image_series_index}')
    # logging.info(f'user_int                 : {image.user_int}')
    # logging.info(f'user_float               : {image.user_float}')
    # logging.info(f'attribute_string_len     : {image.attribute_string_len}')

    logging.info("      ***** FLAGS *****")
    logging.info(f"First in average         : {image.is_flag_set(ismrmrd.IMAGE_FIRST_IN_AVERAGE)}")
    logging.info(f"Last in average          : {image.is_flag_set(ismrmrd.IMAGE_LAST_IN_AVERAGE)}")
    logging.info(f"First in slice           : {image.is_flag_set(ismrmrd.IMAGE_FIRST_IN_SLICE)}")
    logging.info(f"Last in slice            : {image.is_flag_set(ismrmrd.IMAGE_LAST_IN_SLICE)}")
    logging.info(f"First in contrast        : {image.is_flag_set(ismrmrd.IMAGE_FIRST_IN_CONTRAST)}")
    logging.info(f"Last in contrast         : {image.is_flag_set(ismrmrd.IMAGE_LAST_IN_CONTRAST)}")
    logging.info(f"First in phase           : {image.is_flag_set(ismrmrd.IMAGE_FIRST_IN_PHASE)}")
    logging.info(f"Last in phase            : {image.is_flag_set(ismrmrd.IMAGE_LAST_IN_PHASE)}")
    logging.info(f"First in repetition      : {image.is_flag_set(ismrmrd.IMAGE_FIRST_IN_REPETITION)}")
    logging.info(f"Last in repetition       : {image.is_flag_set(ismrmrd.IMAGE_LAST_IN_REPETITION)}")
    logging.info(f"First in set             : {image.is_flag_set(ismrmrd.IMAGE_FIRST_IN_SET)}")
    logging.info(f"Last in set              : {image.is_flag_set(ismrmrd.IMAGE_LAST_IN_SET)}")
    logging.info("      ****************")


def send_back_debug(image: ismrmrd.Image, connection: Connection) -> None:
    """
    Send an image back to the client unmodified and log its metadata.

    Used in debug mode as a passthrough, no processing is applied.
    Sets Keep_image_geometry to 1 in the Meta attributes before sending
    to prevent the client from reversing the image orientation.

    Parameters
    ----------
    image : ismrmrd.Image
        MRD image to inspect and send back.
    connection : Connection
        Active MRD connection used to send the image.
    """
    display_info_images(image)

    tmpMeta = ismrmrd.Meta.deserialize(image.attribute_string)
    tmpMeta['Keep_image_geometry'] = 1
    image.attribute_string = tmpMeta.serialize()

    connection.send_image(image)
