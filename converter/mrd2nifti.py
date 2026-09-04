"""Converts MRD images into NIfTI files."""

import argparse
from collections import defaultdict
import logging
import os

import ismrmrd
import numpy as np
import nibabel as nib

from converter.utils import check_MRDfile, slice_pos
from utils.img_array import flatten


IMTYPE_LABEL = {
    ismrmrd.IMTYPE_MAGNITUDE: "M",
    ismrmrd.IMTYPE_PHASE    : "P",
    ismrmrd.IMTYPE_REAL     : "R",
    ismrmrd.IMTYPE_IMAG     : "I",
    ismrmrd.IMTYPE_COMPLEX  : "C"
}


def rescale_phase(vol: np.ndarray, meta: dict) -> np.ndarray:
    """
    Rescale a phase-image volume from stored integer units to radians.
 
    When applicable, applies the affine transform ``vol * slope + intercept`` 
    using ``RescaleSlope``/``RescaleIntercept`` from the image Meta attributes, 
    casting the result to float32.
 
    Parameters
    ----------
    vol : np.ndarray
        Volume to rescale.
    meta : dict
        Metadata dict as returned by assemble_volume(). Relevant keys:
            image_type       (str)          : rescaling only applies for "P".
            RescaleSlope     (str or float, optional)
            RescaleIntercept (str or float, optional)
 
    Returns
    -------
    np.ndarray
        The rescaled float32 volume if ``image_type == "P"`` and at
        least one of RescaleSlope/RescaleIntercept was found; otherwise
        the original ``vol`` unchanged.

    """
    if meta.get("image_type") != "P":
        return vol

    slope     = meta.get("RescaleSlope")
    intercept = meta.get("RescaleIntercept")

    if slope is None and intercept is None:
        logging.warning("No RescaleSlope/RescaleIntercept in the images meta. No conversion to radian.")
        return vol

    slope     = float(slope) if slope is not None else 1.0
    intercept = float(intercept) if intercept is not None else 0.0

    return (vol.astype(np.float32) * slope + intercept)


def build_affine(first_img: ismrmrd.Image) -> np.ndarray:
    """
    Build a 4x4 RAS affine matrix from MRD ImageHeader geometry fields.
 
    The affine encodes the mapping from voxel indices [i, j, k] to millimetre
    coordinates in RAS space (Right-Anterior-Superior), as expected by NIfTI.
 
    Parameters
    ----------
    first_img : ismrmrd.Image
        The image with the lowest slice_pos in the stack (index 0 in the
        assembled array). Its header provides all geometry fields.
 
    Returns
    -------
    np.ndarray
        4x4 float64 affine matrix in RAS space.
    """
    
    img_header = first_img.getHead()

    fov        = np.array(img_header.field_of_view[:3])    # [x, y, z] in mm
    matrix     = np.array(img_header.matrix_size[:3])      # [x, y, z]
    voxel_size = fov / matrix                              # mm per voxel

    # --- Affine : LPS (MRD) to RAS (NIfTI) -------------------------------
    def lps_to_ras(v):
        return np.array([-v[0], -v[1], v[2]])

    read_dir  = lps_to_ras(img_header.read_dir)
    phase_dir = lps_to_ras(img_header.phase_dir)
    slice_dir = lps_to_ras(img_header.slice_dir)
    position  = lps_to_ras(img_header.position)

    # Construct rotation-scaling matrix
    # The stack_dir factor on the slice column ensures that the affine step
    # matches the actual direction of increasing voxel index in the assembled
    # array, correcting for acquisitions where slice_dir points opposite to
    # the stacking order.
    rotation_scaling_matrix = np.column_stack([
        voxel_size[0] * np.array(read_dir),
        voxel_size[1] * np.array(phase_dir),
        voxel_size[2] * np.array(slice_dir)
    ])

    affine         = np.eye(4)
    affine[:3, :3] = rotation_scaling_matrix
    affine[:3, 3]  = position
    
    return affine


def detect_extra_dims(images: list[ismrmrd.Image]) -> list[str]:
    """
    Detect which ImageHeader index fields vary across a set of images.
 
    Inspects the following fields on every image header, in order:
    contrast, phase, repetition, set, average
 
    A field is considered an active extra dimension when at least two distinct
    integer values are found across all images. The returned list preserves
    the inspection order, which also determines the axis order in the assembled
    NIfTI volume (e.g. [x, y, z, echo, rep] for contrast + repetition).
 
    Parameters
    ----------
    images : list of ismrmrd.Image
        All images to inspect. Typically the full contents of one MRD
        image sub-group, after filtering by image_type.
 
    Returns
    -------
    list of str
        Ordered list of active dimension field names, e.g. ["contrast"] for
        a multi-echo series, or ["contrast", "repetition"] for multi-echo
        dynamic data. Empty list if all images share the same index values.
    """

    possible_dims = ["contrast", "phase", "repetition", "set", "average"]
    active = []

    for dim in possible_dims:
        values = {int(getattr(img.getHead(), dim, 0)) for img in images}
        if len(values) > 1:
            active.append(dim)
    return active


def assemble_volume(images: list[ismrmrd.Image], extra_dims: list[str]) :
    """
    Reassemble a flat list of 2D MRD images into a NIfTI-ready ndarray.
 
    Each MRD image carries one 2D slice with shape [cha, z, y, x] where
    cha=1 and z=1 for standard reconstructed images. This function:

      1. Sorts slices by their position along the slice normal (slice_pos).
      2. For each active extra dimension, discovers all unique index values
         and maps them to contiguous array indices.
      3. Allocates a volume of shape [n_slices, ny, nx, (extra_sizes)].
      4. Fills each slot using the slice position and extra-dimension indices
         read from the ImageHeader.
      5. Transposes the volume from [z, y, x, ...] to [x, y, z, ...] to
         match the NIfTI storage convention.
      6. Builds the RAS affine from the first image (lowest slice_pos).
      7. Collects scalar metadata from the first image's Meta attributes.
 
    Parameters
    ----------
    images : list of ismrmrd.Image
        All images for one series and one image_type. Multi-channel and
        multi-z-per-header images are not supported and must be filtered
        out before calling this function.
    extra_dims : list of str
        Ordered list of ImageHeader field names to stack as extra axes,
        as returned by detect_extra_dims. Pass [] for a plain 3D volume.
 
    Returns
    -------
    vol : np.ndarray
        Assembled volume with shape [x, y, z] or [x, y, z, d0, d1, ...].
    affine : np.ndarray
        4x4 RAS affine matrix built from the first image's geometry.
    meta : dict
        Scalar metadata with the following guaranteed keys:
            image_type   (str)  : one of M, P, R, I, C
            series_index (int)  : image_series_index from the header
            extra_dims   (list) : copy of the extra_dims argument
            extra_values (list) : list of sorted value lists, one per extra dim
            stack_dir    (float): +1.0 or -1.0, as returned by detect_stack_dir
        Optional keys populated from Meta attributes when present:
            EchoTime, InversionTime, RepetitionTime,
            SeriesDescription, SequenceDescription,
            WindowCenter, WindowWidth
    """
    if not images: 
        raise ValueError("Empty image list")
    
    # Slice indexation
    slice_positions = sorted({slice_pos(img) for img in images})
    n_slices   = len(slice_positions)
    pos_to_idx = {p: i for i, p in enumerate(slice_positions)}

    # Extra dimension indexation
    extra_value_sets = []
    for dim in extra_dims:
        vals = sorted({int(getattr(img.getHead(), dim, 0)) for img in images})
        extra_value_sets.append(vals)

    extra_sizes  = [len(v) for v in extra_value_sets]
    extra_to_idx = [{v: i for i, v in enumerate(vals)} for vals in extra_value_sets]

    # Allocate volume
    sample = np.squeeze(images[0].data)
    ny, nx = sample.shape
    dtype  = images[0].data.dtype

    vol = np.zeros((n_slices, ny, nx, *extra_sizes), dtype=dtype)

    # Determine stack direction before building the affine
    # stack_dir  = detect_stack_dir(images)
    first_img  = min(images, key=slice_pos)
    affine     = build_affine(first_img)

    # Populate volume
    for img in images:
        s_idx      = pos_to_idx[slice_pos(img)]
        slice_data = np.squeeze(img.data)   # [y, x]
        e_idxs     = tuple(
            extra_to_idx[k][int(getattr(img.getHead(), dim, 0))]
            for k, dim in enumerate(extra_dims)
        )
        vol[(s_idx, slice(None), slice(None), *e_idxs)] = slice_data

    # Transpose to NIfTI convention (z,y,x,...) -> (x,y,z,...)
    n_extra = len(extra_dims)
    perm    = (2, 1, 0, *range(3, 3 + n_extra))
    vol     = np.transpose(vol, perm)

    # Get metadata
    h = first_img.getHead()
    meta = {
        "image_type"  : IMTYPE_LABEL.get(int(h.image_type), "M"),
        "series_index": int(h.image_series_index),
        "extra_dims"  : extra_dims,
        "extra_values": extra_value_sets,
    }
    try:
        attr = ismrmrd.Meta.deserialize(images[0].attribute_string)
        for key in ("EchoTime", "InversionTime", "RepetitionTime",
                    "SeriesDescription", "SequenceDescription", 
                    "WindowCenter", "WindowWidth", "RescaleSlope",
                    "RescaleIntercept"):
            if attr.get(key) is not None:
                meta[key] = attr[key]
                logging.debug(f" DEBUG: {key} = {meta[key]}")
    except Exception:
        pass

    return vol, affine, meta


#### NIfTI construction #######################################################

def make_nifti(data: np.ndarray[ismrmrd.Image], affine: np.ndarray, meta: dict) -> nib.Nifti1Image:
    """
    Wrap a numpy volume and RAS affine into a Nifti1Image with a populated header.

    Sets voxel dimensions (zooms), spatial and temporal units, and the 80 character
    description field. For 4D or higher volumes, extra-dimension zooms are filled
    from metadata when available (TR for repetition, TE for contrast/echo),
    defaulting to 1.0 otherwise.

    Parameters
    ----------
    data : np.ndarray
        Volume array with shape [x, y, z] or [x, y, z, d0, d1, ...].
        The first three axes must correspond to the RAS spatial axes encoded
        in the affine.
    affine : np.ndarray
        4x4 RAS affine matrix.
    meta : dict
        Metadata dict as returned by assemble_volume. The following keys are 
        used when present:
        extra_dims        (list of str) : names of extra axes beyond z
        RepetitionTime    (float)       : TR in ms, used for "repetition" zoom
        EchoTime          (float)       : TE in ms, used for "contrast" zoom
        SequenceDescription (str)       : written to the descrip header field

    Returns
    -------
    nib.Nifti1Image
        NIfTI image ready to be saved with nib.save().
    """

    img = nib.Nifti1Image(data, affine)
    header = img.header

    if data.ndim > 3:
        base_zooms  = list(header.get_zooms()[:3])
        extra_zooms = []
        for dim in meta.get("extra_dims", []):
            if dim == "repetition":
                extra_zooms.append(float(meta.get("RepetitionTime", 1.0)))
            elif dim == "contrast":
                extra_zooms.append(float(meta.get("EchoTime", 1.0)))
            else:
                extra_zooms.append(1.0)
        header.set_zooms(base_zooms + extra_zooms)

    header.set_xyzt_units("mm", "sec")

    desc_parts = []
    if meta.get("SequenceDescription"):
        desc_parts.append(str(meta["SequenceDescription"]))
    if meta.get("extra_dims"):
        desc_parts.append("+".join(meta["extra_dims"]))
    header["descrip"] = ", ".join(desc_parts)[:80].encode()

    return img


###############################################################################

def nifti_from_image_array(image_array: np.ndarray[ismrmrd.Image] | list[ismrmrd.Image], outfolder: str, extra_dims: list[str] | None = None) -> str:
    """
    Convert an MRDImageArray into a NIfTI image and save it to disk.
 
    MRDImageArray is a numpy object array of shape:
    [slice, contrast, average, phase, repetition, set, image_type]
    where each populated cell holds an ismrmrd.Image. Unpopulated cells
    are None and are silently ignored.
 
    The output filename is derived from the SequenceDescription metadata
    field and the image type label (M, P, R, I, C). The file is written
    to outfolder.
 
    Parameters
    ----------
    image_array : np.ndarray (dtype=object) or list of ismrmrd.Image
        nD MRDImageArray as received by process_image in a reconstruction
        server. Must contain at least one non-None cell. Or directly a list
        of MRD Images.
    outfolder : str
        Directory where NIfTI files will be written.
    extra_dims : list of str or None
        Extra dimension field names to use as additional NIfTI axes.
        If None (default), dimensions are auto-detected via detect_extra_dims.
        Pass [] to force a plain 3D volume.

    Returns
    -------
    str
        path to NIfTI image saved.
    """

    # Flatten the object array and drop None cells
    if isinstance(image_array, np.ndarray):
        images = flatten(image_array)
        if not images:
            raise ValueError("nifti_from_image_array: all cells are None")
    else :
        images = image_array
    if extra_dims is None:
        extra_dims = detect_extra_dims(images)
    data, affine, meta = assemble_volume(images, extra_dims)
    data = rescale_phase(data, meta)
    nifti_image = make_nifti(data, affine, meta)

    sequence_desc = str(meta.get("SequenceDescription", "")).strip()
    img_type = images[0].getHead().image_type
    serie_number = images[0].getHead().image_series_index
    type_label = IMTYPE_LABEL.get(img_type, "X")

    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    if sequence_desc:
        outfile = "%s_%s_%s.nii" % (serie_number, sequence_desc, type_label)
    else:
        outfile = "%s_%s.nii" % (serie_number, type_label)

    out_path = os.path.join(outfolder, outfile)
    nib.save(nifti_image, out_path)
    logging.info(f"{outfile} - shape={str(data.shape)}")

    return out_path

#### CLI ######################################################################

def main(args):
    """
    Convert all image groups in an MRD file to NIfTI files.
 
    For each image sub-group found in the selected HDF5 group:
    - Reads all images, skipping unsupported formats (RGB, multi-channel, 
    multi-z-per-header).
    - Separates images by image_type (magnitude, phase, real, imag, complex) 
    so that different image types always produce separate NIfTI files, 
    even if they share the same MRD sub-group.
    - Auto-detects extra dimensions (contrast, phase, repetition, set, 
    average) unless --no-auto is set.
    - Assembles each (type, extra_dims) combination into a volume and saves 
    it as a .nii file.
 
    Output filenames follow the pattern:
        {series_index}_{SequenceDescription}_{type_label}.nii
    or, if no SequenceDescription is available:
        {series_index}_{type_label}.nii
 
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments. Expected attributes:
            filename   (str)  : path to the MRD (.h5) input file
            in_group   (str)  : HDF5 group to read (None = last group)
            out_folder (str)  : output directory (None = filename stem)
            no_auto    (bool) : if True, disable extra-dim auto-detection
            verbose    (bool) : if True, set log level to DEBUG
    """

    in_group = check_MRDfile(args.filename, args.in_group, args.out_folder)
    if not in_group :
        return

    # Iterate over image sub-group
    dset = ismrmrd.Dataset(args.filename, in_group, False)
    groups = dset.list()

    if ('xml' in groups):
        xml_header = dset.read_xml_header()
        xml_header = xml_header.decode("utf-8")
        mrdHead = ismrmrd.xsd.CreateFromDocument(xml_header)
    else:
        mrdHead = ismrmrd.xsd.ismrmrdHeader()

    filesWritten = 0

    for group in groups:
        if group in ("config", "config_file", "xml", "configAdditional"):
            continue

        n_images = dset.number_of_images(group)
        logging.info(f"Reading images from '{in_group}' ({n_images} images)")

        # --- Read all images in a group -------------------------------------
        images = []
        for imgNum in range(n_images):
            mrdImg = dset.read_image(group, imgNum)
            meta = ismrmrd.Meta.deserialize(mrdImg.attribute_string)
        
            # Skip unsupported (multi-channel, multi-slice per header, RGB data)
            if ((mrdImg.data.shape[0] == 3) and (mrdImg.getHead().image_type == 6)):
                # RGB images
                logging.warning("RGB data not yet supported")
                continue
            else:
                if (mrdImg.data.shape[1] != 1):
                    logging.warning("Multi-slice data not yet supported")
                    continue

                if (mrdImg.data.shape[0] != 1):
                    logging.warning("Multi-channel data not yet supported")
                    continue
            
            images.append(mrdImg)
        
        if not images:
            logging.warning("No usable images. Skipping group")
            continue

        # --- Separate by image_type if multiple types coexist in this group --
        type_buckets: dict[int, list] = defaultdict(list)
        for img in images:
            type_buckets[int(img.getHead().image_type)].append(img)

        for img_type, type_images in sorted(type_buckets.items()):
            type_label = IMTYPE_LABEL.get(img_type, "X")

            if args.no_auto:
                extra_dims = []
            else:
                extra_dims = detect_extra_dims(type_images)

            try:
                data, affine, meta = assemble_volume(type_images, extra_dims)
            except Exception as e:
                logging.error(e)
                continue

            data = rescale_phase(data, meta)
            nifti_image = make_nifti(data, affine, meta)

            # Build filename
            sequence_desc = str(meta.get("SequenceDescription", "")).strip()
            serie_number = type_images[0].getHead().image_series_index

            if sequence_desc:
                outfile = "%s_%s_%s.nii" % (serie_number, sequence_desc, type_label)
            else:
                outfile = "%s_%s.nii" % (serie_number, type_label)

            out_path = os.path.join(args.out_folder, outfile)
            nib.save(nifti_image, out_path)
            logging.info(f"{outfile} - shape={str(data.shape)}")
            filesWritten += 1

    dset.close()
    logging.info(f"Wrote {filesWritten} NIfTI file(s) to {args.out_folder}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MRD image file to NIfTI files",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('filename',                                                 help="Input MRD (.h5) file")
    parser.add_argument('-g', '--in-group',                                         help="Input data group (default: last group)")
    parser.add_argument('-o', '--out-folder',                                       help="Output folder")
    parser.add_argument('--no-auto',        action='store_true', default=False,     help="Disable automatic extra-dimension detection, write a single 3D volume per series")
    parser.add_argument('-v', '--verbose',  action='store_true',                    help='Verbose output.')

    args = parser.parse_args()

    if args.verbose:
        logLevel = logging.DEBUG
    else:
        logLevel = logging.INFO

    # setup logging
    logging.basicConfig(
        level=logLevel,
        format=f"%(levelname)8s: %(message)s"
    )
    
    main(args)
