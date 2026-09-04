"""Rebuilds ismrmrd.Image objects from a NIfTI volume, based on original MRD images."""

import copy
import logging

import ismrmrd
import nibabel as nib
import numpy as np

from converter.utils import slice_pos

def images_from_nifti(
    nifti_path: str,
    template_images: list[ismrmrd.Image],
    extra_dims: list[str] = [],
) -> list[ismrmrd.Image]:
    """
    Rebuild ismrmrd.Image objects from a NIfTI volume produced by an
    external tool (e.g. ROMEO), reusing geometry and metadata from the
    MRD images that were used to create the original input NIfTI.

    This assumes the external tool preserved the voxel grid exactly
    (same shape, same slice ordering, same extra-dimension ordering).

    Parameters
    ----------
    nifti_path : str
        Path to the NIfTI file to convert back (e.g. ROMEO's "unwrapped.nii").
    template_images : list of ismrmrd.Image
        The exact list of images passed to assemble_volume /
        nifti_from_image_array to create the *input* NIfTI (before ROMEO).
        Their headers and Meta are reused to rebuild the output images.
    extra_dims : list of str
        Extra dimension field names, in the same order used to build the 
        input NIfTI (e.g. ["contrast"]). Must match exactly.

    Returns
    -------
    list of ismrmrd.Image
        One image per (slice, \*extra_dims combination), with the same
        headers/positions as template_images, but with data replaced by
        the corresponding slice of the NIfTI volume.
    """
    nii  = nib.load(nifti_path)
    data = np.asarray(nii.dataobj)  # [x, y, z, *extra]

    logging.debug(f"nifti_shape = {data.shape}")

    n_extra = len(extra_dims)
    expected_ndim = 3 + n_extra
    if data.ndim != expected_ndim:
        raise ValueError(
            f"images_from_nifti: NIfTI has {data.ndim} dims, expected "
            f"{expected_ndim} for extra_dims={extra_dims}."
        )

    # Undo the (z,y,x,...) -> (x,y,z,...) transpose done in assemble_volume
    perm = (2, 1, 0, *range(3, expected_ndim))
    data_zyx = np.transpose(data, perm)
    logging.debug(f"nifti_shape after perm = {data_zyx.shape}")

    # Recompute the same indexation used when the volume was assembled
    slice_positions = sorted({slice_pos(img) for img in template_images})
    pos_to_idx = {p: i for i, p in enumerate(slice_positions)}

    extra_value_sets = []
    for dim in extra_dims:
        vals = sorted({int(getattr(img.getHead(), dim, 0)) for img in template_images})
        extra_value_sets.append(vals)
    extra_to_idx = [{v: i for i, v in enumerate(vals)} for vals in extra_value_sets]

    out_images = []
    for img in template_images:
        s_idx = pos_to_idx[slice_pos(img)]
        e_idx = tuple(
            extra_to_idx[k][int(getattr(img.getHead(), dim, 0))]
            for k, dim in enumerate(extra_dims)
        )
        slice_data = data_zyx[(s_idx, slice(None), slice(None), *e_idx)]

        new_head = copy.deepcopy(img.getHead())
        new_meta = ismrmrd.Meta.deserialize(img.attribute_string)
        new_meta["SeriesDescription"] = "ROMEOUnwrapping"
        new_meta["ImageComments"]     = "ROMEO phase unwrapping"

        for stale_key in ("RescaleSlope", "RescaleIntercept"):
            if stale_key in new_meta:
                del new_meta[stale_key]

        new_data = slice_data.reshape(1, 1, *slice_data.shape)
        new_img = ismrmrd.Image.from_array(
            new_data.astype(np.float32),
            transpose=False
        )
        new_head.data_type = ismrmrd.DATATYPE_FLOAT
        new_img.setHead(new_head)
        new_img.attribute_string = new_meta.serialize()
        out_images.append(new_img)

    return out_images
