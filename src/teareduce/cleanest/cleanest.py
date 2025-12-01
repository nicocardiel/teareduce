#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Interpolate pixels identified in a mask."""

from scipy import ndimage
import numpy as np

from .dilatemask import dilatemask
from .interpolation_x import interpolation_x
from .interpolation_y import interpolation_y
from .interpolation_a import interpolation_a


def cleanest(data, mask_crfound, dilation=0,
             interp_method=None, npoints=None, degree=None,
             debug=False):
    """Interpolate pixels identified in a mask.

    The original data and mask are not modified. A copy of both
    arrays are created and returned with the interpolated pixels.

    Parameters
    ----------
    data : 2D numpy.ndarray
        The image data array to be processed.
    mask_crfound : 2D numpy.ndarray of bool
        A boolean mask array indicating which pixels are affected by
        cosmic rays.
    dilation : int, optional
        The number of pixels to dilate the masked pixels before
        interpolation.
    interp_method : str, optional
        The interpolation method to use. Options are:
        'x' : Polynomial interpolation in the X direction.
        'y' : Polynomial interpolation in the Y direction.
        's' : Surface fit (degree 1) interpolation.
        'd' : Median of border pixels interpolation.
        'm' : Mean of border pixels interpolation.
    npoints : int, optional
        The number of points to use for interpolation. This
        parameter is relevant for 'x', 'y', 's', 'd', and 'm' methods.
    degree : int, optional
        The degree of the polynomial to fit. This parameter is
        relevant for 'x' and 'y' methods.
    debug : bool, optional
        If True, print debug information.

    Returns
    -------
    cleaned_data : 2D numpy.ndarray
        The image data array with cosmic rays cleaned.
    mask_fixed : 2D numpy.ndarray of bool
        The updated boolean mask array indicating which pixels
        have been fixed.

    Notes
    -----
    This function has been created to clean cosmic rays without
    the need of a GUI interaction. It can be used in scripts
    or batch processing of images.
    """
    if interp_method is None:
        raise ValueError("interp_method must be specified.")
    if interp_method not in ['x', 'y', 's', 'd', 'm']:
        raise ValueError(f"Unknown interp_method: {interp_method}")
    if npoints is None:
        raise ValueError("npoints must be specified.")
    if degree is None and interp_method in ['x', 'y']:
        raise ValueError("degree must be specified for the chosen interp_method.")

    # Apply dilation to the cosmic ray mask if needed
    if dilation > 0:
        updated_mask_crfound = dilatemask(mask_crfound, dilation)
    else:
        updated_mask_crfound = mask_crfound.copy()

    # Create a mask to keep track of cleaned pixels
    mask_fixed = np.zeros_like(mask_crfound, dtype=bool)

    # Determine number of CR features
    structure = [[1, 1, 1],
                 [1, 1, 1],
                 [1, 1, 1]]
    cr_labels, num_features = ndimage.label(updated_mask_crfound, structure=structure)
    if debug:
        print(f"Number of cosmic ray pixels to be cleaned: {np.sum(updated_mask_crfound)}")
        print(f"Number of cosmic rays (grouped pixels)...: {num_features}")

    # Fix cosmic rays using the specified interpolation method
    cleaned_data = data.copy()
    num_cr_cleaned = 0
    for cr_index in range(1, num_features + 1):
        if interp_method in ['x', 'y']:
            if 2 * npoints <= degree:
                raise ValueError("2*npoints must be greater than degree for polynomial interpolation.")
            if interp_method == 'x':
                interp_func = interpolation_x
            else:
                interp_func = interpolation_y
            interpolation_performed, _, _ = interp_func(
                data=cleaned_data,
                mask_fixed=mask_fixed,
                cr_labels=cr_labels,
                cr_index=cr_index,
                npoints=npoints,
                degree=degree)
            if interpolation_performed:
                num_cr_cleaned += 1
        elif interp_method in ['s', 'd', 'm']:
            if interp_method == 's':
                method = 'surface'
            elif interp_method == 'd':
                method = 'median'
            elif interp_method == 'm':
                method = 'mean'
            interpolation_performed, _, _ = interpolation_a(
                data=cleaned_data,
                mask_fixed=mask_fixed,
                cr_labels=cr_labels,
                cr_index=cr_index,
                npoints=npoints,
                method=method
            )
            if interpolation_performed:
                num_cr_cleaned += 1
        else:
            raise ValueError(f"Unknown interpolation method: {interp_method}")

    if debug:
        print(f"Number of cosmic rays cleaned............: {num_cr_cleaned}")

    return cleaned_data, mask_fixed
