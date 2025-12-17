#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Interpolate pixels identified in a mask."""

try:
    from maskfill import maskfill
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "The 'teareduce.cleanest' module requires the 'ccdproc' and 'maskfill' packages. "
        "Please install teareduce with the 'cleanest' extra dependencies: "
        "`pip install teareduce[cleanest]`."
    ) from e
import numpy as np
from rich import print
from scipy import ndimage
from tqdm.notebook import tqdm

from .dilatemask import dilatemask
from .interpolation_x import interpolation_x
from .interpolation_y import interpolation_y
from .interpolation_a import interpolation_a


def interpolate(data, mask_crfound, dilation=0, interp_method=None, npoints=None, degree=None, debug=False):
    """Interpolate pixels identified in a mask.

    The original data and mask are not modified. A copy of both
    arrays are created and returned with the interpolated pixels.

    Cosmic-ray pixels are initially dilated by the specified number
    of pixels. The resulting flagged pixels are grouped into cosmic-ray
    features. Each cosmic-ray feature is then interpolated using the
    specified interpolation method.

    Note that the interpolation methods `lacosmic` and `auxfile`,
    available in the interactive use of **tea-cleanest** are not
    implemented in this function because both cases are simply
    an inmediate replacement of the cosmic ray pixels in the data
    array by the corresponding pixels in another array using the
    mask array. Therefore, these two methods do not require any
    interpolation algorithm.

    Parameters
    ----------
    data : 2D numpy.ndarray
        The image data array to be processed.
    mask_crfound : 2D numpy.ndarray of bool
        A boolean mask array indicating which pixels are flagged
        and need to be interpolated (True = pixel to be fixed).
    dilation : int, optional
        The number of pixels to dilate the masked pixels before
        interpolation.
    interp_method : str, optional
        The interpolation method to use. Options are:
        'x' : Polynomial interpolation in the X direction using neighbor pixels
        'y' : Polynomial interpolation in the Y direction using neighbor pixels
        's' : Surface fit (degree 1) interpolation using neighbor pixels
        'd' : Median of neighbor pixels interpolation using neighbor pixels
        'm' : Mean of neighbor pixels interpolation using neighbor pixels
        'k' : Maskfill method, as described in van Dokkum & Pasha (2024)
    npoints : int, optional
        The number of points to use for interpolation. This
        parameter is relevant for 'x', 'y', 's', 'd', and 'm' methods.
    degree : int, optional
        The degree of the polynomial to fit. This parameter is
        relevant for 'x' and 'y' methods.
    debug : bool, optional
        If True, print debug information and enable tqdm progress bar.

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
    if interp_method not in ["x", "y", "s", "d", "m", "k"]:
        raise ValueError(f"Unknown interp_method: {interp_method}")
    if interp_method in ["x", "y", "s", "d", "m"] and npoints is None:
        raise ValueError("npoints must be specified for the chosen interp_method.")
    if interp_method in ["x", "y"] and degree is None:
        raise ValueError("degree must be specified for the chosen interp_method.")
    if data.shape != mask_crfound.shape:
        raise ValueError("data and mask_crfound must have the same shape.")

    # Apply dilation to the cosmic ray mask if needed
    if dilation > 0:
        updated_mask_crfound = dilatemask(mask_crfound, dilation)
    else:
        updated_mask_crfound = mask_crfound.copy()

    # Create a mask to keep track of cleaned pixels
    mask_fixed = np.zeros_like(mask_crfound, dtype=bool)

    # Determine number of CR features
    structure = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    cr_labels, num_features = ndimage.label(updated_mask_crfound, structure=structure)
    if debug:
        sdum = str(np.sum(updated_mask_crfound))
        print(f"Number of cosmic ray pixels to be cleaned: {sdum}")
        print(f"Number of cosmic rays (grouped pixels)...: {num_features:>{len(sdum)}}")

    # Fix cosmic rays using the specified interpolation method
    cleaned_data = data.copy()
    if interp_method == "k":
        smoothed_output, _ = maskfill(input_image=data, mask=mask_crfound, size=3, operator="median", smooth=True)
        cleaned_data[mask_crfound] = smoothed_output[mask_crfound]
        mask_fixed[mask_crfound] = True
        num_cr_cleaned = num_features
    elif interp_method in ["x", "y", "s", "d", "m"]:
        num_cr_cleaned = 0
        for cr_index in tqdm(range(1, num_features + 1), disable=not debug, leave=True):
            if interp_method in ["x", "y"]:
                if 2 * npoints <= degree:
                    raise ValueError("2*npoints must be greater than degree for polynomial interpolation.")
                if interp_method == "x":
                    interp_func = interpolation_x
                else:
                    interp_func = interpolation_y
                interpolation_performed, _, _ = interp_func(
                    data=cleaned_data,
                    mask_fixed=mask_fixed,
                    cr_labels=cr_labels,
                    cr_index=cr_index,
                    npoints=npoints,
                    degree=degree,
                )
                if interpolation_performed:
                    num_cr_cleaned += 1
            elif interp_method in ["s", "d", "m"]:
                if interp_method == "s":
                    method = "surface"
                elif interp_method == "d":
                    method = "median"
                elif interp_method == "m":
                    method = "mean"
                interpolation_performed, _, _ = interpolation_a(
                    data=cleaned_data,
                    mask_fixed=mask_fixed,
                    cr_labels=cr_labels,
                    cr_index=cr_index,
                    npoints=npoints,
                    method=method,
                )
                if interpolation_performed:
                    num_cr_cleaned += 1
    else:
        raise ValueError(f"Unknown interpolation method: {interp_method}")

    if debug:
        print(f"Number of cosmic rays cleaned............: {num_cr_cleaned:>{len(sdum)}}")

    return cleaned_data, mask_fixed
