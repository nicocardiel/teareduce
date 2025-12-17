#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Surface interpolation (plane fit) or median interpolation"""

import numpy as np
from rich import print

from .dilatemask import dilatemask


def interpolation_a(data, mask_fixed, cr_labels, cr_index, npoints, method):
    """Fix cosmic ray pixels using surface fit, median or mean.

    Parameters
    ----------
    data : 2D numpy.ndarray
        The image data array where cosmic rays are to be interpolated.
    mask_fixed : 2D numpy.ndarray of bool
        A boolean mask array indicating which pixels have been fixed.
    cr_labels : 2D numpy.ndarray
        An array labeling cosmic ray features.
    cr_index : int
        The index of the current cosmic ray feature to interpolate.
    npoints : int
        The number of points to use for interpolation.
    method : str
        The interpolation method to use ('surface', 'median' or 'mean').

    Returns
    -------
    interpolation_performed : bool
        True if interpolation was performed, False otherwise.
    xfit_all : list
        X-coordinates of border pixels used for interpolation.
    yfit_all : list
        Y-coordinates of border pixels used for interpolation.

    Notes
    -----
    The `data` array is modified in place with interpolated values for the
    cosmic ray pixels. This function also returns an updated `mask_fixed`
    array with interpolated pixels marked as fixed.

    It is important to highlight that contrary to what is performed when
    using the X- and Y-interpolation, this function does not fill the gaps
    between the marked pixels. Only the pixels explicitly marked as affected
    by cosmic rays are interpolated.
    """
    # Mask of CR pixels
    mask = cr_labels == cr_index
    # Dilate the mask to find border pixels
    # dilated_mask = binary_dilation(mask, structure=np.ones((3, 3)), iterations=npoints)
    dilated_mask = dilatemask(mask=mask, iterations=npoints, connectivity=1)
    # Border pixels are those in the dilated mask but not in the original mask
    border_mask = dilated_mask & (~mask)
    # Get coordinates of border pixels
    yfit_all, xfit_all = np.where(border_mask)
    zfit_all = data[yfit_all, xfit_all].tolist()
    # Perform interpolation
    interpolation_performed = False
    if method == "surface":
        if len(xfit_all) > 3:
            # Construct the design matrix for a 2D polynomial fit to a plane,
            # where each row corresponds to a point (x, y, z) and the model
            # is z = C[0]*x + C[1]*y + C[2]
            A = np.c_[xfit_all, yfit_all, np.ones(len(xfit_all))]
            # Least squares polynomial fit
            C, _, _, _ = np.linalg.lstsq(A, zfit_all, rcond=None)
            # recompute all CR pixels to take into account "holes" between marked pixels
            ycr_list, xcr_list = np.where(cr_labels == cr_index)
            for iy, ix in zip(ycr_list, xcr_list):
                data[iy, ix] = C[0] * ix + C[1] * iy + C[2]
                mask_fixed[iy, ix] = True
            interpolation_performed = True
        else:
            print("Not enough points to fit a plane")
    elif method in ["median", "mean"]:
        # Compute median of all surrounding points
        if len(zfit_all) > 0:
            if method == "median":
                zval = np.median(zfit_all)
            else:
                zval = np.mean(zfit_all)
            # recompute all CR pixels to take into account "holes" between marked pixels
            ycr_list, xcr_list = np.where(cr_labels == cr_index)
            for iy, ix in zip(ycr_list, xcr_list):
                data[iy, ix] = zval
                mask_fixed[iy, ix] = True
            interpolation_performed = True
        else:
            print("No surrounding points found for median interpolation")
    else:
        print(f"Unknown interpolation method: {method}")

    return interpolation_performed, xfit_all, yfit_all
