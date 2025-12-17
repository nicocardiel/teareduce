#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Polynomial nterpolation in the Y direction"""

import numpy as np
from rich import print


def interpolation_y(data, mask_fixed, cr_labels, cr_index, npoints, degree):
    """Interpolate cosmic ray affected pixels in Y direction.
    Parameters
    ----------
    data : 2D numpy.ndarray
        The image data array to be processed.
    mask_fixed : 2D numpy.ndarray of bool
        A boolean mask array indicating which pixels have been fixed.
    cr_labels : 2D numpy.ndarray
        An array labeling cosmic ray features.
    cr_index : int
        The index of the current cosmic ray feature to interpolate.
    npoints : int
        The number of points to use for interpolation.
    degree : int
        The degree of the polynomial to fit.

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

    It is important to highlight that this function assumes that at
    every x-coordinate where cosmic ray pixels are found, the pixels
    form a contiguous vertical segment. In this sense, gaps in the
    y-direction are assumed to be also part of the same cosmic ray
    feature, and the `cr_labels` array is updated accordingly. In other
    words, all the pixels between the minimum and maximum y-coordinates
    of the cosmic ray pixels at a given x-coordinate are treated as
    affected by the cosmic ray. This simplyfies the interactive marking
    of cosmic rays, as the user does not need to ensure that all pixels
    in a vertical segment are marked; marking just the extreme pixels
    is sufficient.
    """
    ycr_list, xcr_list = np.where(cr_labels == cr_index)
    xcr_min = np.min(xcr_list)
    xcr_max = np.max(xcr_list)
    xfit_all = []
    yfit_all = []
    interpolation_performed = False
    for xcr in range(xcr_min, xcr_max + 1):
        ymarked = ycr_list[np.where(xcr_list == xcr)]
        if len(ymarked) > 0:
            imin = np.min(ymarked)
            imax = np.max(ymarked)
            # mark intermediate pixels too
            for iy in range(imin, imax + 1):
                cr_labels[iy, xcr] = cr_index
            ymarked = ycr_list[np.where(xcr_list == xcr)]
            yfit = []
            zfit = []
            for i in range(imin - npoints, imin):
                if 0 <= i < data.shape[0]:
                    yfit.append(i)
                    yfit_all.append(i)
                    xfit_all.append(xcr)
                    zfit.append(data[i, xcr])
            for i in range(imax + 1, imax + 1 + npoints):
                if 0 <= i < data.shape[0]:
                    yfit.append(i)
                    yfit_all.append(i)
                    xfit_all.append(xcr)
                    zfit.append(data[i, xcr])
            if len(yfit) > degree:
                p = np.polyfit(yfit, zfit, degree)
                for i in range(imin, imax + 1):
                    if 0 <= i < data.shape[1]:
                        data[i, xcr] = np.polyval(p, i)
                        mask_fixed[i, xcr] = True
                interpolation_performed = True
            else:
                print(f"Not enough points to fit at x={xcr+1}")
                return

    return interpolation_performed, xfit_all, yfit_all
