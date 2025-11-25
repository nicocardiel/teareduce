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


def interpolation_a(data, mask_fixed, cr_labels, cr_index, npoints, method):
    ycr_list, xcr_list = np.where(cr_labels == cr_index)
    ycr_min = np.min(ycr_list)
    ycr_max = np.max(ycr_list)
    xfit_all = []
    yfit_all = []
    zfit_all = []
    interpolation_performed = False
    # First do horizontal lines
    for ycr in range(ycr_min, ycr_max + 1):
        xmarked = xcr_list[np.where(ycr_list == ycr)]
        if len(xmarked) > 0:
            jmin = np.min(xmarked)
            jmax = np.max(xmarked)
            # mark intermediate pixels too
            for ix in range(jmin, jmax + 1):
                cr_labels[ycr, ix] = cr_index
            xmarked = xcr_list[np.where(ycr_list == ycr)]
            for i in range(jmin - npoints, jmin):
                if 0 <= i < data.shape[1]:
                    xfit_all.append(i)
                    yfit_all.append(ycr)
                    zfit_all.append(data[ycr, i])
            for i in range(jmax + 1, jmax + 1 + npoints):
                if 0 <= i < data.shape[1]:
                    xfit_all.append(i)
                    yfit_all.append(ycr)
                    zfit_all.append(data[ycr, i])
                xcr_min = np.min(xcr_list)
    # Now do vertical lines
    xcr_max = np.max(xcr_list)
    for xcr in range(xcr_min, xcr_max + 1):
        ymarked = ycr_list[np.where(xcr_list == xcr)]
        if len(ymarked) > 0:
            imin = np.min(ymarked)
            imax = np.max(ymarked)
            # mark intermediate pixels too
            for iy in range(imin, imax + 1):
                cr_labels[iy, xcr] = cr_index
            ymarked = ycr_list[np.where(xcr_list == xcr)]
            for i in range(imin - npoints, imin):
                if 0 <= i < data.shape[0]:
                    yfit_all.append(i)
                    xfit_all.append(xcr)
                    zfit_all.append(data[i, xcr])
            for i in range(imax + 1, imax + 1 + npoints):
                if 0 <= i < data.shape[0]:
                    yfit_all.append(i)
                    xfit_all.append(xcr)
                    zfit_all.append(data[i, xcr])
    if method == 'surface':
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
    elif method == 'median':
        # Compute median of all surrounding points
        if len(zfit_all) > 0:
            zmed = np.median(zfit_all)
            # recompute all CR pixels to take into account "holes" between marked pixels
            ycr_list, xcr_list = np.where(cr_labels == cr_index)
            for iy, ix in zip(ycr_list, xcr_list):
                data[iy, ix] = zmed
                mask_fixed[iy, ix] = True
            interpolation_performed = True
    else:
        print(f"Unknown interpolation method: {method}")

    return interpolation_performed, xfit_all, yfit_all
