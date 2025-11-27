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
from scipy.ndimage import binary_dilation


def interpolation_a(data, mask_fixed, cr_labels, cr_index, npoints, method):
    # Mask of CR pixels
    mask = (cr_labels == cr_index)
    # Dilate the mask to find border pixels
    dilated_mask = binary_dilation(mask, structure=np.ones((3, 3)), iterations=npoints)
    # Border pixels are those in the dilated mask but not in the original mask
    border_mask = dilated_mask & (~mask)
    # Get coordinates of border pixels
    yfit_all, xfit_all = np.where(border_mask)
    zfit_all = data[yfit_all, xfit_all].tolist()
    # Perform interpolation
    interpolation_performed = False
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
