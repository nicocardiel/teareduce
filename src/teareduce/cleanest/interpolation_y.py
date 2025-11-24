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


def interpolation_y(data, mask_fixed, cr_labels, cr_index, npoints, degree):
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
