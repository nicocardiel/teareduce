#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Polynomial nterpolation in the X direction"""

import numpy as np


def interpolation_x(data, mask_fixed, cr_labels, cr_index, npoints, degree):
    ycr_list, xcr_list = np.where(cr_labels == cr_index)
    ycr_min = np.min(ycr_list)
    ycr_max = np.max(ycr_list)
    xfit_all = []
    yfit_all = []
    interpolation_performed = False
    for ycr in range(ycr_min, ycr_max + 1):
        xmarked = xcr_list[np.where(ycr_list == ycr)]
        if len(xmarked) > 0:
            jmin = np.min(xmarked)
            jmax = np.max(xmarked)
            # mark intermediate pixels too
            for ix in range(jmin, jmax + 1):
                cr_labels[ycr, ix] = cr_index
            xmarked = xcr_list[np.where(ycr_list == ycr)]
            xfit = []
            zfit = []
            for i in range(jmin - npoints, jmin):
                if 0 <= i < data.shape[1]:
                    xfit.append(i)
                    xfit_all.append(i)
                    yfit_all.append(ycr)
                    zfit.append(data[ycr, i])
            for i in range(jmax + 1, jmax + 1 + npoints):
                if 0 <= i < data.shape[1]:
                    xfit.append(i)
                    xfit_all.append(i)
                    yfit_all.append(ycr)
                    zfit.append(data[ycr, i])
            if len(xfit) > degree:
                p = np.polyfit(xfit, zfit, degree)
                for i in range(jmin, jmax + 1):
                    if 0 <= i < data.shape[1]:
                        data[ycr, i] = np.polyval(p, i)
                        mask_fixed[ycr, i] = True
                interpolation_performed = True
            else:
                print(f"Not enough points to fit at y={ycr+1}")

    return interpolation_performed, xfit_all, yfit_all
