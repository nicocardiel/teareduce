# -*- coding: utf-8 -*-
#
# Copyright 2015-2024 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
from scipy import ndimage
from tqdm.notebook import tqdm

from .imshow import imshow
from .polfit import polfit_residuals_with_sigma_rejection


def fit_sdistortion(data, ns_min, ns_max, nc_min, nc_max,
                    median_size=None,
                    ywindow=5, degree_sdistortion=2, plots=True,
                    **kwargs):
    """Straighten 2D spectroscopic image

    Parameters
    ----------
    data : numpy array
        2D spectroscopic image suffering from S distortion.
    ns_min : int
        Initial row (array index) in the spatial direction.
    ns_max : int
        Final row (array index) in the spatial direction.
    nc_min : int
        Initial column (array index) in the spectral direction.
    nc_max : int
        Final column (array index) in the spectral direction.
    median_size : tuple of integers
        Shape that is taken from the input array, at every element
         position, to compute a median filter. Note that the tuple
         order correspond to (Yaxis, Xaxis).
    ywindow : int
        Number of pixels (spatial direction) of the window where the
        peaks are sought. It must be odd.
    degree_sdistortion : int
        Degree of the polymial employed to fit the S distortion.
    plots : bool
        If True, display intermediate plots.
    kwargs : dictionary
        Additional arguments for imshow call.

    Returns
    -------
    data_straight : numpy array
        2D spectroscopic image corrected from S distortion.
    poly_funct_peaks : `~numpy.polynomial.polynomial.Polynomial`
        Fitted polynomial.

    """

    if ywindow % 2 != 1:
        raise ValueError(f'ywindow={ywindow} must be an odd integer')
    semiwindow = (int(ywindow) - 1) // 2
    if semiwindow < 1:
        raise ValueError(f'Unexpected semiwindow={semiwindow}')

    naxis2, naxis1 = data.shape

    if median_size is None:
        data_smooth = data
    else:
        data_smooth = ndimage.median_filter(data, size=median_size)

    if 'aspect' in kwargs:
        aspect = kwargs['aspect']
        del kwargs['aspect']
    else:
        aspect = 'auto'

    if plots:
        vmin = np.min(data_smooth[ns_min:(ns_max+1), nc_min:(nc_max+1)])
        vmax = np.max(data_smooth[ns_min:(ns_max+1), nc_min:(nc_max+1)])
        fig, ax = plt.subplots(figsize=(15, 5))
        imshow(fig, ax, data, vmin=vmin, vmax=vmax, title='initial data', aspect=aspect, **kwargs)
        ax.set_ylim(ns_min-0.5, ns_max+0.5)
        plt.tight_layout()
        plt.show()

        if median_size is not None:
            fig, ax = plt.subplots(figsize=(15, 5))
            imshow(fig, ax, data_smooth, vmin=vmin, vmax=vmax, title='smoothed data', aspect=aspect, **kwargs)
            ax.set_ylim(ns_min-0.5, ns_max+0.5)
            plt.tight_layout()
            plt.show()
    else:
        vmin = None
        vmax = None

    xfit = np.arange(ns_min, ns_max + 1)
    nfit = len(xfit)

    npeaks = nc_max - nc_min + 1
    xpeak = np.zeros(npeaks, dtype=float)
    ypeak = np.zeros(npeaks, dtype=float)

    # loop in X axis: determine peaks
    for i in range(nc_min, nc_max + 1):
        yfit = data_smooth[ns_min:(ns_max + 1), i]
        jmax = np.argmax(yfit)
        j1 = jmax - semiwindow
        if j1 < 0:
            j1 = 0
            j2 = j1 + 2 * semiwindow
        else:
            j2 = j1 + 2 * semiwindow
            if j2 > nfit - 1:
                j2 = nfit - 1
                j1 = j2 - 2 * semiwindow
        poly_funct = Polynomial.fit(xfit[j1:(j2 + 1)], yfit[j1:(j2 + 1)], deg=2)
        poly_funct = Polynomial.cast(poly_funct)
        coef = poly_funct.coef
        if len(coef) == 3:
            if coef[2] != 0:
                refined_peak = -coef[1] / (2.0 * coef[2])
            else:
                refined_peak = float(jmax)
        else:
            refined_peak = float(jmax)

        xpeak[i - nc_min] = float(i)
        ypeak[i - nc_min] = refined_peak

    # fit S distortion
    poly_funct_peaks, yres, reject = polfit_residuals_with_sigma_rejection(
        x=xpeak,
        y=ypeak,
        deg=degree_sdistortion,
        times_sigma_reject=3
    )
    ypeak_mean = np.mean(ypeak[~reject])

    if plots:
        fig, ax = plt.subplots(figsize=(15, 5))
        imshow(fig, ax, data_smooth, vmin=vmin, vmax=vmax, title='fitting S distortion', aspect=aspect, **kwargs)
        ax.set_ylim(ns_min-0.5, ns_max+0.5)
        ax.plot(xpeak, ypeak, 'C0+')
        ax.plot(xpeak[reject], ypeak[reject], 'rx')
        ax.axhline(ypeak_mean, ls='--', color='C4')
        xpredict = np.arange(naxis1)
        ypredict = poly_funct_peaks(xpredict)
        ax.plot(xpredict, ypredict, 'C1-')
        plt.tight_layout()
        plt.show()

    # correct S distortion
    accum_flux = np.zeros((naxis2 + 1, naxis1))
    accum_flux[1:, :] = np.cumsum(data, axis=0)

    data_straight = np.zeros((naxis2, naxis1))
    new_pix_borders = -0.5 + np.arange(naxis2 + 1)

    for k in tqdm(range(naxis1)):
        old_pix_borders = new_pix_borders - (poly_funct_peaks(k) - ypeak_mean)
        flux_borders = np.interp(
            x=new_pix_borders,
            xp=old_pix_borders,
            fp=accum_flux[:, k],
            left=0,
            right=accum_flux[-1, k]
        )
        data_straight[:, k] = flux_borders[1:] - flux_borders[:-1]

    if plots:
        fig, ax = plt.subplots(figsize=(15, 5))
        imshow(fig, ax, data_straight, vmin=vmin, vmax=vmax, title='initial data', aspect=aspect, **kwargs)
        ax.set_ylim(ns_min-0.5, ns_max+0.5)
        plt.tight_layout()
        plt.show()

    # return result
    return data_straight, poly_funct_peaks
