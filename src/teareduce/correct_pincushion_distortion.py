#
# Copyright 2024 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

from astropy.io import fits
import numpy as np
from numpy.polynomial import Polynomial


def correct_pincushion_distortion(coeff_filename, data):
    """Correct pincushion distortion.

    Parameters
    ----------
    coeff_filename : str
        Name of the FITS file containing the polynomial coefficients
        of the transformation between the Y-coordinates of the central
        image column and the measured Y-coordinates (for each image
        column).
    data : `~numpy.ndarray`
        Array containing the image to be corrected.

    Returns
    -------
    data_rectified : `~numpy.ndarray`
        Rectified image.
    """

    # read FITS file with polynomial coefficients

    with fits.open(coeff_filename) as hdul:
        table = hdul[1].data

    # check dimensions
    naxis2, naxis1 = data.shape
    if naxis1 != len(table):
        raise ValueError(f'Incompatible dimensions: naxis1:{naxis1} != len(table): {len(table)}')

    # rectify image
    accum_flux = np.zeros((naxis2 + 1, naxis1))
    accum_flux[1:, :] = np.cumsum(data, axis=0)
    new_y_borders = np.arange(naxis2 + 1) - 0.5
    data_rectified = np.zeros((naxis2, naxis1))
    for i in range(naxis1):
        poly = Polynomial(coef=table[i])
        flux_borders = np.interp(
            x=new_y_borders,
            xp=poly(new_y_borders),
            fp=accum_flux[:, i],
            left=0,
            right=accum_flux[-1, i]
        )
        data_rectified[:, i] = flux_borders[1:] - flux_borders[:-1]

    return data_rectified
