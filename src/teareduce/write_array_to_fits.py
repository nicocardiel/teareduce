#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

from astropy.io import fits


def write_array_to_fits(data, filename, overwrite):
    """Write a single array to a FITS file.

    Parameters
    ----------
    data : numpy.ndarray
        Array to write.
    filename : str
        Path to the FITS file.
    overwrite : bool
        Whether to overwrite the existing FITS file.
    """

    hdu = fits.PrimaryHDU(data)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(filename, overwrite=overwrite)
