#
# Copyright 2022-2024 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import numpy as np


def zscale(image, factor=0.25):
    """Compute z1 and z2 cuts in a similar way to Iraf.
    
    If the total number of pixels is less than 10, the function simply 
    returns the minimum and the maximum values.

    Parameters
    ----------
    image : np.ndarray
        Image array.
    factor : float
        Factor.

    Returns
    -------
    z1 : float
        Background value.
    z2 : float
        Foreground value.

    """

    # protections
    if not isinstance(image, np.ndarray):
        raise ValueError('image must be a numpy.ndarray')

    npixels = image.size

    if npixels < 10:
        z1 = np.min(image)
        z2 = np.max(image)
    else:
        q000, q375, q500, q625, q1000 = np.percentile(image, [00.0, 37.5, 50.0, 62.5, 100.0])
        zslope = (q625-q375)/(0.25*npixels)
        z1 = q500-(zslope*npixels/2)/factor
        z1 = max(z1, q000)
        z2 = q500+(zslope*npixels/2)/factor
        z2 = min(z2, q1000)
    
    return z1, z2
