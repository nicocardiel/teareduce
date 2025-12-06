#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Execute LACosmic algorithm on a padded image."""

from ccdproc import cosmicray_lacosmic
import numpy as np


def lacosmicpad(pad_width, **kwargs):
    """Execute LACosmic algorithm on a padded array.

    This function pads the input image array before applying the LACosmic
    cosmic ray cleaning algorithm. After processing, the padding is removed
    to return an array of the original size.

    The padding helps to mitigate edge effects that can occur during the
    cosmic ray detection and cleaning process.

    Apart from the `pad_width` parameter, all other keyword arguments
    are passed directly to the `cosmicray_lacosmic` function from the
    `ccdproc` package.

    Parameters
    ----------
    pad_width : int
        Width of the padding to be applied to the image before executing
        the LACosmic algorithm.
    **kwargs : dict
        Keyword arguments to be passed to the `cosmicray_lacosmic` function.

    Returns
    -------
    clean_array : 2D numpy.ndarray
        The cleaned image array after applying the LACosmic algorithm with padding.
    mask_array : 2D numpy.ndarray of bool
        The mask array indicating detected cosmic rays.
    """
    if "ccd" not in kwargs:
        raise ValueError("The 'ccd' keyword argument must be provided.")
    array = kwargs.pop("ccd")
    if not isinstance(array, np.ndarray):
        raise TypeError("The 'ccd' keyword argument must be a numpy ndarray.")
    # Pad the array
    padded_array = np.pad(array, pad_width, mode="reflect")
    # Apply LACosmic algorithm to the padded array
    cleaned_padded_array, mask_padded_array = cosmicray_lacosmic(ccd=padded_array, **kwargs)
    # Remove padding
    cleaned_array = cleaned_padded_array[
        pad_width:-pad_width,
        pad_width:-pad_width
    ]
    mask_array = mask_padded_array[
        pad_width:-pad_width,
        pad_width:-pad_width
    ]
    return cleaned_array, mask_array
