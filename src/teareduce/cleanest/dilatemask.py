#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Dilate cosmic ray mask"""

from scipy import ndimage


def dilatemask(mask, iterations, rank=None, connectivity=1):
    """Dilate mask by a given number of points.

    Parameters
    ----------
    mask : numpy.ndarray of bool
        A boolean mask array indicating cosmic ray affected pixels.
    iterations : int
        The number of dilation iterations to perform. Each iteration
        expands the mask by one pixel in the specified connectivity.
    rank : int, optional
        The rank of the array. If None, it is inferred from crmask.
        See scipy.ndimage.generate_binary_structure for details.
    connectivity : int, optional
        The connectivity for the structuring element. Default is 1.
        See scipy.ndimage.generate_binary_structure for details.

    Returns
    -------
    dilated_mask : numpy.ndarray of bool
        The dilated mask.
    """
    if rank is None:
        rank = mask.ndim
    structure = ndimage.generate_binary_structure(rank, connectivity)
    dilated_mask = ndimage.binary_dilation(mask, structure=structure, iterations=iterations)
    return dilated_mask
