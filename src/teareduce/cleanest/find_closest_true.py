#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Find the closest true (CR) pixel to a given (x, y) position."""

import numpy as np


def find_closest_true(mask, x, y):
    """Find the closest True pixel in a boolean mask to the given (x, y) position.

    Parameters
    ----------
    mask : 2D numpy.ndarray of bool
        Boolean mask where True indicates the presence of a cosmic ray pixel.
    x : int
        X-coordinate (column index) of the reference position (0-based).
    y : int
        Y-coordinate (row index) of the reference position (0-based).

    Returns
    -------
    (closest_x, closest_y) : tuple of int
        Coordinates (0-based) of the closest True pixel in the mask.
        Returns (None, None) if no True pixels are found.
    min_distance : float
        Euclidean distance to the closest True pixel.
    """
    true_indices = np.argwhere(mask)
    if true_indices.size == 0:
        return None, None

    distances = np.sqrt((true_indices[:, 1] - x) ** 2 + (true_indices[:, 0] - y) ** 2)
    min_index = np.argmin(distances)
    closest_y, closest_x = true_indices[min_index]
    min_distance = distances[min_index]

    return (closest_x, closest_y), min_distance
