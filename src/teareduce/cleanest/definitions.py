#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Definitions for the cleanest module."""

# Default parameters for L.A.Cosmic algorithm
lacosmic_default_dict = {
    'gain': {'value': 1.0, 'type': float},
    'readnoise': {'value': 6.5, 'type': float},
    'sigclip': {'value': 4.5, 'type': float},
    'sigfrac': {'value': 0.3, 'type': float},
    'objlim': {'value': 5.0, 'type': float},
    'niter': {'value': 4, 'type': int},
    'verbose': {'value': False, 'type': bool}
}

# Maximum pixel distance to consider when finding closest CR pixel
MAX_PIXEL_DISTANCE_TO_CR = 15
