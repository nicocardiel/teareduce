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
# Note that 'type' is set to the expected data type for each parameter
# using the intrinsic Python types, so that they can be easily cast
# when reading user input.
lacosmic_default_dict = {
    # L.A.Cosmic parameters for run 1
    'run1_gain': {'value': 1.0, 'type': float, 'positive': True},
    'run1_readnoise': {'value': 6.5, 'type': float, 'positive': True},
    'run1_sigclip': {'value': 5.0, 'type': float, 'positive': True},
    'run1_sigfrac': {'value': 0.3, 'type': float, 'positive': True},
    'run1_objlim': {'value': 5.0, 'type': float, 'positive': True},
    'run1_niter': {'value': 4, 'type': int, 'positive': True},
    'run1_verbose': {'value': False, 'type': bool},
    # L.A.Cosmic parameters for run 2
    'run2_gain': {'value': 1.0, 'type': float, 'positive': True},
    'run2_readnoise': {'value': 6.5, 'type': float, 'positive': True},
    'run2_sigclip': {'value': 3.0, 'type': float, 'positive': True},
    'run2_sigfrac': {'value': 0.3, 'type': float, 'positive': True},
    'run2_objlim': {'value': 5.0, 'type': float, 'positive': True},
    'run2_niter': {'value': 4, 'type': int, 'positive': True},
    'run2_verbose': {'value': False, 'type': bool},
    # Dilation of the mask
    'dilation': {'value': 0, 'type': int, 'positive': True},
    # Limits for the image section to process (pixels start at 1)
    'xmin': {'value': 1, 'type': int, 'positive': True},
    'xmax': {'value': None, 'type': int, 'positive': True},
    'ymin': {'value': 1, 'type': int, 'positive': True},
    'ymax': {'value': None, 'type': int, 'positive': True},
    # Number of runs to execute L.A.Cosmic
    'nruns': {'value': 1, 'type': int, 'positive': True}
}

# Default parameters for cleaning methods
VALID_CLEANING_METHODS = [
    'x interp.',
    'y interp.',
    'surface interp.',
    'median',
    'mean',
    'lacosmic',
    'auxdata'
]

# Maximum pixel distance to consider when finding closest CR pixel
MAX_PIXEL_DISTANCE_TO_CR = 15

# Default number of points for interpolation
DEFAULT_NPOINTS_INTERP = 2

# Default degree for interpolation
DEFAULT_DEGREE_INTERP = 1
