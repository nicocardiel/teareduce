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
LACOSMIC_DEFAULT_SIGMA_DETECT = 5.0
LACOSMIC_DEFAULT_SIGMA_CLIP = 3.0
LACOSMIC_DEFAULT_OBJ_LIMIT = 5.0
LACOSMIC_DEFAULT_SMOOTHING = 1.0
LACOSMIC_DEFAULT_GROW = 1
LACOSMIC_DEFAULT_ITERATIONS = 4
LACOSMIC_DEFAULT_SATUR_LEVEL = 65535.0
LACOSMIC_DEFAULT_READNOISE = 10.0
LACOSMIC_DEFAULT_GAIN = 1.0
LACOSMIC_DEFAULT_FILTER = 'laplace'

# Maximum pixel distance to consider when finding closest CR pixel
MAX_PIXEL_DISTANCE_TO_CR = 15
