#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Definitions for the cleanest module."""

VALID_LACOSMIC_CLEANTYPE_VALUES = ["median", "medmask", "meanmask", "idw"]
VALID_LACOSMIC_FSMODE_VALUES = ["median", "convolve"]
VALID_LACOSMIC_PSFMODEL_VALUES = ["gauss", "moffat", "gaussx", "gaussy", "gaussxy"]

# Default parameters for L.A.Cosmic algorithm
# Note that 'type' is set to the expected data type for each parameter
# using the intrinsic Python types, so that they can be easily cast
# when reading user input.
lacosmic_default_dict = {
    # L.A.Cosmic parameters for run 1
    "run1_sigclip": {"value": 5.0, "type": float, "positive": True},
    "run1_sigfrac": {"value": 0.3, "type": float, "positive": True},
    "run1_objlim": {"value": 5.0, "type": float, "positive": True},
    "run1_gain": {"value": 1.0, "type": float, "positive": True},
    "run1_readnoise": {"value": 6.5, "type": float, "positive": True},
    "run1_satlevel": {"value": 65535, "type": float, "positive": True},
    "run1_niter": {"value": 4, "type": int, "positive": True, "intmode": "any"},
    "run1_sepmed": {"value": True, "type": bool},
    "run1_cleantype": {"value": "meanmask", "type": str, "valid_values": VALID_LACOSMIC_CLEANTYPE_VALUES},
    "run1_fsmode": {"value": "median", "type": str, "valid_values": VALID_LACOSMIC_FSMODE_VALUES},
    "run1_psfmodel": {
        "value": "gaussxy",
        "type": str,
        "valid_values": VALID_LACOSMIC_PSFMODEL_VALUES,
    },
    "run1_psffwhm_x": {"value": 2.5, "type": float, "positive": True},
    "run1_psffwhm_y": {"value": 2.5, "type": float, "positive": True},
    "run1_psfsize": {"value": 7, "type": int, "positive": True, "intmode": "odd"},
    "run1_psfbeta": {"value": 4.765, "type": float, "positive": True},
    "run1_verbose": {"value": True, "type": bool},
    # L.A.Cosmic parameters for run 2
    "run2_sigclip": {"value": 3.0, "type": float, "positive": True},
    "run2_sigfrac": {"value": 0.3, "type": float, "positive": True},
    "run2_objlim": {"value": 5.0, "type": float, "positive": True},
    "run2_gain": {"value": 1.0, "type": float, "positive": True},
    "run2_readnoise": {"value": 6.5, "type": float, "positive": True},
    "run2_satlevel": {"value": 65535, "type": float, "positive": True},
    "run2_niter": {"value": 4, "type": int, "positive": True, "intmode": "any"},
    "run2_sepmed": {"value": True, "type": bool},
    "run2_cleantype": {"value": "meanmask", "type": str, "valid_values": VALID_LACOSMIC_CLEANTYPE_VALUES},
    "run2_fsmode": {"value": "median", "type": str, "valid_values": VALID_LACOSMIC_FSMODE_VALUES},
    "run2_psfmodel": {
        "value": "gaussxy",
        "type": str,
        "valid_values": VALID_LACOSMIC_PSFMODEL_VALUES,
    },
    "run2_psffwhm_x": {"value": 2.5, "type": float, "positive": True},
    "run2_psffwhm_y": {"value": 2.5, "type": float, "positive": True},
    "run2_psfsize": {"value": 7, "type": int, "positive": True, "intmode": "odd"},
    "run2_psfbeta": {"value": 4.765, "type": float, "positive": True},
    "run2_verbose": {"value": True, "type": bool},
    # Dilation of the mask
    "dilation": {"value": 0, "type": int, "positive": True},
    "borderpadd": {"value": 10, "type": int, "positive": True},
    # Limits for the image section to process (pixels start at 1)
    "xmin": {"value": 1, "type": int, "positive": True},
    "xmax": {"value": None, "type": int, "positive": True},
    "ymin": {"value": 1, "type": int, "positive": True},
    "ymax": {"value": None, "type": int, "positive": True},
    # Number of runs to execute L.A.Cosmic
    "nruns": {"value": 1, "type": int, "positive": True, "intmode": "any"},
}

# Valid cleaning methods (as shown to the user and their internal codes)
VALID_CLEANING_METHODS = {
    "x interp.": "x",
    "y interp.": "y",
    "surface interp.": "a-plane",
    "median": "a-median",
    "mean": "a-mean",
    "lacosmic": "lacosmic",
    "maskfill": "maskfill",
    "auxdata": "auxdata",
}

# Maximum pixel distance to consider when finding closest CR pixel
MAX_PIXEL_DISTANCE_TO_CR = 15

# Default number of points for interpolation
DEFAULT_NPOINTS_INTERP = 2

# Default degree for interpolation
DEFAULT_DEGREE_INTERP = 1

# Default maskfill parameters
DEFAULT_MASKFILL_SIZE = 3
DEFAULT_MASKFILL_OPERATOR = "median"
MASKFILL_OPERATOR_VALUES = ["median", "mean"]
DEFAULT_MASKFILL_SMOOTH = True
DEFAULT_MASKFILL_VERBOSE = False

# Default Tk window size
DEFAULT_TK_WINDOW_SIZE_X = 800
DEFAULT_TK_WINDOW_SIZE_Y = 700

# Default font settings
DEFAULT_FONT_FAMILY = "Helvetica"
DEFAULT_FONT_SIZE = 14
