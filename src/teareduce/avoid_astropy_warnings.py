#
# Copyright 2022-2024 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

def avoid_astropy_warnings(avoid_warnings):
    """Auxiliary function to help disable astropy warnings

    Parameters
    ----------
    avoid_warnings : bool
        If True, disable the warnings.

    """

    if avoid_warnings:
        import warnings
        warnings.resetwarnings()
        warnings.filterwarnings('ignore', category=UserWarning, append=True)
        from astropy.utils.exceptions import AstropyWarning
        warnings.simplefilter('ignore', AstropyWarning)
