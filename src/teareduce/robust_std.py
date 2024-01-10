#
# Copyright 2022-2024 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import numpy as np


def robust_std(x, debug=False):
    """Compute a robust estimator of the standard deviation

    See Eq. 3.36 (page 84) in Statistics, Data Mining, and Machine
    in Astronomy, by Ivezic, Connolly, VanderPlas & Gray

    Parameters
    ----------
    x : 1d numpy array, float
        Array of input values which standard deviation is requested.
    debug : bool
        If True display computed values

    Returns
    -------
    sigmag : float
        Robust estimator of the standar deviation
    """

    x = np.asarray(x)

    # compute percentiles and robust estimator
    q25, q75 = np.percentile(x, [25, 75])
    sigmag = 0.7413 * (q75 - q25)

    if debug:
        print('debug|robust_std -> q25......................:', q25)
        print('debug|robust_std -> q75......................:', q75)
        print('debug|robust_std -> Robust standard deviation:', sigmag)

    return sigmag
