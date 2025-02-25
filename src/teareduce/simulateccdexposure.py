#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import numpy as np


class SimulateCCDExposure:
    def __init__(self, naxis1=None, naxis2=None, gain=1, rnoise=0, bias=0):
        self.image = np.full((naxis2, naxis1), np.nan)
        self.naxis1 = naxis1
        self.naxis2 = naxis2
        self.gain = gain
        self.rnoise = rnoise
        self.bias = bias

    def run(self, seed=None, imgtype=None):
        rng = np.random.default_rng(seed)
        if imgtype == 'bias':
            self.image = np.random.normal(
                loc=self.bias,
                scale=self.rnoise,
                size=(self.naxis2,self.naxis1)
            )
        else:
            raise ValueError(f'Unexpected {imgtype=}')

        return self.image


