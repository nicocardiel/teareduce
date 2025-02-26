#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

# ToDo: include documentation
# ToDo: use astropy units?

import numpy as np

from .sliceregion import SliceRegion2D

VALID_PARAMETERS = ["gain", "readout_noise", "bias", "dark", "flatfield", "data_model"]
VALID_IMGTYPES = ["bias", "dark", "object"]


class SimulateCCDExposure:
    def __init__(
            self, naxis1=None, naxis2=None,
            gain=np.nan, readout_noise=np.nan,
            bias=np.nan, dark=np.nan,
            flatfield=np.nan
    ):
        # protections
        if naxis1 is None or naxis2 is None:
            raise RuntimeError("Basic image parameters (naxis1, naxis2) must be provided")

        self.bias = np.full(shape=(naxis2, naxis1), fill_value=bias, dtype=float)
        self.gain = np.full(shape=(naxis2, naxis1), fill_value=gain, dtype=float)
        self.readout_noise = np.full(shape=(naxis2, naxis1), fill_value=readout_noise, dtype=float)
        self.dark = np.full(shape=(naxis2, naxis1), fill_value=dark, dtype=float)
        self.flatfield = np.full(shape=(naxis2, naxis1), fill_value=flatfield, dtype=float)
        self.data_model = np.full(shape=(naxis2, naxis1), fill_value=0, dtype=float)
        self.result = None
        self.naxis1 = naxis1
        self.naxis2 = naxis2

    def set_constant(self, parameter=None, value=np.nan, region=None):
        full_frame = SliceRegion2D(np.s_[1:self.naxis1, 1:self.naxis2], mode='fits')
        # protections
        if parameter.lower() not in VALID_PARAMETERS:
            raise RuntimeError(f"Invalid {parameter=}\n{VALID_PARAMETERS=}")
        if region is None:
            region = full_frame
        else:
            if isinstance(region, SliceRegion2D):
                # check region is within NAXIS1, NAXIS2 rectangle
                if not region.within(full_frame):
                    raise RuntimeError(f"Region {region=} outside of frame {full_frame=}")
            else:
                raise TypeError(f"The parameter {region=} must be an instance of SliceRegion2D")
        if not isinstance(value, (int, float)) or isinstance(value, np.ndarray):
            raise TypeError("The parameter 'value' must be a single number")

        # set parameter
        try:
            getattr(self, parameter)[region.python] = value
        except AttributeError:
            raise RuntimeError(f"The parameter {parameter=} must be an attribute of SimulateCCDExposure")

    def set_array2d(self, parameter=None, array2d=None, region=None):
        full_frame = SliceRegion2D(np.s_[1:self.naxis1, 1:self.naxis2], mode='fits')
        # protections
        if parameter.lower() not in VALID_PARAMETERS:
            raise RuntimeError(f"Invalid {parameter=}\n{VALID_PARAMETERS=}")
        if region is None:
            region = full_frame
        else:
            if isinstance(region, SliceRegion2D):
                # check region is within NAXIS1, NAXIS2 rectangle
                if not region.within(full_frame):
                    raise RuntimeError(f"Region {region=} outside of frame {full_frame=}")
            else:
                raise TypeError(f"The parameter {region=} must be an instance of SliceRegion2D")
        if not isinstance(array2d, np.ndarray):
            raise TypeError("The parameter 'array2d' must be a numpy array")

        naxis2_, naxis1_ = array2d.shape
        if self.naxis1 != naxis1_ or self.naxis2 != naxis2_:
            print(f"{array2d.shape=}")
            raise ValueError(f"The parameter 'array2d' must have shape ({self.naxis1=}, {self.naxis2=})")

        # set parameter
        try:
            getattr(self, parameter)[region.python] = array2d[region.python]
        except AttributeError:
            raise RuntimeError(f"The parameter {parameter=} must be an attribute of SimulateCCDExposure")

    def run(self, seed=None, imgtype=None, method="Poisson"):
        # protections
        if imgtype.lower() not in VALID_IMGTYPES:
            raise ValueError(f'Unexpected {imgtype=}')
        if method.lower() not in ["poisson", "gaussian"]:
            raise ValueError(f'Unexpected {method=}')
        user_defined_attributes = [
            attr for attr in dir(self) if not attr.startswith("__") and not callable(getattr(self, attr))
        ]
        user_defined_attributes.remove('result')
        for attr in user_defined_attributes:
            if np.isnan(getattr(self, attr)).any():
                raise ValueError(f"The parameter {attr=} contains NaN")

        rng = np.random.default_rng(seed)

        # BIAS and Readout Noise
        self.result = rng.normal(
            loc=self.bias,
            scale=self.readout_noise
        )
        if imgtype == "bias":
            return self.result

        # DARK
        self.result += self.dark
        if imgtype == "dark":
            return self.result

        # OBJECT
        if method.lower() == "poisson":
            # transform data_model from ADU to electrons,
            # generate Poisson distribution
            # and transform back from electrons to ADU
            self.result += self.flatfield * rng.poisson(self.data_model * self.gain) / self.gain
        elif method.lower() == "gaussian":
            self.result += self.flatfield * rng.normal(
                loc=self.data_model,
                scale=np.sqrt(self.data_model/self.gain)
            )
        else:
            raise RuntimeError(f"Unknown method: {method}")

        return self.result
