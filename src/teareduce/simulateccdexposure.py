#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

# ToDo: use astropy units?
# ToDo: allow to define SimulateCCDExposure from numpy arrays

import matplotlib.pyplot as plt
import numpy as np

from .sliceregion import SliceRegion2D
from .imshow import imshow

VALID_PARAMETERS = ["bias", "gain", "readout_noise", "dark", "flatfield", "data_model"]
VALID_IMAGE_TYPES = ["bias", "dark", "object"]
VALID_METHODS = ["poisson", "gaussian"]


class SimulatedCCDResult:
    """Result of executing SimulateCCDExposure.run().

    Auxiliary class to store the result and all the relevant
    parameters employed when generating the simulated CCD image.

    Attributes
    ----------
    data : numpy.ndarray or None
        Data array with the result of the simulated CCD exposure.
    imgtype : str
        Type of image to be generated. It must be one of
        VALID_IMAGE_TYPES.
    method : str
        Method used to generate the simulated CCD image.
        It must be one of VALID_METHODS.
    parameters : dict or None
        CCD parameters employed during the simulation procedure.

    Methods
    -------
    imshow(**kwargs)
        Display simulated CCD image using tea.imshow().
    """
    def __init__(self, data, imgtype, method, parameters):
        """
        Initialize the class attributes.

        Parameters
        ----------
        data : numpy.ndarray or None
            Data array with the result of the simulated CCD exposure.
        imgtype : str
            Type of image to be generated. It must be one of
            VALID_IMAGE_TYPES.
        method : str
            Method used to generate the simulated CCD image.
            It must be one of VALID_METHODS.
        parameters : dict
            CCD parameters employed during the simulation procedure.
        """
        self.data = data
        self.imgtype = imgtype
        self.method = method
        self.parameters = parameters

    def __repr__(self):
        output = f'{self.__class__.__name__}(\n'
        output += f'    data={self.data!r},\n'
        output += f'    imgtype={self.imgtype!r},\n'
        output += f'    method={self.method!r},\n'
        output += f'    parameters={self.parameters!r}\n'
        output += ')'
        return output

    def imshow(self, **kwargs):
        """Plot simulated CCD image

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to be passed to
            teareduce.imshow().
        """
        fig, ax = plt.subplots()
        imshow(fig, ax, self.data, **kwargs)
        plt.tight_layout()
        plt.show()


class SimulateCCDExposure:
    """Simulated image generator from first principles.

    CCD exposures are simulated making use of basic CCD parameters,
    such as gain, readout noise, bias, dark and flat field.
    A data model can also be employed to simulate more realising
    CCD exposures.

    Attributes
    ----------
    bias : numpy.ndarray
        Detector bias level.
    gain : numpy.ndarray
        Detector gain (electrons/ADU).
    readout_noise : numpy.ndarray
        Readout noise (ADU).
    dark : numpy.ndarray
        Total dark current (ADU). This number should not be
        the dark current rate (ADU/s). The provided number must
        correspond to the total dark current since the exposure
        time is not defined.
    flatfield : numpy.ndarray
        Pixel to pixel sensitivity.
    data_model : numpy.ndarray
        Model of the source to be simulated (ADU).
    naxis1 : int
        NAXIS1 value.
    naxis2 : int
        NAXIS2 value.

    Methods
    -------
    set_constant(parameter, value, region)
        Set the value of a particular CCD parameter to a constant
        value. The parameter can be any of VALID_PARAMETERS. The
        constant value can be employed for all pixels or in a specific
        region.
    set_array2d(parameter, value, region)
        Set the value of a particular CCD parameter to a 2D array.
        The parameter can be any of VALID_PARAMETERS. The 2D array
        may correspond to the whole simulated array or to a specific
        region.
    run(imgtype, seed, method)
        Execute the generation of the simulated CCD exposure of
        type 'imgtype', where 'imgtype' is one of VALID_IMAGE_TYPES.
        The signal can be generated using either method: Poisson
        or Gaussian. It is possible to set the seed in order to
        initialize the random number generator.

    """
    def __init__(self,
                 naxis1=None,
                 naxis2=None,
                 bias=np.nan,
                 gain=np.nan,
                 readout_noise=np.nan,
                 dark=np.nan,
                 flatfield=np.nan,
                 data_model=np.nan):
        """Initialize the class attributes.

        The simulated array dimensions are mandatory. This function
        initializes the 2D parameters assuming constant values.
        If no parameters are provided, the default values are set to NaN.
        The constant value of each parameter for the entire image can be
        subsequently modified using methods that allow these values to be
        changed in specific regions of the CCD.

        Parameters
        ----------
        naxis1 : int
            NAXIS1 value.
        naxis2 : int
            NAXIS2 value.
        bias : float
            Detector bias level.
        gain : float
            Detector gain (electrons/ADU).
        readout_noise : float
            Readout noise (ADU).
        dark : float
            Total dark current (ADU). This number should not be the
            dark current rate (ADU/s). The provided number must
            be the total dark current since the exposure time is not
            defined.
        flatfield : float
            Pixel to pixel sensitivity.
        data_model : float
            Model of the source to be simulated.

        """
        # protections
        if naxis1 is None or naxis2 is None:
            raise RuntimeError("Basic image parameters (naxis1, naxis2) must be provided")
        if not isinstance(naxis1, int):
            raise ValueError(f"{naxis1=} must be an integer")
        if not isinstance(naxis2, int):
            raise ValueError(f"{naxis2=} must be an integer")
        if naxis1 < 0 or naxis2 < 0:
            raise ValueError(f"Both {naxis1=} and {naxis2=} must be positive")

        # check the parameters are either np.nan or a number (integer or float)
        parameters = {
            "bias": bias,
            "gain": gain,
            "readout_noise": readout_noise,
            "dark": dark,
            "flatfield": flatfield,
            "data_model": data_model
        }
        for parameter, value in parameters.items():
            if not np.isnan(value) and not isinstance(value, (int, float)):
                raise ValueError(f"{parameter}={value} must be numeric or np.nan")

        # define the CCD parameters using a constant value for the full aray
        self.bias = np.full(shape=(naxis2, naxis1), fill_value=bias, dtype=float)
        self.gain = np.full(shape=(naxis2, naxis1), fill_value=gain, dtype=float)
        self.readout_noise = np.full(shape=(naxis2, naxis1), fill_value=readout_noise, dtype=float)
        self.dark = np.full(shape=(naxis2, naxis1), fill_value=dark, dtype=float)
        self.flatfield = np.full(shape=(naxis2, naxis1), fill_value=flatfield, dtype=float)
        self.data_model = np.full(shape=(naxis2, naxis1), fill_value=data_model, dtype=float)
        self.naxis1 = naxis1
        self.naxis2 = naxis2

    def __repr__(self):
        output = f'{self.__class__.__name__}(\n'
        output += f'    naxis1={self.naxis1},\n'
        output += f'    naxis2={self.naxis2},\n'
        for parameter in VALID_PARAMETERS:
            output += f'    {parameter}={getattr(self, parameter)!r},\n'
        output += ')'
        return output

    def _precheck_set_function(self, parameter, region):
        """Auxiliary function to check function inputs.

        This function checks whether the parameters provided to
        the functions in charge of defining pixels values are correct.

        """
        full_frame = SliceRegion2D(f"[1:{self.naxis1}, 1:{self.naxis2}]", mode='fits')
        # protections
        if parameter not in VALID_PARAMETERS:
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
        return region

    def set_constant(self, parameter, value, region=None):
        """
        Set the value of a particular parameter to a constant value.

        The parameter can be any of VALID_PARAMETERS. The constant
        value can be employed for all pixels or in a specific region.

        Parameters
        ----------
        parameter : str
            CCD parameter to set. It must be any of VALID_PARAMETERS.
        value : float
            Constant value for the parameter.
        region : SliceRegion2D
            Region in which to define de parameter. When it is None, it
            indicates that 'value' should be set for all pixels.
        """
        # protections
        parameter = parameter.lower()
        region = self._precheck_set_function(parameter, region)
        if not isinstance(value, (int, float)) or isinstance(value, np.ndarray):
            raise TypeError("The parameter 'value' must be a single number")

        # set parameter
        try:
            getattr(self, parameter)[region.python] = value
        except AttributeError:
            raise RuntimeError(f"The parameter {parameter=} must be an attribute of SimulateCCDExposure")

    def set_array2d(self, parameter, array2d, region=None):
        """
        Set the value of a particular parameter to a 2D array.

        The parameter can be any of VALID_PARAMETERS. The 2D array
        may correspond to the whole simulated array or to a specific
        region.

        Parameters
        ----------
        parameter : str
            CCD parameter to set. It must be any of VALID_PARAMETERS.
        array2d : numpy.ndarray
            Array of values to be used to define 'parameter'.
        region : SliceRegion2D
            Region in which to define de parameter. When it is None, it
            indicates that 'array2d' has the same shape as the simulated
            image.
        """
        # protections
        parameter = parameter.lower()
        region = self._precheck_set_function(parameter, region)
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

    def run(self, imgtype, method="Poisson", seed=None, return_all=False):
        """
        Execute the generation of the simulated CCD exposure.

        This function generates an image of type 'imgtype', which must
        be one of VALID_IMAGE_TYPES. The signal can be generated using
        either method: Poisson or Gaussian. It is possible to set the
        seed in order to initialize the random number generator.

        Parameters
        ----------
        imgtype : str
            Type of image to be generated. It must be one of
            VALID_IMAGE_TYPES.
        method : str
            Method to generate the simulated data. It must be one of
            VALID_METHODS.
        seed : int, optional
            Seed for the random number generator. The default is None.
        return_all : bool, optional
            If True, return all the parameters in VALID_PARAMETERS
            in the 'parameters' attribute of the returned result.
            The default is False.

        Returns
        -------
        result : SimulatedCCDResult
            Instance of SimulatedCCDResult to store the simulated image
            and the associated parameters employed to define how to
            generate the simulated CCD exposure.
        """
        # protections
        imgtype = imgtype.lower()
        method = method.lower()
        if imgtype not in VALID_IMAGE_TYPES:
            raise ValueError(f'Unexpected {imgtype=}.\nValid image types: {VALID_IMAGE_TYPES}')
        if method not in VALID_METHODS:
            raise ValueError(f'Unexpected {method=}.\nValid methods: {VALID_METHODS}')
        user_defined_attributes = [
            attr for attr in dir(self) if not attr.startswith("__") and not callable(getattr(self, attr))
        ]
        for attr in user_defined_attributes:
            if np.isnan(getattr(self, attr)).any():
                raise ValueError(f"The parameter {attr=} contains NaN")

        rng = np.random.default_rng(seed)

        if return_all:
            parameters = dict()
            for attr in VALID_PARAMETERS:
                parameters[attr] = getattr(self, attr)
        else:
            parameters = None

        result = SimulatedCCDResult(
            data=None,
            imgtype=imgtype,
            method=method,
            parameters=parameters
        )

        # BIAS and Readout Noise
        image2d = rng.normal(
            loc=self.bias,
            scale=self.readout_noise
        )
        if imgtype == "bias":
            result.data = image2d
            return result

        # DARK
        image2d += self.dark
        if imgtype == "dark":
            result.data = image2d
            return result

        # OBJECT
        if method.lower() == "poisson":
            # transform data_model from ADU to electrons,
            # generate Poisson distribution
            # and transform back from electrons to ADU
            image2d += self.flatfield * rng.poisson(self.data_model * self.gain) / self.gain
        elif method.lower() == "gaussian":
            image2d += self.flatfield * rng.normal(
                loc=self.data_model,
                scale=np.sqrt(self.data_model/self.gain)
            )
        else:
            raise RuntimeError(f"Unknown method: {method}")

        result.data = image2d
        return result
