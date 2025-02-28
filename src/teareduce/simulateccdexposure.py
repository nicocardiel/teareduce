#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#
from astropy.units import Unit
from astropy.units import Quantity
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
    unit : astropy.units.Unit
        Units of the simulated CCD exposure.
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

    def __init__(self, data, unit, imgtype, method, parameters):
        """
        Initialize the class attributes.

        Parameters
        ----------
        data : numpy.ndarray or None
            Data array with the result of the simulated CCD exposure.
        unit : astropy.units.Unit
            Units of the simulated CCD exposure.
        imgtype : str
            Type of image to be generated. It must be one of
            VALID_IMAGE_TYPES.
        method : str
            Method used to generate the simulated CCD image.
            It must be one of VALID_METHODS.
        parameters : dict or None
            CCD parameters employed during the simulation procedure.
        """
        self.data = data
        self.unit = unit
        self.imgtype = imgtype
        self.method = method
        self.parameters = parameters

    def __repr__(self):
        output = f'{self.__class__.__name__}(\n'
        output += f'    data={self.data!r},\n'
        output += f'    unit={self.unit!r},\n'
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
    naxis1 : int
        NAXIS1 value.
    naxis2 : int
        NAXIS2 value.
    bias : Quantity
        Numpy array with the detector bias level (ADU).
    gain : Quantity
        Numpy array with the detector gain (electrons/ADU).
    readout_noise : Quantity
        Numpy array with the readout noise (ADU).
    dark : Quantity
        Numpy array with the total dark current (ADU) for each pixel.
        These numbers should not be the dark current rate (ADU/s).
        The provided numbers must correspond to the total dark current
        since the exposure time is not defined.
    flatfield : numpy.ndarray
        Numpy array with the pixel to pixel sensitivity (without units).
    data_model : Quantity
        Numpy array with the model of the source to be simulated (ADU).

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
                 bias=np.nan * Unit('adu'),
                 gain=np.nan * Unit('electron') / Unit('adu'),
                 readout_noise=np.nan * Unit('adu'),
                 dark=np.nan * Unit('adu'),
                 flatfield=np.nan,
                 data_model=np.nan * Unit('adu')):
        """Initialize the class attributes.

        The simulated array dimensions are mandatory. If any additional
        parameter is not provided, all the pixel values are set to NaN.
        The values of each parameter can be subsequently modified using
        methods that allow these values to be changed in specific regions
        of the CCD.

        The input parameters must be a Quantity whose value is either
        a single number, which is expanded to fill the numpy.array,
        or a numpy.array with the expected shape (NAXIS2, NAXIS1).

        Parameters
        ----------
        naxis1 : int
            NAXIS1 value.
        naxis2 : int
            NAXIS2 value.
        bias : Quantity
            Detector bias level (ADU).
        gain : Quantity
            Detector gain (electrons/ADU).
        readout_noise : Quantity
            Readout noise (ADU).
        dark : Quantity
            Total dark current (ADU). This number should not be the
            dark current rate (ADU/s). The provided number must
            be the total dark current since the exposure time is not
            defined.
        flatfield : float or numpy.ndarray
            Pixel to pixel sensitivity (without units).
        data_model : Quantity
            Model of the source to be simulated (ADU).
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

        # image shape
        self.naxis1 = naxis1
        self.naxis2 = naxis2
        self.bias = None
        self.gain = None
        self.readout_noise = None
        self.dark = None
        self.flatfield = None
        self.data_model = None

        # check that each input parameter is a Quantity with the expected units,
        # and that the quantity.value is either a single number (integer or float)
        # or a numpy.array with the expected shape
        parameters = {
            "bias": bias,
            "gain": gain,
            "readout_noise": readout_noise,
            "dark": dark,
            "flatfield": flatfield,
            "data_model": data_model
        }
        for parameter, quantity in parameters.items():
            if parameter == "gain":
                expected_units = Unit('electron') / Unit('adu')
            elif parameter == "flatfield":
                expected_units = None
            else:
                expected_units = Unit('adu')
            if expected_units is not None:
                if not isinstance(quantity, Quantity):
                    raise TypeError(f"{parameter} must be a Quantity: {expected_units=}")
                if quantity.unit != expected_units:
                    raise ValueError(f"{quantity.unit=} != {expected_units=}")
                value = quantity.value
            else:
                if isinstance(quantity, Quantity):
                    raise ValueError(f"'{parameter}' should not be a Quantity")
                value = quantity
            if isinstance(value, (int, float)):
                # constant value for the full array
                if parameter == "flatfield":
                    setattr(self, parameter, np.full(shape=(naxis2, naxis1), fill_value=value))
                else:
                    setattr(self, parameter, np.full(shape=(naxis2, naxis1), fill_value=value) * expected_units)
            elif isinstance(value, np.ndarray):
                naxis2_, naxis1_ = value.shape
                if naxis1_ == naxis1 and naxis2_ == naxis2:
                    # array of the expected shape
                    setattr(self, parameter, quantity)
                else:
                    msg = (f"Parameter {parameter}: NAXIS1={naxis1_}, NAXIS2={naxis2_} "
                           f"are not compatible with expected values NAXIS1={naxis1}, NAXIS2={naxis2}")
                    raise ValueError(msg)
            else:
                raise ValueError(f"Unexpected {parameter=} with {type(value)=}")

    def __repr__(self):
        output = f'{self.__class__.__name__}(\n'
        output += f'    naxis1={self.naxis1},\n'
        output += f'    naxis2={self.naxis2},\n'
        for parameter in VALID_PARAMETERS:
            output += f'    {parameter}={getattr(self, parameter)!r},\n'
        output += ')'
        return output

    def _precheck_set_function(self, parameter, quantity, region):
        """Auxiliary function to check function inputs.

        This function checks whether the parameters provided to
        the functions in charge of defining pixels values are correct.

        Parameters
        ----------
        parameter : str
            CCD parameter to set. It must be any of VALID_PARAMETERS.
        quantity : Quantity
            Float or numpy.array with units (except for flatfield).
        region : SliceRegion2D or None
            Region in which to define de parameter. When it is None, it
            indicates that 'value' should be set for all pixels.

        Returns
        -------
        region : SliceRegion2D
            Updated region in which to define de parameter. When the input
            value is None, the returned region will be the full frame.
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

        if parameter == "gain":
            expected_units = Unit('electron') / Unit('adu')
        elif parameter == "flatfield":
            expected_units = None
        else:
            expected_units = Unit('adu')
        if expected_units is not None:
            if not isinstance(quantity, Quantity):
                raise TypeError(f"{parameter} must be a Quantity: {expected_units=}")
            if quantity.unit != expected_units:
                raise ValueError(f"{quantity.unit=} != {expected_units=}")

        return region

    def set_constant(self, parameter, constant, region=None):
        """
        Set the value of a particular parameter to a constant value.

        The parameter can be any of VALID_PARAMETERS. The constant
        value can be employed for all pixels or in a specific region.

        Parameters
        ----------
        parameter : str
            CCD parameter to set. It must be any of VALID_PARAMETERS.
        constant : Quantity or float
            Constant value for the parameter.
        region : SliceRegion2D or None
            Region in which to define de parameter. When it is None, it
            indicates that 'value' should be set for all pixels.
        """
        # protections
        parameter = parameter.lower()
        region = self._precheck_set_function(parameter, constant, region)
        if parameter == "flatfield":
            value = constant
        else:
            value = constant.value

        if not isinstance(value, (int, float)) or isinstance(value, np.ndarray):
            raise TypeError("The parameter 'quantity' must be a single number")

        # set parameter
        try:
            getattr(self, parameter)[region.python] = constant
        except AttributeError:
            raise RuntimeError(f"Unexpected parameter '{parameter}' for SimulateCCDExposure")

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
        array2d : Quantity
            Array of values to be used to define 'parameter'.
        region : SliceRegion2D
            Region in which to define de parameter. When it is None, it
            indicates that 'quantity' has the same shape as the simulated
            image.
        """
        # protections
        parameter = parameter.lower()
        region = self._precheck_set_function(parameter, array2d, region)
        if parameter == "flatfield":
            value = array2d
        else:
            value = array2d.value

        if not isinstance(value, np.ndarray):
            raise TypeError("The parameter 'quantity' must be a numpy array of quantities")
        naxis2_, naxis1_ = value.shape
        if self.naxis1 != naxis1_ or self.naxis2 != naxis2_:
            print(f"{value.shape=}")
            raise ValueError(f"The parameter 'quantity' must have shape ({self.naxis1=}, {self.naxis2=})")

        # set parameter
        try:
            getattr(self, parameter)[region.python] = array2d[region.python]
        except AttributeError:
            raise RuntimeError(f"Unexpected parameter '{parameter}' for SimulateCCDExposure")

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

        rng = np.random.default_rng(seed)

        if return_all:
            parameters = dict()
            for attr in VALID_PARAMETERS:
                parameters[attr] = getattr(self, attr)
        else:
            parameters = None

        # initialize result instance
        result = SimulatedCCDResult(
            data=None,
            unit=Unit('adu'),
            imgtype=imgtype,
            method=method,
            parameters=parameters
        )

        # BIAS and Readout Noise
        if np.isnan(self.bias.value).any():
            raise ValueError(f"The parameter 'bias' contains NaN")
        if np.isnan(self.readout_noise.value).any():
            raise ValueError(f"The parameter 'readout_noise' contains NaN")
        image2d = rng.normal(
            loc=self.bias.value,
            scale=self.readout_noise.value
        )
        if imgtype == "bias":
            result.data = image2d
            return result

        # DARK
        if np.isnan(self.dark.value).any():
            raise ValueError(f"The parameter 'dark' contains NaN")
        image2d += self.dark.value
        if imgtype == "dark":
            result.data = image2d
            return result

        # OBJECT
        if np.isnan(self.flatfield).any():
            raise ValueError(f"The parameter 'flatfield' contains NaN")
        if np.isnan(self.data_model.value).any():
            raise ValueError(f"The parameter 'data_model' contains NaN")
        if np.isnan(self.gain.value).any():
            raise ValueError(f"The parameter 'gain' contains NaN")
        if method.lower() == "poisson":
            # transform data_model from ADU to electrons,
            # generate Poisson distribution
            # and transform back from electrons to ADU
            image2d += self.flatfield * rng.poisson(
                self.data_model.value * self.gain.value
            ) / self.gain.value
        elif method.lower() == "gaussian":
            image2d += self.flatfield * rng.normal(
                loc=self.data_model.value,
                scale=np.sqrt(self.data_model.value/self.gain.value)
            )
        else:
            raise RuntimeError(f"Unknown method: {method}")

        result.data = image2d
        return result
