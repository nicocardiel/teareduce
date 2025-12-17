#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Execute LACosmic algorithm on a padded image."""

try:
    from ccdproc import cosmicray_lacosmic
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "The 'teareduce.cleanest' module requires the 'ccdproc' and 'maskfill' packages. "
        "Please install teareduce with the 'cleanest' extra dependencies: "
        "`pip install teareduce[cleanest]`."
    ) from e
import numpy as np
from rich import print

from .definitions import VALID_LACOSMIC_PSFMODEL_VALUES
from .gausskernel2d_elliptical import gausskernel2d_elliptical


def lacosmicpad(pad_width, show_arguments=False, **kwargs):
    """Execute LACosmic algorithm on a padded array.

    This function pads the input image array before applying the LACosmic
    cosmic ray cleaning algorithm. After processing, the padding is removed
    to return an array of the original size. If `inbkg` or `invar` arrays
    are provided, they are also padded accordingly.

    The padding helps to mitigate edge effects that can occur during the
    cosmic ray detection and cleaning process.

    Apart from the `pad_width` parameter, all other keyword arguments
    are passed directly to the `cosmicray_lacosmic` function from the
    `ccdproc` package.

    Parameters
    ----------
    pad_width : int
        Width of the padding to be applied to the image before executing
        the LACosmic algorithm.
    show_arguments : bool
        If True, display LACosmic arguments being employed.
    **kwargs : dict
        Keyword arguments to be passed to the `cosmicray_lacosmic` function.

    Returns
    -------
    clean_array : 2D numpy.ndarray
        The cleaned image array after applying the LACosmic algorithm with padding.
    mask_array : 2D numpy.ndarray of bool
        The mask array indicating detected cosmic rays.
    """
    # Check for required 'ccd' argument
    if "ccd" not in kwargs:
        raise ValueError("The 'ccd' keyword argument must be provided.")
    array = kwargs.pop("ccd")
    if not isinstance(array, np.ndarray):
        raise TypeError("The 'ccd' keyword argument must be a numpy ndarray.")
    # Pad the array
    padded_array = np.pad(array, pad_width, mode="reflect")

    # Pad inbkg if provided
    if "inbkg" in kwargs:
        if kwargs["inbkg"] is not None:
            inbkg = kwargs["inbkg"]
            if not isinstance(inbkg, np.ndarray):
                raise TypeError("The 'inbkg' keyword argument must be a numpy ndarray.")
            kwargs["inbkg"] = np.pad(inbkg, pad_width, mode="reflect")
    else:
        kwargs["inbkg"] = None

    # Pad invar if provided
    if "invar" in kwargs:
        if kwargs["invar"] is not None:
            invar = kwargs["invar"]
            if not isinstance(invar, np.ndarray):
                raise TypeError("The 'invar' keyword argument must be a numpy ndarray.")
            kwargs["invar"] = np.pad(invar, pad_width, mode="reflect")
    else:
        kwargs["invar"] = None

    # check for fsmode
    if "fsmode" not in kwargs:
        raise ValueError("The 'fsmode' keyword argument must be provided.")
    else:
        fsmode = kwargs["fsmode"]
        if fsmode == "convolve":
            if "psfmodel" not in kwargs:
                raise ValueError("The 'psfmodel' keyword argument must be provided when fsmode is 'convolve'.")
            psfmodel = kwargs["psfmodel"]
            if psfmodel not in VALID_LACOSMIC_PSFMODEL_VALUES:
                raise ValueError(f"The 'psfmodel' keyword argument must be one of {VALID_LACOSMIC_PSFMODEL_VALUES}.")
            if "psffwhm" in kwargs:
                raise ValueError(
                    "When 'fsmode' is 'convolve', 'psffwhm' should not be provided; use 'psffwhm_x' and 'psffwhm_y' instead."
                )
            if "psffwhm_x" not in kwargs or "psffwhm_y" not in kwargs:
                raise ValueError("When 'fsmode' is 'convolve', both 'psffwhm_x' and 'psffwhm_y' must be provided.")
            fwhm_x = kwargs["psffwhm_x"]
            fwhm_y = kwargs["psffwhm_y"]
            if "psfsize" not in kwargs:
                raise ValueError("When 'fsmode' is 'convolve', 'psfsize' must be provided.")
            psfsize = kwargs["psfsize"]
            if kwargs["psfmodel"] == "gaussxy":
                if "psfk" in kwargs:
                    raise ValueError(
                        "When 'fsmode' is 'convolve' and 'psfmodel' is 'gaussxy', 'psfk' should not be provided; it will be generated from 'psffwhm_x' and 'psffwhm_y'."
                    )
                kwargs["psfk"] = gausskernel2d_elliptical(fwhm_x, fwhm_y, psfsize)
                if show_arguments:
                    print(
                        f"Generated elliptical Gaussian kernel with fwhm_x={fwhm_x}, fwhm_y={fwhm_y}, size={psfsize}."
                    )
            elif kwargs["psfmodel"] in ["gauss", "moffat"]:
                kwargs["psffwhm"] = (fwhm_x + fwhm_y) / 2.0  # average for circular psf
                if show_arguments:
                    print(f"Set psffwhm to average of fwhm_x and fwhm_y: {kwargs['psffwhm']}.")
            elif kwargs["psfmodel"] == "gaussx":
                kwargs["psffwhm"] = fwhm_x
                if show_arguments:
                    print(f"Set psffwhm to fwhm_x: {fwhm_x}.")
            elif kwargs["psfmodel"] == "gaussy":
                kwargs["psffwhm"] = fwhm_y
                if show_arguments:
                    print(f"Set psffwhm to fwhm_y: {fwhm_y}.")
            else:
                raise ValueError(f"Unsupported psfmodel: {kwargs['psfmodel']}")
            if show_arguments:
                print("Deleting 'psffwhm_x' and 'psffwhm_y' from kwargs.")
            del kwargs["psffwhm_x"]
            del kwargs["psffwhm_y"]
        elif fsmode == "median":
            # Remove unnecessary parameters for median fsmode
            for param in ["psfmodel", "psfsize", "psffwhm", "psffwhm_x", "psffwhm_y"]:
                if param in kwargs:
                    if show_arguments:
                        print(f"Removing '{param}' argument since fsmode is 'median'.")
                    del kwargs[param]
        else:
            raise ValueError("The 'fsmode' keyword argument must be either 'convolve' or 'median'.")

    # Show LACosmic arguments if requested
    if show_arguments:
        for key, value in kwargs.items():
            print(f"LACosmic parameter: {key} = {value}")
    # Apply LACosmic algorithm to the padded array
    cleaned_padded_array, mask_padded_array = cosmicray_lacosmic(ccd=padded_array, **kwargs)
    # Remove padding
    cleaned_array = cleaned_padded_array[pad_width:-pad_width, pad_width:-pad_width]
    mask_array = mask_padded_array[pad_width:-pad_width, pad_width:-pad_width]
    return cleaned_array, mask_array
