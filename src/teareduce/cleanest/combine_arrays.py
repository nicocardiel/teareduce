#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Combine arrays with different interpolation methods."""

import numpy as np

from .definitions import VALID_ALGORITHMS
from .lacosmicpad import lacosmicpad

VALID_CLEANING_STRATEGIES = ["local", "median_all", "median_others", "mean_others", "none"]


def detect_cosmic_rays(arr, detection_algorithm, **kwargs):
    """Detect cosmic rays in the input array

    Apply the specified algorithm and parameters.

    Parameters
    ----------
    arr : 2D numpy.ndarray
        The input array in which to detect cosmic rays.
    detection_algorithm : str
        The algorithm used to detect cosmic rays. Supported algorithms are:
        - "lacosmic": Use the L.A.Cosmic algorithm for CR detection.
        - "pycosmic": Use the PyCosmic algorithm for CR detection.
        - "deepcr": Use the DeepCR algorithm for CR detection.
        - "conn": Use the Cosmic-CoNN algorithm for CR detection.
    **kwargs : dict
        Additional keyword arguments passed to the CR detection algorithm.

    Returns
    -------
    cleaned_arr : 2D numpy.ndarray
        The cleaned image array after applying the specified CR detection algorithm.
    mask_crfound : 2D numpy.ndarray of bool
        A boolean mask array indicating which pixels are flagged as
        cosmic rays (True = CR pixel).
    """
    # Check input array is a valid numpy 2D array
    if not isinstance(arr, np.ndarray) or arr.ndim != 2:
        raise ValueError("Input array must be a 2D numpy array.")

    # Check detection_algorithm is valid
    if detection_algorithm not in VALID_ALGORITHMS:
        raise ValueError(f"detection_algorithm '{detection_algorithm}' is not one of {VALID_ALGORITHMS}.")

    # Check kwargs is a dictionary
    if not isinstance(kwargs, dict):
        raise ValueError("kwargs must be a dictionary.")

    # Apply the specified detection algorithm to identify cosmic rays in the input array
    if detection_algorithm == "lacosmic":
        if "ccd" in kwargs:
            raise ValueError(
                "The 'ccd' parameter should not be included here.\n"
                "It will be created internally using the different arrays\n"
                "in the 'list_arrays' provided to the combine_arrays function."
            )
        kwargs["ccd"] = arr
        if "pad_width" not in kwargs:
            raise ValueError("The 'pad_width' parameter must be included in 'kwargs'")
        cleaned_arr, mask_crfound = lacosmicpad(**kwargs)
    elif detection_algorithm == "pycosmic":
        raise NotImplementedError("PyCosmic detection algorithm is not yet implemented in this function.")
    elif detection_algorithm == "deepcr":
        raise NotImplementedError("DeepCR detection algorithm is not yet implemented in this function.")
    elif detection_algorithm == "conn":
        raise NotImplementedError("Cosmic-CoNN detection algorithm is not yet implemented in this function.")
    else:
        raise ValueError(f"Invalid detection_algorithm. Must be one of {VALID_ALGORITHMS}.")

    return cleaned_arr, mask_crfound


def clean_array(arr, mask_crfound, strategy, all_arrays=None, other_arrays=None):
    """Clean the input array based on the specified strategy and mask.

    This function replaces the pixels flagged in `mask_crfound`
    using the information from the other arrays in `all_arrays` or `other_arrays`
    according to the specified `strategy`.

    Parameters
    ----------
    arr : 2D numpy.ndarray
        The input array to be cleaned.
    mask_crfound : 2D numpy.ndarray of bool
        A boolean mask array indicating which pixels are flagged as
        cosmic rays (True = CR pixel).
    strategy : str
        The cleaning strategy to be applied.
    all_arrays : list of 2D numpy.ndarray, optional
        A list of all arrays, used for certain cleaning strategies.
    other_arrays : list of 2D numpy.ndarray, optional
        A list of arrays excluding the one being cleaned,
        used for certain cleaning strategies.

    Returns
    -------
    cleaned_arr : 2D numpy.ndarray
        The cleaned array after applying the specified strategy.
    """
    # Check input array is a valid numpy 2D array
    if not isinstance(arr, np.ndarray) or arr.ndim != 2:
        raise ValueError("Input array must be a 2D numpy array.")

    # Check mask_crfound is a boolean array of the same shape as arr
    if not isinstance(mask_crfound, np.ndarray) or mask_crfound.shape != arr.shape or mask_crfound.dtype != bool:
        raise ValueError("mask_crfound must be a boolean array of the same shape as arr.")

    # Check strategy is valid
    if strategy not in VALID_CLEANING_STRATEGIES:
        raise ValueError(f"strategy '{strategy}' is not one of {VALID_CLEANING_STRATEGIES}.")

    # Check strategy is not "local" nor "none", since this function is not intended
    # to be used for those strategies
    if strategy in ["local", "none"]:
        raise ValueError(
            f"The '{strategy}' strategy should not be used in this function,\n"
            "since it does not use the information from the other arrays\n"
            "to clean the cosmic ray pixels."
        )

    # Check that only one of all_arrays or other_arrays is provided, not both
    if all_arrays is not None and other_arrays is not None:
        raise ValueError("Only one of 'all_arrays' or 'other_arrays' should be provided, not both.")

    # If all_arrays is provided, check it is a list of 2D arrays of the same shape as arr
    if all_arrays is not None:
        if not isinstance(all_arrays, list) or len(all_arrays) == 0:
            raise ValueError("all_arrays must be a non-empty list of 2D numpy arrays.")
        for a in all_arrays:
            if not isinstance(a, np.ndarray) or a.ndim != 2:
                raise ValueError("All elements in all_arrays must be 2D numpy arrays.")
            if a.shape != arr.shape:
                raise ValueError("All arrays in all_arrays must have the same shape as arr.")

    # If other_arrays is provided, check it is a list of 2D arrays of the same shape as arr
    if other_arrays is not None:
        if not isinstance(other_arrays, list) or len(other_arrays) == 0:
            raise ValueError("other_arrays must be a non-empty list of 2D numpy arrays.")
        for a in other_arrays:
            if not isinstance(a, np.ndarray) or a.ndim != 2:
                raise ValueError("All elements in other_arrays must be 2D numpy arrays.")
            if a.shape != arr.shape:
                raise ValueError("All arrays in other_arrays must have the same shape as arr.")

    # Clean the array based on the specified strategy
    if strategy == "median_all":
        cleaned_arr = np.copy(arr)
        cleaned_arr[mask_crfound] = np.median([a[mask_crfound] for a in all_arrays], axis=0)
    elif strategy == "median_others":
        cleaned_arr = np.copy(arr)
        cleaned_arr[mask_crfound] = np.median([a[mask_crfound] for a in other_arrays], axis=0)
    elif strategy == "mean_others":
        cleaned_arr = np.copy(arr)
        cleaned_arr[mask_crfound] = np.mean([a[mask_crfound] for a in other_arrays], axis=0)
    else:
        raise ValueError(f"Invalid strategy '{strategy}'.")


def combine_arrays(
    list_arrays, detection_algorithm, cleaning_strategy, combination_method, show_progress=False, **kwargs
):
    """Combine arrays with different interpolation methods.

    The function takes a list of 2D arrays, applies the specified
    cleaning method to each individual array using the
    chosen strategy, and then combines the cleaned arrays employing
    one of the available combination methods.

    Parameters
    ----------
    list_arrays : list of 2D numpy.ndarray
        A list of 2D arrays to be combined.
    detection_algorithm : str
        The algorithm used to detect cosmic rays. Supported algorithms are:
        - "lacosmic": Use the L.A.Cosmic algorithm for CR detection.
        - "pycosmic": Use the PyCosmic algorithm for CR detection.
        - "deepcr": Use the DeepCR algorithm for CR detection.
        - "conn": Use the Cosmic-CoNN algorithm for CR detection.
    cleaning_strategy : str
        The strategy used for cleaning the arrays. Supported strategies are:
        - "local": replace the cosmic ray pixels in each array with the values
          obtained by applying the specified detection algorithm to that array
          alone. Note that this strategy does not use the information from the
          other arrays in `list_arrays` to clean the cosmic ray pixels,
          but only to combine the results after cleaning.
        - "median_all": replace the cosmic ray pixels with the median of all
          the arrays in `list_arrays`.
        - "median_others": replace the cosmic ray pixels with the median of all the
            arrays in `list_arrays` except the one being cleaned.
        - "mean_others": replace the cosmic ray pixels with the mean of all the
            arrays in `list_arrays` except the one being cleaned.
        - "none": do not attempt to clean the arrays, just generate
            the masks of the detected cosmic rays and combine them using the
            masked version of the specified combination method.
    combination_method : str
        The method used to combine the arrays. Supported methods are:
        - "median": median of the individually cleaned or masked arrays.
        - "mean": mean of the individually cleaned or masked arrays.
    show_progress : bool, optional
        If True, print show_progress information during the combination process.
    **kwargs : dict
        Additional keyword arguments passed to the CR detection algorithm.

    Returns
    -------
    combined_array : 2D numpy.ndarray
        The resulting array after combining the input arrays using the specified methods.
    """

    # Check list_arrays is a list of 2D arrays of the same shape
    if not isinstance(list_arrays, list) or len(list_arrays) == 0:
        raise ValueError("list_arrays must be a non-empty list of 2D numpy arrays.")
    array_shape = list_arrays[0].shape
    for arr in list_arrays:
        if not isinstance(arr, np.ndarray) or arr.ndim != 2:
            raise ValueError("All elements in list_arrays must be 2D numpy arrays.")
        if arr.shape != array_shape:
            raise ValueError("All arrays in list_arrays must have the same shape.")

    # Check combination_method is valid
    if combination_method not in ["median", "mean"]:
        raise ValueError("combination_method must be either 'median' or 'mean'.")

    # Check detection_algorithm is valid
    if detection_algorithm not in VALID_ALGORITHMS:
        raise ValueError(f"detection_algorithm '{detection_algorithm}' is not one of {VALID_ALGORITHMS}.")

    # Check cleaning_strategy is valid
    if cleaning_strategy not in VALID_CLEANING_STRATEGIES:
        raise ValueError(f"cleaning_strategy '{cleaning_strategy}' is not one of {VALID_CLEANING_STRATEGIES}.")

    # Check combination_method is valid
    if combination_method not in ["median", "mean"]:
        raise ValueError("combination_method must be either 'median' or 'mean'.")

    # When using the "local" cleaning strategy, check that the detection
    # algorithm is not "conn", since this algorithm
    # do not provide a cleaned array, but only a mask of detected CRs
    if cleaning_strategy == "local" and detection_algorithm == "conn":
        raise ValueError(
            "The 'local' cleaning strategy is not compatible with the 'conn'\n"
            "detection algorithm, since it does not provide a cleaned array,\n"
            "but only a mask of detected CRs."
        )

    # Define a 3D masked array to hold the cleaned or masked arrays
    cleaned_arrays = np.ma.masked_array(np.zeros((len(list_arrays), *array_shape)), mask=False)

    # Loop over each array, detect cosmic rays, and clean or mask them
    for i, arr in enumerate(list_arrays):
        if show_progress:
            print(f"Processing array {i+1}/{len(list_arrays)}...")

        # Detect cosmic rays using the specified algorithm and parameters
        cleaned_arr, mask_crfound = detect_cosmic_rays(arr, detection_algorithm, **kwargs)

        # Clean or mask the array based on the cleaning strategy
        if cleaning_strategy == "local":
            # For the "local" strategy, we keep the cleaned array as is,
            # without using the information from the other arrays
            pass
        elif cleaning_strategy == "median_all":
            # Overwrite the cleaned_arr with the median of all arrays
            # in list_arrays at the positions of the detected CRs
            cleaned_arr = clean_array(arr, mask_crfound, strategy="median_all", all_arrays=list_arrays)
            # After cleaning with median_all, no pixels should remain masked
            mask_crfound = np.zeros_like(arr, dtype=bool)
        elif cleaning_strategy == "median_others":
            # Overwrite the cleaned_arr with the median of all arrays
            # in list_arrays except the one being cleaned
            cleaned_arr = clean_array(
                arr,
                mask_crfound,
                strategy="median_others",
                other_arrays=[a for j, a in enumerate(list_arrays) if j != i],
            )
            # After cleaning with median_others, no pixels should remain masked
            mask_crfound = np.zeros_like(arr, dtype=bool)
        elif cleaning_strategy == "mean_others":
            # Overwrite the cleaned_arr with the mean of all arrays
            # in list_arrays except the one being cleaned
            cleaned_arr = clean_array(
                arr, mask_crfound, strategy="mean_others", other_arrays=[a for j, a in enumerate(list_arrays) if j != i]
            )
            # After cleaning with mean_others, no pixels should remain masked
            mask_crfound = np.zeros_like(arr, dtype=bool)
        elif cleaning_strategy == "none":
            # For the "none" strategy, we do not attempt to clean the array,
            # but we will use the mask of detected CRs to combine the arrays
            cleaned_arr = arr.copy()
        else:
            raise ValueError(f"Invalid cleaning_strategy '{cleaning_strategy}'.")

        # Store the cleaned array and its corresponding mask in the 3D masked array
        cleaned_arrays[i] = np.ma.masked_array(cleaned_arr, mask=mask_crfound)

    # Combine the cleaned or masked arrays using the specified combination method
    if show_progress:
        print(f"Combining arrays using method '{combination_method}'...")
    if combination_method == "median":
        combined_array = np.ma.median(cleaned_arrays, axis=0).filled(np.nan)
    elif combination_method == "mean":
        combined_array = np.ma.mean(cleaned_arrays, axis=0).filled(np.nan)
    else:
        raise ValueError("Invalid combination_method. Must be 'median' or 'mean'.")

    return combined_array
