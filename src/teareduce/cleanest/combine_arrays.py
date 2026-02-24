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

try:
    import PyCosmic

    PYCOSMIC_AVAILABLE = True
except ModuleNotFoundError as e:
    PYCOSMIC_AVAILABLE = False

try:
    import deepCR

    DEEPCR_AVAILABLE = True
except ModuleNotFoundError as e:
    DEEPCR_AVAILABLE = False

try:
    import cosmic_conn

    COSMIC_CONN_AVAILABLE = True
except ModuleNotFoundError as e:
    COSMIC_CONN_AVAILABLE = False

from .definitions import VALID_ALGORITHMS
from .lacosmicpad import lacosmicpad
from .mergemasks import merge_peak_tail_masks

VALID_CLEANING_STRATEGIES = ["crmethod", "median_all", "median_others", "mean_others", "none"]


def split_in_two_dictionaries(kwargs):
    """Split the input kwargs dictionary into two dictionaries.

    The input dictionary contains the input parameters for a CR detection
    algorithm. This algorithm can be run once or twice, depending on the
    value of the parameters. If all the parameters are single values,
    the algorithm will be run once, but if some parameters are lists or
    tuples of two values, the algorithm will be run twice, using the first
    value for the first run and the second value for the second run.
    This function splits the input dictionary into two dictionaries,
    one for the first run and one for the second run. If all the input
    parameters are single values, or if all the items in the lists/tuples
    are equal, the second dictionary will be identical to the first and the
    function will return a list containing only the first dictionary.

    Parameters
    ----------
    kwargs : dict
        The input dictionary containing the parameters for the CR detection
        algorithm.

    Returns
    -------
    list_kwargs: list of one or two dictionaries
        A list containing one or two dictionaries, one for the first run
        and one for the second run, if applicable.
    """
    # Check kwargs is a dictionary
    if not isinstance(kwargs, dict):
        raise ValueError("kwargs must be a dictionary.")

    # Create two dictionaries for the first and second run
    kwargs_run1 = {}
    kwargs_run2 = {}

    # Loop over the input dictionary and split the parameters
    all_equal = True
    for key, value in kwargs.items():
        if isinstance(value, (list, tuple)) and len(value) == 2:
            kwargs_run1[key] = value[0]
            kwargs_run2[key] = value[1]
            if value[0] != value[1]:
                all_equal = False
        else:
            if isinstance(value, (int, float, str, bool)):
                kwargs_run1[key] = value
                kwargs_run2[key] = value
            else:
                raise ValueError(
                    f"Invalid value for parameter '{key}'.\n"
                    "Values must be either single values (int, float, str, bool) or lists/tuples of two values."
                )

    if all_equal:
        list_kwargs = [kwargs_run1]
    else:
        list_kwargs = [kwargs_run1, kwargs_run2]
    return list_kwargs


def check_double_2element_list(value):
    """Check if the input value is a list of a double list of two elements.

    This is used to check if the input parameters for the CR detection
    algorithm is a single list of two numbers (integer or float) or a list
    of two lists of two numbers, which would indicate that the algorithm should
    be run twice, using the first list for the first run and the second list
    for the second run.

    Parameters
    ----------
    value : any
        The input value to be checked.

    Returns
    -------
    is_double_2element_list : bool
        True if the input value is a list of a double list of two elements,
        False otherwise.
    """
    if isinstance(value, (list, tuple)) and len(value) == 2:
        if all(isinstance(v, (int, float)) for v in value):
            return False
        elif all(
            isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(n, (int, float)) for n in v) for v in value
        ):
            return True
    raise ValueError(
        f"Invalid value: {value}. Value must be either a list of two numbers (int or float) or a list of two lists of two numbers."
    )


def detect_cosmic_rays(arr, detection_algorithm, show_progress, **kwargs):
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
    show_progress : bool
        If True, print show_progress information during the CR detection process.
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
    if not PYCOSMIC_AVAILABLE and detection_algorithm == "pycosmic":
        raise ValueError(
            "The 'teareduce.cleanest' module requires the 'PyCosmic' package.\n"
            "Please install this module using:\n"
            "`pip install git+https://github.com/nicocardiel/PyCosmic.git@test`"
        )

    if not DEEPCR_AVAILABLE and detection_algorithm == "deepcr":
        raise ValueError(
            "The 'teareduce.cleanest' module requires the 'deepCR' package. "
            "Please install teareduce with the 'cleanest' extra dependencies: "
            '`pip install "teareduce[cleanest]"`.'
        )

    if not COSMIC_CONN_AVAILABLE and detection_algorithm == "conn":
        raise ValueError(
            "The 'teareduce.cleanest' module requires the 'cosmic-conn' package. "
            "Please install teareduce with the 'cleanest' extra dependencies: "
            '`pip install "teareduce[cleanest]"`.'
        )

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
        if "pad_width" not in kwargs:
            raise ValueError("The 'pad_width' parameter must be included in 'kwargs'")
        list_kwargs = split_in_two_dictionaries(kwargs)
        list_kwargs[0]["ccd"] = arr  # include the input array in the parameters for the first run of the algorithm
        if show_progress:
            print(f"Running L.A.Cosmic algorithm (run 1/{len(list_kwargs)})...")
        cleaned_arr, mask_crfound = lacosmicpad(**list_kwargs[0])
        if len(list_kwargs) == 2:
            # If there are two sets of parameters, run the algorithm a second time
            # with the second set of parameters. Note that the cleaned array from
            # the first run is overwritten here
            list_kwargs[1]["ccd"] = arr  # include the input array in the parameters for the second run of the algorithm
            if show_progress:
                print(f"Running L.A.Cosmic algorithm (run 2/{len(list_kwargs)})...")
            cleaned_arr, mask_crfound_2 = lacosmicpad(**list_kwargs[1])
            # Merge the two masks of detected CRs using the merge_peak_tail_masks function
            if show_progress:
                print("Merging masks of detected cosmic rays from both runs...")
            mask_crfound = merge_peak_tail_masks(mask_crfound, mask_crfound_2, verbose=show_progress)
    elif detection_algorithm == "pycosmic":
        if "data" in kwargs:
            raise ValueError(
                "The 'data' parameter should not be included here.\n"
                "It will be created internally using the different arrays\n"
                "in the 'list_arrays' provided to the combine_arrays function."
            )
        if "fwhm_gauss" in kwargs:
            fwhm_gauss = kwargs["fwhm_gauss"]
            if not check_double_2element_list(fwhm_gauss):
                list_fwhm_gauss = [fwhm_gauss, fwhm_gauss]
            else:
                list_fwhm_gauss = fwhm_gauss
            del kwargs["fwhm_gauss"]
        else:
            fwhm_gauss = None
        if "replace_box" in kwargs:
            replace_box = kwargs["replace_box"]
            if not check_double_2element_list(replace_box):
                list_replace_box = [replace_box, replace_box]
            else:
                list_replace_box = replace_box
            del kwargs["replace_box"]
        else:
            replace_box = None
        list_kwargs = split_in_two_dictionaries(kwargs)
        list_kwargs[0]["data"] = arr  # include the input array in the parameters for the first run of the algorithm
        if fwhm_gauss is not None:
            list_kwargs[0]["fwhm_gauss"] = list_fwhm_gauss[0]
        if replace_box is not None:
            list_kwargs[0]["replace_box"] = list_replace_box[0]
        if show_progress:
            print(f"Running PyCosmic algorithm (run 1/{len(list_kwargs)})...")
        out = PyCosmic.det_cosmics(**list_kwargs[0])
        cleaned_arr = out.data
        mask_crfound = out.mask.astype(bool)
        if len(list_kwargs) == 2:
            # If there are two sets of parameters, run the algorithm a second time
            # with the second set of parameters. Note that the cleaned array from
            # the first run is overwritten here
            list_kwargs[1][
                "data"
            ] = arr  # include the input array in the parameters for the second run of the algorithm
            if fwhm_gauss is not None:
                list_kwargs[1]["fwhm_gauss"] = list_fwhm_gauss[1]
            if replace_box is not None:
                list_kwargs[1]["replace_box"] = list_replace_box[1]
            if show_progress:
                print(f"Running PyCosmic algorithm (run 2/{len(list_kwargs)})...")
            out_2 = PyCosmic.det_cosmics(**list_kwargs[1])
            cleaned_arr = out_2.data
            mask_crfound_2 = out_2.mask.astype(bool)
            # Merge the two masks of detected CRs using the merge_peak_tail_masks function
            if show_progress:
                print("Merging masks of detected cosmic rays from both runs...")
            mask_crfound = merge_peak_tail_masks(mask_crfound, mask_crfound_2, verbose=show_progress)
    elif detection_algorithm == "deepcr":
        if "img0" in kwargs:
            raise ValueError(
                "The 'img0' parameter should not be included here.\n"
                "It will be created internally using the different arrays\n"
                "in the 'list_arrays' provided to the combine_arrays function."
            )
        mdl = deepCR.deepCR(mask="ACS-WFC")
        list_kwargs = split_in_two_dictionaries(kwargs)
        list_kwargs[0]["img0"] = arr  # include the input array in the parameters for the first run of the algorithm
        if show_progress:
            print(f"Running DeepCR algorithm (run 1/{len(list_kwargs)})...")
        mask_crfound, cleaned_arr = mdl.clean(**list_kwargs[0])
        mask_crfound = mask_crfound.astype(bool)
        if len(list_kwargs) == 2:
            # If there are two sets of parameters, run the algorithm a second time
            # with the second set of parameters. Note that the cleaned array from
            # the first run is overwritten here
            list_kwargs[1][
                "img0"
            ] = arr  # include the input array in the parameters for the second run of the algorithm
            if show_progress:
                print(f"Running DeepCR algorithm (run 2/{len(list_kwargs)})...")
            mask_crfound_2, cleaned_arr = mdl.clean(**list_kwargs[1])
            mask_crfound_2 = mask_crfound_2.astype(bool)
            # Merge the two masks of detected CRs using the merge_peak_tail_masks function
            if show_progress:
                print("Merging masks of detected cosmic rays from both runs...")
            mask_crfound = merge_peak_tail_masks(mask_crfound, mask_crfound_2, verbose=show_progress)
    elif detection_algorithm == "conn":
        if "image" in kwargs:
            raise ValueError(
                "The 'image' parameter should not be included here.\n"
                "It will be created internally using the different arrays\n"
                "in the 'list_arrays' provided to the combine_arrays function."
            )
        cr_model = cosmic_conn.init_model("ground_imaging")
        list_kwargs = split_in_two_dictionaries(kwargs)
        list_kwargs[0]["image"] = arr.astype(
            np.float32
        )  # include the input array in the parameters for the first run of the algorithm
        if show_progress:
            print(f"Running Cosmic-CoNN algorithm (run 1/{len(list_kwargs)})...")
        cr_prob = cr_model.detect_cr(list_kwargs[0]["image"])
        if "threshold" in list_kwargs[0]:
            threshold = list_kwargs[0]["threshold"]
        else:
            raise ValueError(
                "The 'threshold' parameter must be included in 'kwargs' for the 'conn' detection algorithm."
            )
        mask_crfound = cr_prob > threshold
        if len(list_kwargs) == 2:
            # If there are two sets of parameters, run the algorithm a second time
            # with the second set of parameters. Note that the cleaned array from
            # the first run is overwritten here
            list_kwargs[1]["image"] = arr.astype(
                np.float32
            )  # include the input array in the parameters for the second run of the algorithm
            if show_progress:
                print(f"Running Cosmic-CoNN algorithm (run 2/{len(list_kwargs)})...")
            cr_prob_2 = cr_model.detect_cr(list_kwargs[1]["image"])
            if "threshold" in list_kwargs[1]:
                threshold_2 = list_kwargs[1]["threshold"]
            else:
                raise ValueError(
                    "The 'threshold' parameter must be included in 'kwargs' for the 'conn' detection algorithm."
                )
            mask_crfound_2 = cr_prob_2 > threshold_2
            # Merge the two masks of detected CRs using the merge_peak_tail_masks function
            if show_progress:
                print("Merging masks of detected cosmic rays from both runs...")
            mask_crfound = merge_peak_tail_masks(mask_crfound, mask_crfound_2, verbose=show_progress)
        # For the 'conn' algorithm, we do not have a cleaned array, but only a mask of detected CRs, so we will return the input array as the cleaned array
        cleaned_arr = arr.copy()
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

    # Check strategy is not "crmethod" nor "none", since this function is not intended
    # to be used for those strategies
    if strategy in ["crmethod", "none"]:
        raise ValueError(
            f"The '{strategy}' strategy should not be used in this function,\n"
            "since it does not use the information from the other arrays\n"
            "to clean the cosmic ray pixels."
        )

    # Check that only one of all_arrays or other_arrays is provided, not both
    if all_arrays is not None and other_arrays is not None:
        raise ValueError("Only one of 'all_arrays' or 'other_arrays' should be provided, not both.")

    # Check that at least one of all_arrays or other_arrays is provided
    if all_arrays is None and other_arrays is None:
        raise ValueError("At least one of 'all_arrays' or 'other_arrays' must be provided.")

    # If all_arrays is provided, check it is a list of 2D arrays of the same shape as arr
    if all_arrays is not None:
        if not isinstance(all_arrays, list) or len(all_arrays) == 0:
            raise ValueError("all_arrays must be a non-empty list of 2D numpy arrays.")
        naux = len(all_arrays)
        image3d = np.zeros((naux, *arr.shape))
        for i, a in enumerate(all_arrays):
            if not isinstance(a, np.ndarray) or a.ndim != 2:
                raise ValueError("All elements in all_arrays must be 2D numpy arrays.")
            if a.shape != arr.shape:
                raise ValueError("All arrays in all_arrays must have the same shape as arr.")
            image3d[i] = a

    # If other_arrays is provided, check it is a list of 2D arrays of the same shape as arr
    if other_arrays is not None:
        if not isinstance(other_arrays, list) or len(other_arrays) == 0:
            raise ValueError("other_arrays must be a non-empty list of 2D numpy arrays.")
        naux = len(other_arrays)
        image3d = np.zeros((naux, *arr.shape))
        for i, a in enumerate(other_arrays):
            if not isinstance(a, np.ndarray) or a.ndim != 2:
                raise ValueError("All elements in other_arrays must be 2D numpy arrays.")
            if a.shape != arr.shape:
                raise ValueError("All arrays in other_arrays must have the same shape as arr.")
            image3d[i] = a

    # Clean the array based on the specified strategy
    if strategy in ["median_all", "median_others"]:
        cleaned_arr = np.copy(arr)
        cleaned_arr[mask_crfound] = np.median(image3d, axis=0)[mask_crfound]
    elif strategy == "mean_others":
        cleaned_arr = np.copy(arr)
        cleaned_arr[mask_crfound] = np.mean(image3d, axis=0)[mask_crfound]
    else:
        raise ValueError(f"Invalid strategy '{strategy}'.")

    return cleaned_arr


def combine_arrays(
    list_arrays,
    detection_algorithm,
    cleaning_strategy,
    combination_method,
    show_progress=False,
    return_array_mask_lists=False,
    **kwargs,
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
        - "crmethod": replace the cosmic ray pixels in each array with the values
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
    return_array_mask_lists : bool, optional
        If True, return a list of the individually cleaned arrays and another
        list with the corresponding masks.
    **kwargs : dict
        Additional keyword arguments passed to the CR detection algorithm.

    Returns
    -------
    combined_array : 2D numpy.ndarray
        The resulting array after combining the input arrays using
        the specified methods.
    out_list_arrays: list of 2D numpy.ndarray, optional
        If return_list_arrays_masks is True, a list of the individually
        cleaned arrays.
    out_list_masks: list of 2D numpy.ndarray of bool, optional
        If return_list_arrays_masks is True, a list of the corresponding
        masks of detected CRs for each array.
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
    if len(list_arrays) < 2:
        raise ValueError("list_arrays must contain at least two arrays to combine.")

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

    # When using the "crmethod" cleaning strategy, check that the detection
    # algorithm is not "conn", since this algorithm
    # do not provide a cleaned array, but only a mask of detected CRs
    if cleaning_strategy == "crmethod" and detection_algorithm == "conn":
        raise ValueError(
            "The 'crmethod' cleaning strategy is not compatible with the 'conn'\n"
            "detection algorithm, since it does not provide a cleaned array,\n"
            "but only a mask of detected CRs."
        )

    # Define a 3D masked array to hold the cleaned or masked arrays
    cleaned_arrays = np.ma.masked_array(np.zeros((len(list_arrays), *array_shape)), mask=False)

    # If return_array_mask_lists is True, define lists to hold the individually
    # cleaned arrays and their corresponding masks
    if return_array_mask_lists:
        out_list_arrays = []
        out_list_masks = []

    # Loop over each array, detect cosmic rays, and clean or mask them
    for i, arr in enumerate(list_arrays):
        if show_progress:
            if i > 0:
                print()  # print a new line before the next progress message
            print(f"Processing array {i+1}/{len(list_arrays)}...")

        # Detect cosmic rays using the specified algorithm and parameters
        cleaned_arr, mask_crfound = detect_cosmic_rays(arr, detection_algorithm, show_progress, **kwargs)

        # Clean or mask the array based on the cleaning strategy
        if cleaning_strategy == "crmethod":
            # For the "crmethod" strategy, we keep the cleaned array as is,
            # without using the information from the other arrays
            pass
        elif cleaning_strategy == "median_all":
            # Overwrite the cleaned_arr with the median of all arrays
            # in list_arrays at the positions of the detected CRs
            cleaned_arr = clean_array(arr, mask_crfound, strategy="median_all", all_arrays=list_arrays)
        elif cleaning_strategy == "median_others":
            # Overwrite the cleaned_arr with the median of all arrays
            # in list_arrays except the one being cleaned
            cleaned_arr = clean_array(
                arr,
                mask_crfound,
                strategy="median_others",
                other_arrays=[a for j, a in enumerate(list_arrays) if j != i],
            )
        elif cleaning_strategy == "mean_others":
            # Overwrite the cleaned_arr with the mean of all arrays
            # in list_arrays except the one being cleaned
            cleaned_arr = clean_array(
                arr, mask_crfound, strategy="mean_others", other_arrays=[a for j, a in enumerate(list_arrays) if j != i]
            )

        elif cleaning_strategy == "none":
            # For the "none" strategy, we do not attempt to clean the array,
            # but we will use the mask of detected CRs to combine the arrays
            cleaned_arr = arr.copy()
        else:
            raise ValueError(f"Invalid cleaning_strategy '{cleaning_strategy}'.")

        # If return_array_mask_lists is True, store the cleaned array and
        # its corresponding mask in the output lists
        if return_array_mask_lists:
            out_list_arrays.append(cleaned_arr)
            out_list_masks.append(mask_crfound)

        # For the "none" cleaning strategy, we will keep the mask of detected CRs
        # to combine the arrays using the masked version of the specified
        # combination method, but for the other strategies, after cleaning,
        # no pixels should remain masked, so we will set the mask to False for all pixels
        if cleaning_strategy != "none":
            mask_crfound = np.zeros_like(arr, dtype=bool)

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
    if show_progress:
        print("\nCombination complete!")

    if return_array_mask_lists:
        return combined_array, out_list_arrays, out_list_masks
    else:
        return combined_array
