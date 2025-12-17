#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Merge peak and tail masks for cosmic ray cleaning."""

import numpy as np
from scipy import ndimage
from rich import print


def merge_peak_tail_masks(mask_peaks, mask_tails, verbose=False):
    """Merge peak and tail masks for cosmic ray cleaning.

    Tail pixels are preserved only if they correspond to CR features
    that are also present in the peak mask.

    Parameters
    ----------
    mask_peaks : ndarray
        Boolean array indicating the pixels identified as cosmic ray peaks.
    mask_tails : ndarray
        Boolean array indicating the pixels identified as cosmic ray tails.
    verbose: bool
        If True, print additional information during processing.

    Returns
    -------
    merged_mask : ndarray
        Boolean array indicating the merged cosmic ray mask.
    """
    # check that input masks are numpy arrays
    if not isinstance(mask_peaks, np.ndarray) or not isinstance(mask_tails, np.ndarray):
        raise TypeError("Input masks must be numpy arrays.")
    # check that input masks have the same shape
    if mask_peaks.shape != mask_tails.shape:
        raise ValueError("Input masks must have the same shape.")
    # check that input masks are boolean arrays
    if mask_peaks.dtype != bool or mask_tails.dtype != bool:
        raise TypeError("Input masks must be boolean arrays.")

    if verbose:
        ndigits = np.max([len(str(np.sum(mask_peaks))), len(str(np.sum(mask_tails)))])
        print(f"Number of cosmic ray pixels (peaks)...............: {np.sum(mask_peaks):{ndigits}d}")
        print(f"Number of cosmic ray pixels (tails)...............: {np.sum(mask_tails):{ndigits}d}")
    # find structures in tail mask
    structure = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    cr_labels_tails, num_crs_tails = ndimage.label(mask_tails, structure=structure)
    # generate mask of ones at peak pixels
    mask_peaks_ones = np.zeros(mask_peaks.shape, dtype=float)
    mask_peaks_ones[mask_peaks] = 1.0
    # preserve only those tail pixels that are flagged as peaks
    cr_labels_tails_preserved = mask_peaks_ones * cr_labels_tails
    # generate new mask with preserved tail pixels
    merged_mask = np.zeros_like(mask_peaks, dtype=bool)
    for icr in np.unique(cr_labels_tails_preserved):
        if icr > 0:
            merged_mask[cr_labels_tails == icr] = True

    if verbose:
        print(f"Number of cosmic ray pixels (merged peaks & tails): {np.sum(merged_mask):{ndigits}d}")

    return merged_mask
