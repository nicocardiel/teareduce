#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Ask extension dialog for input FITS files."""

from astropy.io import fits
from tkinter import simpledialog, messagebox
from pathlib import Path


def ask_extension_input_image(filename, imgshape=None):
    """Ask the user for the FITS extension to use for the input image.

    Parameters
    ----------
    filename : str
        The name of the FITS file.

    Returns
    -------
    ext : int or str
        The selected FITS extension (0-based index or name).
    """
    # Open the FITS file to get the list of extensions
    try:
        with fits.open(filename) as hdul:
            ext_names = [hdu.name for hdu in hdul]
            ext_indices = list(range(len(hdul)))
            ext_bitpix = [hdu.header["BITPIX"] for hdu in hdul]
    except Exception as e:
        messagebox.showerror("Error", f"Unable to open FITS file '{filename}':\n{str(e)}")
        return None

    if len(ext_indices) == 1:
        ext = 0
    else:
        # ask for the extension number in a dialog
        ext_str = simpledialog.askstring(
            "Select Extension",
            f"\nEnter extension number (0-{len(ext_indices)-1}) for file:\n{Path(filename).name}\n"
            f"Available extensions:\n"
            + "\n".join(
                [f"{i}: {name} (BITPIX={bitpix})" for i, name, bitpix in zip(ext_indices, ext_names, ext_bitpix)]
            ),
        )
        if ext_str is None:
            return None
        # Validate the input
        try:
            ext = int(ext_str)
            if ext < 0 or ext >= len(ext_indices):
                raise ValueError("Extension number out of range")
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Error: {str(e)}")
            return None

    # check if ext is a valid array
    with fits.open(filename) as hdul:
        if hdul[ext] is None:
            messagebox.showerror("Invalid Input", f"Extension {ext} does not exist in file '{filename}'")
            return None
        elif hdul[ext].data is None:
            messagebox.showerror("Invalid Input", f"Extension {ext} in file '{filename}' has no data")
            return None
        elif hdul[ext].data.ndim != 2:
            messagebox.showerror("Invalid Input", f"Extension {ext} in file '{filename}' is not 2D")
            return None
        elif hdul[ext].data.size == 0:
            messagebox.showerror("Invalid Input", f"Extension {ext} in file '{filename}' has no data")
            return None
        elif imgshape is not None and (
            hdul[ext].data.shape[0] != imgshape[0] or hdul[ext].data.shape[1] != imgshape[1]
        ):
            messagebox.showerror(
                "Invalid Input",
                f"Extension {ext} in file '{filename}' has unexpected shape {hdul[ext].data.shape}, expected {imgshape}",
            )
            return None
    return ext
