#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Interactive Cosmic Ray cleaning tool."""

import argparse
import tkinter as tk
import os
import platform
from rich import print
from rich_argparse import RichHelpFormatter

from .definitions import DEFAULT_FONT_FAMILY
from .definitions import DEFAULT_FONT_SIZE
from .definitions import DEFAULT_TK_WINDOW_SIZE_X
from .definitions import DEFAULT_TK_WINDOW_SIZE_Y
from .cosmicraycleanerapp import CosmicRayCleanerApp

import matplotlib
matplotlib.use("TkAgg")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive cosmic ray cleaner for FITS images.",
        formatter_class=RichHelpFormatter)
    parser.add_argument("input_fits", help="Path to the FITS file to be cleaned.")
    parser.add_argument("--extension", type=int, default=0,
                        help="FITS extension to use (default: 0).")
    parser.add_argument("--auxfile", type=str, default=None,
                        help="Auxiliary FITS file")
    parser.add_argument("--extension_auxfile", type=int, default=0,
                        help="FITS extension for auxiliary file (default: 0).")
    parser.add_argument("--fontfamily", type=str, default=DEFAULT_FONT_FAMILY,
                        help=f"Font family for the GUI (default: {DEFAULT_FONT_FAMILY}).")
    parser.add_argument("--fontsize", type=int, default=DEFAULT_FONT_SIZE,
                        help=f"Font size for the GUI (default: {DEFAULT_FONT_SIZE}).")
    parser.add_argument("--width", type=int, default=DEFAULT_TK_WINDOW_SIZE_X,
                        help=f"Width of the GUI window in pixels (default: {DEFAULT_TK_WINDOW_SIZE_X}).")
    parser.add_argument("--height", type=int, default=DEFAULT_TK_WINDOW_SIZE_Y,
                        help=f"Height of the GUI window in pixels (default: {DEFAULT_TK_WINDOW_SIZE_Y}).")
    args = parser.parse_args()

    if not os.path.isfile(args.input_fits):
        print(f"Error: File '{args.input_fits}' does not exist.")
        return
    if args.auxfile is not None and not os.path.isfile(args.auxfile):
        print(f"Error: Auxiliary file '{args.auxfile}' does not exist.")
        return

    # Initialize Tkinter root
    root = tk.Tk()
    system = platform.system()
    if system == "Darwin":  # macOS
        # Center the window on the screen
        xoffset = root.winfo_screenwidth() // 2 - args.width // 2
        yoffset = root.winfo_screenheight() // 2 - args.height // 2
    else:
        # Note that geometry("XxY+Xoffset+Yoffset") does not work properly on Fedora Linux
        xoffset = 0
        yoffset = 0
    root.geometry(f"+{xoffset}+{yoffset}")
    root.focus_force()  # Request focus
    root.lift()         # Bring to front

    # Create and run the application
    CosmicRayCleanerApp(
        root=root,
        input_fits=args.input_fits,
        extension=args.extension,
        auxfile=args.auxfile,
        extension_auxfile=args.extension_auxfile,
        fontfamily=args.fontfamily,
        fontsize=args.fontsize,
        width=args.width,
        height=args.height
    )

    # Execute
    root.mainloop()


if __name__ == "__main__":
    main()
