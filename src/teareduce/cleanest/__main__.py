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

from .cosmicraycleanerapp import CosmicRayCleanerApp

import matplotlib
matplotlib.use("TkAgg")


def main():
    parser = argparse.ArgumentParser(description="Interactive cosmic ray cleaner for FITS images.")
    parser.add_argument("input_fits", help="Path to the FITS file to be cleaned.")
    parser.add_argument("--extension", type=int, default=0,
                        help="FITS extension to use (default: 0).")
    parser.add_argument("--output_fits", type=str, default=None,
                        help="Path to save the cleaned FITS file")
    args = parser.parse_args()

    if not os.path.isfile(args.input_fits):
        print(f"Error: File '{args.input_fits}' does not exist.")
        return
    if args.output_fits is not None and os.path.isfile(args.output_fits):
        print(f"Error: Output file '{args.output_fits}' already exists.")
        return

    # Initialize Tkinter root
    root = tk.Tk()

    # Create and run the application
    CosmicRayCleanerApp(root, args.input_fits, args.extension, args.output_fits)

    # Execute
    root.mainloop()


if __name__ == "__main__":
    main()
