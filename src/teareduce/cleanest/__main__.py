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
from tkinter import filedialog
from tkinter import simpledialog
import os
from pathlib import Path
import platform
from rich import print
from rich_argparse import RichHelpFormatter

from .definitions import DEFAULT_FONT_FAMILY
from .definitions import DEFAULT_FONT_SIZE
from .definitions import DEFAULT_TK_WINDOW_SIZE_X
from .definitions import DEFAULT_TK_WINDOW_SIZE_Y
from .cosmicraycleanerapp import CosmicRayCleanerApp
from ..version import VERSION

import matplotlib

matplotlib.use("TkAgg")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive cosmic ray cleaner for FITS images.",
        formatter_class=RichHelpFormatter,
    )
    parser.add_argument("input_fits", nargs="?", default=None, help="Path to the FITS file to be cleaned.")
    parser.add_argument("--extension", type=str, default="0", help="FITS extension to use (default: 0).")
    parser.add_argument("--auxfile", type=str, default=None, help="Auxiliary FITS file")
    parser.add_argument(
        "--extension_auxfile",
        type=str,
        default="0",
        help="FITS extension for auxiliary file (default: 0).",
    )
    parser.add_argument(
        "--fontfamily",
        type=str,
        default=DEFAULT_FONT_FAMILY,
        help=f"Font family for the GUI (default: {DEFAULT_FONT_FAMILY}).",
    )
    parser.add_argument(
        "--fontsize",
        type=int,
        default=DEFAULT_FONT_SIZE,
        help=f"Font size for the GUI (default: {DEFAULT_FONT_SIZE}).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_TK_WINDOW_SIZE_X,
        help=f"Width of the GUI window in pixels (default: {DEFAULT_TK_WINDOW_SIZE_X}).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_TK_WINDOW_SIZE_Y,
        help=f"Height of the GUI window in pixels (default: {DEFAULT_TK_WINDOW_SIZE_Y}).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output.")
    args = parser.parse_args()

    # Welcome message
    print("[bold green]Cosmic Ray Cleaner[/bold green]")
    print("Interactive tool to clean cosmic rays from FITS images.")
    print("teareduce version: " + VERSION)
    print(f"https://nicocardiel.github.io/teareduce-cookbook/docs/cleanest/cleanest.html\n")

    # If input_file is not provided, ask for it using a file dialog
    if args.input_fits is None:
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        args.input_fits = filedialog.askopenfilename(
            title="Select FITS file to be cleaned",
            filetypes=[("FITS files", "*.fits *.fit *.fts"), ("All files", "*.*")],
        )
        if not args.input_fits:
            print("No input FITS file selected. Exiting.")
            exit(1)
        print(f"Selected input FITS file: {args.input_fits}")
        args.extension = simpledialog.askstring(
            "Select Extension",
            f"\nEnter extension number or name for file:\n{Path(args.input_fits).name}",
            initialvalue=args.extension,
        )
        # Ask for auxiliary file if not provided
        if args.auxfile is None:
            use_auxfile = tk.messagebox.askyesno(
                "Auxiliary File",
                "Do you want to use an auxiliary FITS file?",
                default=tk.messagebox.NO,
            )
            if use_auxfile:
                args.auxfile = filedialog.askopenfilename(
                    title="Select Auxiliary FITS file",
                    filetypes=[("FITS files", "*.fits *.fit *.fts"), ("All files", "*.*")],
                    initialfile=args.auxfile,
                )
                if not args.auxfile:
                    print("No auxiliary FITS file selected. Exiting.")
                    exit(1)
        else:
            use_auxfile = True
        if use_auxfile:
            print(f"Selected auxiliary FITS file: {args.auxfile}")
            args.extension_auxfile = simpledialog.askstring(
                "Select Extension for Auxiliary File",
                f"\nEnter extension number or name for auxiliary file:\n{Path(args.auxfile).name}",
                initialvalue=args.extension_auxfile,
            )
        root.destroy()

    # Check that input files, and the corresponding extensions, exist
    if not os.path.isfile(args.input_fits):
        print(f"Error: File '{args.input_fits}' does not exist.")
        exit(1)
    if args.auxfile is not None and not os.path.isfile(args.auxfile):
        print(f"Error: Auxiliary file '{args.auxfile}' does not exist.")
        exit(1)

    # Initialize Tkinter root
    try:
        root = tk.Tk()
    except tk.TclError as e:
        print("Error: Unable to initialize Tkinter. Make sure a display is available.")
        print("Detailed error message:")
        print(e)
        exit(1)
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
    root.lift()  # Bring to front

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
        height=args.height,
        verbose=args.verbose,
    )

    # Execute
    root.mainloop()


if __name__ == "__main__":
    main()
