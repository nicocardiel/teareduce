#
# Copyright 2025-2026 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Interactive Cosmic Ray cleaning tool."""

import argparse
import glob
import tkinter as tk
from tkinter import filedialog
import os
import platform
from rich import print
from rich_argparse import RichHelpFormatter

from .askextension import ask_extension_input_image
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
    parser.add_argument("--auxfile", type=str, default=None, help="Auxiliary FITS files (comma-separated if several).")
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
            initialdir=os.getcwd(),
            filetypes=[("FITS files", "*.fits *.fit *.fts"), ("All files", "*.*")],
        )
        if not args.input_fits:
            print("No input FITS file selected. Exiting.")
            exit(1)
        print(f"Selected input FITS file: {args.input_fits}")
        args.extension = ask_extension_input_image(args.input_fits, imgshape=None)
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
            extension_auxfile = ask_extension_input_image(args.auxfile, imgshape=None)
            args.auxfile = f"{args.auxfile}[{extension_auxfile}]"
        else:
            args.auxfile = None
        root.destroy()

    # Check that input file exists
    if "[" in args.input_fits:
        input_fits = args.input_fits[: args.input_fits.index("[")]
        extension = args.input_fits[args.input_fits.index("[") + 1 : args.input_fits.index("]")]
    else:
        input_fits = args.input_fits
        extension = "0"
    if not os.path.isfile(input_fits):
        print(f"Error: File '{input_fits}' does not exist.")
        exit(1)

    # Process auxiliary files if provided
    auxfile_list = []
    extension_auxfile_list = []
    if args.auxfile is not None:
        # Check several auxiliary files separated by commas
        if "\n" in args.auxfile:
            if "?" in args.auxfile or "*" in args.auxfile:
                print("Error: Cannot combine newlines and wildcards in --auxfile specification.")
                exit(1)
            # Replace newlines by commas to allow:
            # --auxfile="`ls file??.fits`"   -> without extensions
            # --auxfile="`for f in file??.fits; do echo "${f}[primary]"; done`"  -> with extension name (the same for all files!)
            args.auxfile = args.auxfile.replace("\n", ",")
        elif "?" in args.auxfile or "*" in args.auxfile:
            # Handle wildcards to allow:
            # --auxfile="file??.fits"   -> without extensions
            # --auxfile="file??.fits[primary]"  -> with extension name (the same for all files!)
            if "," in args.auxfile:
                print("Error: Cannot combine wildcards with commas in --auxfile specification.")
                exit(1)
            # Expand possible wildcards in the auxiliary file specification
            if "[" in args.auxfile:
                s = args.auxfile.strip()
                ext = None
                if s.endswith("]"):
                    ext = s[s.rfind("[") + 1 : s.rfind("]")]
                    s = s[: s.rfind("[")]
                matched_files = sorted(glob.glob(s))
                if len(matched_files) == 0:
                    print(f"Error: No files matched the pattern '{s}'.")
                    exit(1)
                args.auxfile = ",".join(
                    [f"{fname}[{ext}]" if ext is not None else fname for fname in matched_files]
                )
            else:
                matched_files = sorted(glob.glob(args.auxfile))
                if len(matched_files) == 0:
                    print(f"Error: No files matched the pattern '{args.auxfile}'.")
                    exit(1)
                args.auxfile = ",".join(matched_files)
        # Now process the comma-separated auxiliary files
        for item in args.auxfile.split(","):
            # Extract possible [ext] from filename
            if "[" in item:
                # Separate filename and extension, removing blanks
                fname = item[: item.index("[")].strip()
                # Check that file exists
                if not os.path.isfile(fname):
                    print(f"Error: File '{fname}' does not exist.")
                    exit(1)
                auxfile_list.append(fname)
                extension_auxfile_list.append(item[item.index("[") + 1 : item.index("]")])
            else:
                auxfile_list.append(item.strip())
                extension_auxfile_list.append("0")

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
        input_fits=input_fits,
        extension=extension,
        auxfile_list=auxfile_list,
        extension_auxfile_list=extension_auxfile_list,
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
