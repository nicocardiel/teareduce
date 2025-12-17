#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Define the CosmicRayCleanerApp class."""

import tkinter as tk
from tkinter import filedialog
from tkinter import font as tkfont
from tkinter import messagebox
from tkinter import simpledialog
import sys

from astropy.io import fits

try:
    from maskfill import maskfill
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "The 'teareduce.cleanest' module requires the 'ccdproc' and 'maskfill' packages. "
        "Please install teareduce with the 'cleanest' extra dependencies: "
        "`pip install teareduce[cleanest]`."
    ) from e
import matplotlib.pyplot as plt
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy import ndimage
import numpy as np
import os
from pathlib import Path
from rich import print

from .askextension import ask_extension_input_image
from .centerchildparent import center_on_parent
from .definitions import lacosmic_default_dict
from .definitions import DEFAULT_NPOINTS_INTERP
from .definitions import DEFAULT_DEGREE_INTERP
from .definitions import DEFAULT_MASKFILL_SIZE
from .definitions import DEFAULT_MASKFILL_OPERATOR
from .definitions import DEFAULT_MASKFILL_SMOOTH
from .definitions import DEFAULT_MASKFILL_VERBOSE
from .definitions import MAX_PIXEL_DISTANCE_TO_CR
from .definitions import DEFAULT_TK_WINDOW_SIZE_X
from .definitions import DEFAULT_TK_WINDOW_SIZE_Y
from .definitions import DEFAULT_FONT_FAMILY
from .definitions import DEFAULT_FONT_SIZE
from .dilatemask import dilatemask
from .find_closest_true import find_closest_true
from .interpolation_a import interpolation_a
from .interpolation_x import interpolation_x
from .interpolation_y import interpolation_y
from .interpolationeditor import InterpolationEditor
from .imagedisplay import ImageDisplay
from .lacosmicpad import lacosmicpad
from .mergemasks import merge_peak_tail_masks
from .modalprogressbar import ModalProgressBar
from .parametereditor import ParameterEditor
from .reviewcosmicray import ReviewCosmicRay
from .trackedbutton import TrackedTkButton

from ..imshow import imshow
from ..sliceregion import SliceRegion2D
from ..version import VERSION
from ..zscale import zscale

import matplotlib

matplotlib.use("TkAgg")


class CosmicRayCleanerApp(ImageDisplay):
    """Main application class for cosmic ray cleaning."""

    def __init__(
        self,
        root,
        input_fits,
        extension="0",
        auxfile=None,
        extension_auxfile="0",
        fontfamily=DEFAULT_FONT_FAMILY,
        fontsize=DEFAULT_FONT_SIZE,
        width=DEFAULT_TK_WINDOW_SIZE_X,
        height=DEFAULT_TK_WINDOW_SIZE_Y,
        verbose=False,
    ):
        """
        Initialize the application.

        Parameters
        ----------
        root : tk.Tk
            The main Tkinter window.
        input_fits : str
            Path to the FITS file to be cleaned.
        extension : str, optional
            FITS extension to use (default is "0").
        auxfile : str, optional
            Path to an auxiliary FITS file (default is None).
        extension_auxfile : str, optional
            FITS extension for auxiliary file (default is "0").
        fontfamily : str, optional
            Font family for the GUI (default is "Helvetica").
        fontsize : int, optional
            Font size for the GUI (default is 14).
        width : int, optional
            Width of the GUI window in pixels (default is 800).
        height : int, optional
            Height of the GUI window in pixels (default is 600).
        verbose : bool, optional
            Enable verbose output (default is False).

        Methods
        -------
        process_detected_cr(dilation)
            Process the detected cosmic ray mask.
        load_detected_cr_from_file()
            Load detected cosmic ray mask from a FITS file.
        load_fits_file()
            Load the FITS file and auxiliary file (if provided).
        save_fits_file()
            Save the cleaned data to a FITS file.
        create_widgets()
            Create the GUI widgets.
        run_lacosmic()
            Run the L.A.Cosmic algorithm.
        toggle_cr_overlay()
            Toggle the overlay of cosmic ray pixels on the image.
        update_cr_overlay()
            Update the overlay of cosmic ray pixels on the image.
        apply_lacosmic()
            Apply the L.A.Cosmic algorithm to the data.
        review_detected_cr()
            Examine detected cosmic rays.
        stop_app()
            Stop the application.
        on_key(event)
            Handle key press events.
        on_click(event)
            Handle mouse click events.

        Attributes
        ----------
        root : tk.Tk
            The main Tkinter window.
        fontfamily : str
            Font family for the GUI.
        fontsize : int
            Font size for the GUI.
        default_font : tkfont.Font
            The default font used in the GUI.
        width : int
            Width of the GUI window in pixels.
        height : int
            Height of the GUI window in pixels.
        verbose : bool
            Enable verbose output.
        lacosmic_params : dict
            Dictionary of L.A.Cosmic parameters.
        input_fits : str
            Path to the FITS file to be cleaned.
        extension : int
            FITS extension to use.
        data : np.ndarray
            The image data from the FITS file.
        auxfile : str
            Path to an auxiliary FITS file.
        extension_auxfile : int
            FITS extension for auxiliary file.
        auxdata : np.ndarray
            The image data from the auxiliary FITS file.
        overplot_cr_pixels : bool
            Flag to indicate whether to overlay cosmic ray pixels.
        mask_crfound : np.ndarray
            Boolean mask of detected cosmic ray pixels.
        last_xmin : int
            Last used minimum x-coordinate for region selection.
            From 1 to NAXIS1.
        last_xmax : int
            Last used maximum x-coordinate for region selection.
            From 1 to NAXIS1.
        last_ymin : int
            Last used minimum y-coordinate for region selection.
            From 1 to NAXIS2.
        last_ymax : int
            Last used maximum y-coordinate for region selection.
            From 1 to NAXIS2.
        last_inbkg : str or None
            Last used input background image FITS file.
        last_extnum_inbkg : int or None
            Last used FITS extension number for the input background image.
        last_invar : str or None
            Last used input variance image FITS file.
        last_extnum_invar : int or None
            Last used FITS extension number for the input variance image.
        last_npoints : int
            Last used number of points for interpolation.
        last_degree : int
            Last used degree for interpolation.
        last_maskfill_size : int
            Last used size parameter for maskfill.
        last_maskfill_operator : str
            Last used operator parameter for maskfill.
        last_maskfill_smooth : bool
            Last used smooth parameter for maskfill.
        last_maskfill_verbose : bool
            Last used verbose parameter for maskfill.
        cleandata_lacosmic : np.ndarray
            The cleaned data returned from L.A.Cosmic.
        cr_labels : np.ndarray
            Labeled cosmic ray features.
        num_features : int
            Number of detected cosmic ray features.
        working_in_review_window : bool
            Flag to indicate if the review window is active.
        """
        self.root = root
        # self.root.geometry("800x700+50+0")  # This does not work in Fedora
        self.width = width
        self.height = height
        self.verbose = verbose
        self.root.minsize(self.width, self.height)
        self.root.update_idletasks()
        self.root.title(f"Cosmic Ray Cleaner (TEA version {VERSION})")
        self.fontfamily = fontfamily
        self.fontsize = fontsize
        self.default_font = tkfont.nametofont("TkDefaultFont")
        self.default_font.configure(
            family=fontfamily, size=fontsize, weight="normal", slant="roman", underline=0, overstrike=0
        )
        self.lacosmic_params = lacosmic_default_dict.copy()
        self.lacosmic_params["run1_verbose"]["value"] = self.verbose
        self.lacosmic_params["run2_verbose"]["value"] = self.verbose
        self.input_fits = input_fits
        self.extension = extension
        self.data = None
        self.auxfile = auxfile
        self.extension_auxfile = extension_auxfile
        self.auxdata = None
        self.overplot_cr_pixels = True
        self.mask_crfound = None
        self.load_fits_file()
        self.last_xmin = 1
        self.last_xmax = self.data.shape[1]
        self.last_ymin = 1
        self.last_ymax = self.data.shape[0]
        self.last_inbkg = None
        self.last_extnum_inbkg = None
        self.inbkg_data = None
        self.last_invar = None
        self.last_extnum_invar = None
        self.invar_data = None
        self.last_npoints = DEFAULT_NPOINTS_INTERP
        self.last_degree = DEFAULT_DEGREE_INTERP
        self.last_maskfill_size = DEFAULT_MASKFILL_SIZE
        self.last_maskfill_operator = DEFAULT_MASKFILL_OPERATOR
        self.last_maskfill_smooth = DEFAULT_MASKFILL_SMOOTH
        self.last_maskfill_verbose = DEFAULT_MASKFILL_VERBOSE
        self.create_widgets()
        self.cleandata_lacosmic = None
        self.cr_labels = None
        self.num_features = 0
        self.working_in_review_window = False

    def process_detected_cr(self, dilation):
        """Process the detected cosmic ray mask.

        Parameters
        ----------
        dilation : int
            Number of pixels to dilate the cosmic ray mask.
        """
        # Process the mask: dilation and labeling
        if np.any(self.mask_crfound):
            num_cr_pixels_before_dilation = np.sum(self.mask_crfound)
            if dilation > 0:
                # Dilate the mask by the specified number of pixels
                self.mask_crfound = dilatemask(mask=self.mask_crfound, iterations=dilation, connectivity=1)
                num_cr_pixels_after_dilation = np.sum(self.mask_crfound)
                sdum = str(num_cr_pixels_after_dilation)
            else:
                sdum = str(num_cr_pixels_before_dilation)
            if dilation > 0:
                print(
                    "Number of cosmic ray pixels before dilation.......: "
                    f"{num_cr_pixels_before_dilation:>{len(sdum)}}"
                )
                print(
                    f"Number of cosmic ray pixels after dilation........: "
                    f"{num_cr_pixels_after_dilation:>{len(sdum)}}"
                )
            # Label connected components in the mask; note that by default,
            # structure is a cross [0,1,0;1,1,1;0,1,0], but we want to consider
            # diagonal connections too, so we define a 3x3 square.
            structure = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
            self.cr_labels, self.num_features = ndimage.label(self.mask_crfound, structure=structure)
            print(f"Number of cosmic ray features (grouped pixels)....: {self.num_features:>{len(sdum)}}")
            self.replace_detected_cr_button.config(state=tk.NORMAL)
            self.review_detected_cr_button.config(state=tk.NORMAL)
            self.update_cr_overlay()
            self.use_cursor = True
            self.use_cursor_button.config(text="[c]ursor: ON ")
        else:
            print("No cosmic ray pixels detected!")
            self.cr_labels = None
            self.num_features = 0
            self.replace_detected_cr_button.config(state=tk.DISABLED)
            self.review_detected_cr_button.config(state=tk.DISABLED)

    def load_detected_cr_from_file(self):
        """Load detected cosmic ray mask from a FITS file."""
        if np.any(self.mask_crfound):
            overwrite = messagebox.askyesno(
                "Overwrite Cosmic Ray Mask",
                "A cosmic ray mask is already defined.\n\nDo you want to overwrite it?",
            )
            if not overwrite:
                return
        crmask_file = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            title="Select FITS file with cosmic ray mask",
            filetypes=[("FITS files", "*.fits"), ("All files", "*.*")],
        )
        if crmask_file:
            print(f"Selected input FITS file: {crmask_file}")
            extension = ask_extension_input_image(crmask_file, self.data.shape)
            try:
                with fits.open(crmask_file, mode="readonly") as hdul:
                    if isinstance(extension, int):
                        if extension < 0 or extension >= len(hdul):
                            raise IndexError(f"Extension index {extension} out of range.")
                    else:
                        if extension not in hdul:
                            raise KeyError(f"Extension name '{extension}' not found.")
                    if hdul[extension].header["BITPIX"] not in [8, 16]:
                        raise ValueError("Cosmic ray mask must be of integer type (BITPIX=8 or 16).")
                    mask_crfound_loaded = hdul[extension].data.astype(bool)
                    if mask_crfound_loaded.shape != self.data.shape:
                        print(f"data shape...: {self.data.shape}")
                        print(f"mask shape...: {mask_crfound_loaded.shape}")
                        raise ValueError("Cosmic ray mask has different shape.")
                    self.mask_crfound = mask_crfound_loaded
                    print(f"Loaded cosmic ray mask from {crmask_file}")
                    dilation = simpledialog.askinteger(
                        "Dilation", "Enter Dilation (min=0):", initialvalue=0, minvalue=0
                    )
                    self.process_detected_cr(dilation=dilation)
                    # self.lacosmic_params["dilation"]["value"] = dilation
                    self.cleandata_lacosmic = None  # Invalidate previous L.A.Cosmic cleaned data
            except Exception as e:
                messagebox.showerror("Error", f"Error loading cosmic ray mask: {e}")

    def load_fits_file(self):
        """Load the FITS file and auxiliary file (if provided).

        Returns
        -------
        None

        Notes
        -----
        This method loads the FITS file specified by `self.input_fits` and
        reads the data from the specified extension. If an auxiliary file is
        provided, it also loads the auxiliary data from the specified extension.
        The loaded data is stored in `self.data` and `self.auxdata` attributes.
        """
        # check if extension is compatible with an integer
        try:
            extnum = int(self.extension)
            self.extension = extnum
        except ValueError:
            # Keep as string (delaying checking until opening the file)
            self.extension = self.extension.upper()  # Convert to uppercase
        try:
            with fits.open(self.input_fits, mode="readonly") as hdul:
                if isinstance(self.extension, int):
                    if self.extension < 0 or self.extension >= len(hdul):
                        raise IndexError(f"Extension index {self.extension} out of range.")
                else:
                    if self.extension not in hdul:
                        raise KeyError(f"Extension name '{self.extension}' not found.")
                print(f"Reading file [bold green]{self.input_fits}[/bold green], extension {self.extension}")
                self.data = hdul[self.extension].data
                if "CRMASK" in hdul:
                    self.mask_fixed = hdul["CRMASK"].data.astype(bool)
                else:
                    self.mask_fixed = np.zeros(self.data.shape, dtype=bool)
        except Exception as e:
            print(f"Error loading FITS file: {e}")
            sys.exit(1)
        self.mask_crfound = np.zeros(self.data.shape, dtype=bool)
        naxis2, naxis1 = self.data.shape
        self.region = SliceRegion2D(f"[1:{naxis1}, 1:{naxis2}]", mode="fits").python
        # Read auxiliary file if provided
        if self.auxfile is not None:
            # check if extension_auxfile is compatible with an integer
            try:
                extnum_aux = int(self.extension_auxfile)
                self.extension_auxfile = extnum_aux
            except ValueError:
                # Keep as string (delaying checking until opening the file)
                self.extension_auxfile = self.extension_auxfile.upper()  # Convert to uppercase
            try:
                with fits.open(self.auxfile, mode="readonly") as hdul_aux:
                    if isinstance(self.extension_auxfile, int):
                        if self.extension_auxfile < 0 or self.extension_auxfile >= len(hdul_aux):
                            raise IndexError(f"Extension index {self.extension_auxfile} out of range.")
                    else:
                        if self.extension_auxfile not in hdul_aux:
                            raise KeyError(f"Extension name '{self.extension_auxfile}' not found.")
                    print(
                        f"Reading auxiliary file [bold green]{self.auxfile}[/bold green], extension {self.extension_auxfile}"
                    )
                    self.auxdata = hdul_aux[self.extension_auxfile].data
                    if self.auxdata.shape != self.data.shape:
                        print(f"data shape...: {self.data.shape}")
                        print(f"auxdata shape: {self.auxdata.shape}")
                        raise ValueError("Auxiliary file has different shape.")
            except Exception as e:
                sys.exit(f"Error loading auxiliary FITS file: {e}")

    def load_auxdata_from_file(self):
        """Load auxiliary data from a FITS file."""
        if self.auxfile is not None:
            overwrite = messagebox.askyesno(
                "Overwrite Auxiliary Data",
                f"An auxiliary file is already loaded:\n\n{self.auxfile}\n\n" "Do you want to overwrite it?",
            )
            if not overwrite:
                return
        auxfile = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            title="Select auxiliary FITS file",
            filetypes=[("FITS files", "*.fits"), ("All files", "*.*")],
        )
        if auxfile:
            extension = ask_extension_input_image(auxfile, self.data.shape)
            try:
                with fits.open(auxfile, mode="readonly") as hdul:
                    if isinstance(extension, int):
                        if extension < 0 or extension >= len(hdul):
                            raise IndexError(f"Extension index {extension} out of range.")
                    else:
                        if extension not in hdul:
                            raise KeyError(f"Extension name '{extension}' not found.")
                    auxdata_loaded = hdul[extension].data
                    if auxdata_loaded.shape != self.data.shape:
                        print(f"data shape...: {self.data.shape}")
                        print(f"auxdata shape: {auxdata_loaded.shape}")
                        raise ValueError("Auxiliary file has different shape.")
                    self.auxfile = auxfile
                    self.auxdata = auxdata_loaded
                    self.extension_auxfile = extension
                    print(f"Loaded auxiliary data from {auxfile}")
                    self.toggle_auxdata_button.config(state=tk.NORMAL)
            except Exception as e:
                print(f"Error loading auxiliary FITS file: {e}")

    def save_fits_file(self):
        """Save the cleaned FITS file.

        This method prompts the user to select a location and filename to
        save the cleaned FITS file. It writes the cleaned data and
        the cosmic ray mask to the specified FITS file.

        If the initial file contains a 'CRMASK' extension, it updates
        that extension with the new mask. Otherwise, it creates a new
        'CRMASK' extension to store the mask.

        Returns
        -------
        None

        Notes
        -----
        After successfully saving the cleaned FITS file, the chosen output
        filename is stored in `self.input_fits`, and the save button is disabled
        to prevent multiple saves without further modifications.
        """
        base, ext = os.path.splitext(self.input_fits)
        suggested_name = f"{base}_cleaned"
        output_fits = filedialog.asksaveasfilename(
            initialdir=os.getcwd(),
            title="Save cleaned FITS file",
            defaultextension=".fits",
            filetypes=[("FITS files", "*.fits"), ("All files", "*.*")],
            initialfile=suggested_name,
        )
        try:
            with fits.open(self.input_fits, mode="readonly") as hdul:
                hdul[self.extension].data = self.data
                if "CRMASK" in hdul:
                    hdul["CRMASK"].data = self.mask_fixed.astype(np.uint8)
                else:
                    crmask_hdu = fits.ImageHDU(self.mask_fixed.astype(np.uint8), name="CRMASK")
                    hdul.append(crmask_hdu)
                hdul.writeto(output_fits, overwrite=True)
            print(f"Cleaned data saved to {output_fits}")
            self.ax.set_title(os.path.basename(output_fits))
            self.canvas.draw_idle()
            self.input_fits = os.path.basename(output_fits)
            self.save_button.config(state=tk.DISABLED)
        except Exception as e:
            print(f"Error saving FITS file: {e}")

    def create_widgets(self):
        """Create the GUI widgets.

        Returns
        -------
        None

        Notes
        -----
        This method sets up the GUI layout, including buttons for running
        L.A.Cosmic, toggling cosmic ray overlay, applying cleaning methods,
        examining detected cosmic rays, saving the cleaned FITS file, and
        stopping the application. It also initializes the matplotlib figure
        and canvas for image display, along with the toolbar for navigation.
        The relevant attributes are stored in the instance for later use.
        """
        # Define instance of TrackedTkButton, that facilitates to show help information
        # for each button displayed in the current application window.
        tkbutton = TrackedTkButton(self.root)

        # Row 1 of buttons
        self.button_frame1 = tk.Frame(self.root)
        self.button_frame1.pack(pady=5)
        self.run_lacosmic_button = tkbutton.new(
            self.button_frame1,
            text="Run L.A.Cosmic",
            command=self.run_lacosmic,
            help_text="Run the L.A.Cosmic algorithm to detect cosmic rays in the image.",
        )
        self.run_lacosmic_button.pack(side=tk.LEFT, padx=5)
        self.load_auxdata_button = tkbutton.new(
            self.button_frame1,
            text="Load auxdata",
            command=self.load_auxdata_from_file,
            help_text="Load an auxiliary FITS file for display.",
        )
        self.load_auxdata_button.pack(side=tk.LEFT, padx=5)
        self.load_detected_cr_button = tkbutton.new(
            self.button_frame1,
            text="Load CR mask",
            command=self.load_detected_cr_from_file,
            help_text="Load a previously saved cosmic ray mask from a FITS file.",
        )
        self.load_detected_cr_button.pack(side=tk.LEFT, padx=5)
        self.replace_detected_cr_button = tkbutton.new(
            self.button_frame1,
            text="Replace detected CRs",
            command=self.apply_lacosmic,
            help_text="Apply the cleaning to the detected cosmic rays.",
        )
        self.replace_detected_cr_button.pack(side=tk.LEFT, padx=5)
        self.replace_detected_cr_button.config(state=tk.DISABLED)  # Initially disabled
        self.review_detected_cr_button = tkbutton.new(
            self.button_frame1,
            text="Review detected CRs",
            command=lambda: self.review_detected_cr(1),
            help_text="Review the detected cosmic rays.",
        )
        self.review_detected_cr_button.pack(side=tk.LEFT, padx=5)
        self.review_detected_cr_button.config(state=tk.DISABLED)  # Initially disabled

        # Row 2 of buttons
        self.button_frame2 = tk.Frame(self.root)
        self.button_frame2.pack(pady=5)
        self.use_cursor = False
        self.use_cursor_button = tkbutton.new(
            self.button_frame2,
            text="[c]ursor: OFF",
            command=self.set_cursor_onoff,
            help_text="Toggle the cursor ON or OFF (to select CR with mouse).",
            alttext="[c]ursor: ??",
        )
        self.use_cursor_button.pack(side=tk.LEFT, padx=5)
        self.toggle_auxdata_button = tkbutton.new(
            self.button_frame2,
            text="[t]oggle data",
            command=self.toggle_auxdata,
            help_text="Toggle the display of auxiliary data.",
        )
        self.toggle_auxdata_button.pack(side=tk.LEFT, padx=5)
        if self.auxdata is None:
            self.toggle_auxdata_button.config(state=tk.DISABLED)
        else:
            self.toggle_auxdata_button.config(state=tk.NORMAL)
        self.image_aspect = "equal"
        self.toggle_aspect_button = tkbutton.new(
            self.button_frame2,
            text=f"[a]spect: {self.image_aspect}",
            command=self.toggle_aspect,
            help_text="Toggle the image aspect ratio.",
        )
        self.toggle_aspect_button.pack(side=tk.LEFT, padx=5)
        self.save_button = tkbutton.new(
            self.button_frame2,
            text="Save cleaned FITS",
            command=self.save_fits_file,
            help_text="Save the cleaned FITS file.",
        )
        self.save_button.pack(side=tk.LEFT, padx=5)
        self.save_button.config(state=tk.DISABLED)  # Initially disabled
        self.stop_button = tkbutton.new(
            self.button_frame2, text="Stop program", command=self.stop_app, help_text="Stop the application."
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Row 3 of buttons
        self.button_frame3 = tk.Frame(self.root)
        self.button_frame3.pack(pady=5)
        vmin, vmax = zscale(self.data)
        self.vmin_button = tkbutton.new(
            self.button_frame3,
            text=f"vmin: {vmin:.2f}",
            command=self.set_vmin,
            help_text="Set the minimum value for the display scale.",
            alttext="vmin: ??",
        )
        self.vmin_button.pack(side=tk.LEFT, padx=5)
        self.vmax_button = tkbutton.new(
            self.button_frame3,
            text=f"vmax: {vmax:.2f}",
            command=self.set_vmax,
            help_text="Set the maximum value for the display scale.",
            alttext="vmax: ??",
        )
        self.vmax_button.pack(side=tk.LEFT, padx=5)
        self.set_minmax_button = tkbutton.new(
            self.button_frame3,
            text="minmax [,]",
            command=self.set_minmax,
            help_text="Set the minimum and maximum values for the display scale.",
        )
        self.set_minmax_button.pack(side=tk.LEFT, padx=5)
        self.set_zscale_button = tkbutton.new(
            self.button_frame3,
            text="zscale [/]",
            command=self.set_zscale,
            help_text="Set the display scale using zscale.",
        )
        self.set_zscale_button.pack(side=tk.LEFT, padx=5)
        if self.overplot_cr_pixels:
            self.overplot_cr_button = tkbutton.new(
                self.button_frame3,
                text="CR overlay: ON ",
                command=self.toggle_cr_overlay,
                help_text="Toggle the cosmic ray overlay ON or OFF.",
                alttext="CR overlay: ??",
            )
        else:
            self.overplot_cr_button = tkbutton.new(
                self.button_frame3,
                text="CR overlay: OFF",
                command=self.toggle_cr_overlay,
                help_text="Toggle the cosmic ray overlay ON or OFF.",
                alttext="CR overlay: ??",
            )
        self.overplot_cr_button.pack(side=tk.LEFT, padx=5)
        self.help_button = tkbutton.new(
            self.button_frame3,
            text="Help",
            command=tkbutton.show_help,
            help_text="Show help information for all buttons.",
        )
        self.help_button.pack(side=tk.LEFT, padx=5)

        # Figure
        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        fig_dpi = 100
        image_ratio = 480 / 640  # Default image ratio
        fig_width_inches = self.width / fig_dpi
        fig_height_inches = self.height * image_ratio / fig_dpi
        self.fig, self.ax = plt.subplots(figsize=(fig_width_inches, fig_height_inches), dpi=fig_dpi)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.config(width=self.width, height=self.height * image_ratio)
        canvas_widget.pack(expand=True)
        # The next two instructions prevent a segmentation fault when pressing "q"
        self.canvas.mpl_disconnect(self.canvas.mpl_connect("key_press_event", key_press_handler))
        self.canvas.mpl_connect("key_press_event", self.on_key)
        self.canvas.mpl_connect("button_press_event", self.on_click)
        canvas_widget = self.canvas.get_tk_widget()

        # Matplotlib toolbar
        self.toolbar_frame = tk.Frame(self.root)
        self.toolbar_frame.pack(fill=tk.X, expand=False, pady=5)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()

        # update the image display
        xlabel = "X pixel (from 1 to NAXIS1)"
        ylabel = "Y pixel (from 1 to NAXIS2)"
        extent = [0.5, self.data.shape[1] + 0.5, 0.5, self.data.shape[0] + 0.5]
        self.image_aspect = "equal"
        self.displaying_auxdata = False
        self.image, _, _ = imshow(
            fig=self.fig,
            ax=self.ax,
            data=self.data,
            vmin=vmin,
            vmax=vmax,
            title=f"data: {os.path.basename(self.input_fits)}",
            xlabel=xlabel,
            ylabel=ylabel,
            extent=extent,
            aspect=self.image_aspect,
        )
        self.fig.tight_layout()

    def set_cursor_onoff(self):
        """Toggle cursor selection mode on or off."""
        if not self.use_cursor:
            self.use_cursor = True
            self.use_cursor_button.config(text="[c]ursor: ON ")
        else:
            self.use_cursor = False
            self.use_cursor_button.config(text="[c]ursor: OFF")

    def toggle_auxdata(self):
        """Toggle between main data and auxiliary data for display."""
        if self.displaying_auxdata:
            # Switch to main data
            vmin = self.get_vmin()
            vmax = self.get_vmax()
            self.image.set_data(self.data)
            self.image.set_clim(vmin=vmin, vmax=vmax)
            self.displaying_auxdata = False
            self.ax.set_title(f"data: {os.path.basename(self.input_fits)}")
        else:
            # Switch to auxiliary data
            vmin = self.get_vmin()
            vmax = self.get_vmax()
            self.image.set_data(self.auxdata)
            self.image.set_clim(vmin=vmin, vmax=vmax)
            self.displaying_auxdata = True
            self.ax.set_title(f"auxdata: {os.path.basename(self.auxfile)}")
        self.canvas.draw_idle()

    def toggle_aspect(self):
        """Toggle the aspect ratio of the image display."""
        if self.image_aspect == "equal":
            self.image_aspect = "auto"
        else:
            self.image_aspect = "equal"
        print(f"Setting image aspect to: {self.image_aspect}")
        self.toggle_aspect_button.config(text=f"[a]spect: {self.image_aspect}")
        self.ax.set_aspect(self.image_aspect)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def run_lacosmic(self):
        """Run L.A.Cosmic to detect cosmic rays."""
        if np.any(self.mask_crfound):
            overwrite = messagebox.askyesno(
                "Overwrite Cosmic Ray Mask",
                "A cosmic ray mask is already defined.\n\nDo you want to overwrite it?",
            )
            if not overwrite:
                return
        self.run_lacosmic_button.config(state=tk.DISABLED)
        # Define parameters for L.A.Cosmic from default dictionary
        editor_window = tk.Toplevel(self.root)
        center_on_parent(child=editor_window, parent=self.root)
        editor = ParameterEditor(
            root=editor_window,
            param_dict=self.lacosmic_params,
            window_title="Cosmic Ray Mask Generation Parameters",
            xmin=self.last_xmin,
            xmax=self.last_xmax,
            ymin=self.last_ymin,
            ymax=self.last_ymax,
            imgshape=self.data.shape,
            inbkg=self.last_inbkg,
            extnum_inbkg=self.last_extnum_inbkg,
            invar=self.last_invar,
            extnum_invar=self.last_extnum_invar,
        )
        # Make it modal (blocks interaction with main window)
        editor_window.transient(self.root)
        editor_window.grab_set()
        # Wait for the editor window to close
        self.root.wait_window(editor_window)
        # Get the result after window closes
        updated_params = editor.get_result()
        if updated_params is not None:
            # Update last used region values
            self.last_xmin = updated_params["xmin"]["value"]
            self.last_xmax = updated_params["xmax"]["value"]
            self.last_ymin = updated_params["ymin"]["value"]
            self.last_ymax = updated_params["ymax"]["value"]
            self.last_inbkg = updated_params["inbkg"]["value"]
            self.last_extnum_inbkg = updated_params["extnum_inbkg"]["value"]
            self.last_invar = updated_params["invar"]["value"]
            self.last_extnum_invar = updated_params["extnum_invar"]["value"]
            if self.last_inbkg is not None:
                with fits.open(self.last_inbkg, mode="readonly") as hdul_inbkg:
                    if self.last_extnum_inbkg < 0 or self.last_extnum_inbkg >= len(hdul_inbkg):
                        raise IndexError(f"Extension index {self.last_extnum_inbkg} out of range.")
                    self.inbkg_data = hdul_inbkg[self.last_extnum_inbkg].data.astype(np.float32)
            else:
                self.inbkg_data = None
            if self.last_invar is not None:
                with fits.open(self.last_invar, mode="readonly") as hdul_invar:
                    if self.last_extnum_invar < 0 or self.last_extnum_invar >= len(hdul_invar):
                        raise IndexError(f"Extension index {self.last_extnum_invar} out of range.")
                    self.invar_data = hdul_invar[self.last_extnum_invar].data.astype(np.float32)
            else:
                self.invar_data = None
            usefulregion = SliceRegion2D(
                f"[{self.last_xmin}:{self.last_xmax},{self.last_ymin}:{self.last_ymax}]", mode="fits"
            ).python
            usefulmask = np.zeros_like(self.data)
            usefulmask[usefulregion] = 1.0
            # Update parameter dictionary with new values
            self.lacosmic_params = updated_params
            if self.verbose:
                print("Parameters updated:")
                for key, info in self.lacosmic_params.items():
                    print(f"  {key}: {info['value']}")
            if self.lacosmic_params["nruns"]["value"] not in [1, 2]:
                raise ValueError("nruns must be 1 or 2")
            # Execute L.A.Cosmic with updated parameters
            print("[bold green]Executing L.A.Cosmic (run 1)...[/bold green]")
            borderpadd = updated_params["borderpadd"]["value"]
            cleandata_lacosmic, mask_crfound = lacosmicpad(
                pad_width=borderpadd,
                show_arguments=self.verbose,
                ccd=self.data,
                gain_apply=True,  # Always apply gain
                sigclip=self.lacosmic_params["run1_sigclip"]["value"],
                sigfrac=self.lacosmic_params["run1_sigfrac"]["value"],
                objlim=self.lacosmic_params["run1_objlim"]["value"],
                gain=self.lacosmic_params["run1_gain"]["value"],
                readnoise=self.lacosmic_params["run1_readnoise"]["value"],
                satlevel=self.lacosmic_params["run1_satlevel"]["value"],
                niter=self.lacosmic_params["run1_niter"]["value"],
                sepmed=self.lacosmic_params["run1_sepmed"]["value"],
                cleantype=self.lacosmic_params["run1_cleantype"]["value"],
                fsmode=self.lacosmic_params["run1_fsmode"]["value"],
                psfmodel=self.lacosmic_params["run1_psfmodel"]["value"],
                psffwhm_x=self.lacosmic_params["run1_psffwhm_x"]["value"],
                psffwhm_y=self.lacosmic_params["run1_psffwhm_y"]["value"],
                psfsize=self.lacosmic_params["run1_psfsize"]["value"],
                psfbeta=self.lacosmic_params["run1_psfbeta"]["value"],
                verbose=self.lacosmic_params["run1_verbose"]["value"],
                inbkg=self.inbkg_data,
                invar=self.invar_data,
            )
            # Apply usefulmask to consider only selected region
            cleandata_lacosmic *= usefulmask
            mask_crfound = mask_crfound & (usefulmask.astype(bool))
            # Second execution if nruns == 2
            if self.lacosmic_params["nruns"]["value"] == 2:
                print("[bold green]Executing L.A.Cosmic (run 2)...[/bold green]")
                cleandata_lacosmic2, mask_crfound2 = lacosmicpad(
                    pad_width=borderpadd,
                    show_arguments=self.verbose,
                    ccd=self.data,
                    gain_apply=True,  # Always apply gain
                    sigclip=self.lacosmic_params["run2_sigclip"]["value"],
                    sigfrac=self.lacosmic_params["run2_sigfrac"]["value"],
                    objlim=self.lacosmic_params["run2_objlim"]["value"],
                    gain=self.lacosmic_params["run2_gain"]["value"],
                    readnoise=self.lacosmic_params["run2_readnoise"]["value"],
                    satlevel=self.lacosmic_params["run2_satlevel"]["value"],
                    niter=self.lacosmic_params["run2_niter"]["value"],
                    sepmed=self.lacosmic_params["run2_sepmed"]["value"],
                    cleantype=self.lacosmic_params["run2_cleantype"]["value"],
                    fsmode=self.lacosmic_params["run2_fsmode"]["value"],
                    psfmodel=self.lacosmic_params["run2_psfmodel"]["value"],
                    psffwhm_x=self.lacosmic_params["run2_psffwhm_x"]["value"],
                    psffwhm_y=self.lacosmic_params["run2_psffwhm_y"]["value"],
                    psfsize=self.lacosmic_params["run2_psfsize"]["value"],
                    psfbeta=self.lacosmic_params["run2_psfbeta"]["value"],
                    verbose=self.lacosmic_params["run2_verbose"]["value"],
                    inbkg=self.inbkg_data,
                    invar=self.invar_data,
                )
                # Apply usefulmask to consider only selected region
                cleandata_lacosmic2 *= usefulmask
                mask_crfound2 = mask_crfound2 & (usefulmask.astype(bool))
                # Combine results from both runs
                if np.any(mask_crfound):
                    mask_crfound = merge_peak_tail_masks(mask_crfound, mask_crfound2, verbose=True)
                # Use the cleandata from the second run
                cleandata_lacosmic = cleandata_lacosmic2
            # Select the image region to process
            self.cleandata_lacosmic = self.data.copy()
            self.cleandata_lacosmic[usefulregion] = cleandata_lacosmic[usefulregion]
            self.mask_crfound = np.zeros_like(self.data, dtype=bool)
            self.mask_crfound[usefulregion] = mask_crfound[usefulregion]
            # Process the mask: dilation and labeling
            self.process_detected_cr(dilation=self.lacosmic_params["dilation"]["value"])
        else:
            print("Parameter editing cancelled. L.A.Cosmic detection skipped!")
        self.run_lacosmic_button.config(state=tk.NORMAL)

    def toggle_cr_overlay(self):
        """Toggle the overlay of cosmic ray pixels on the image."""
        self.overplot_cr_pixels = not self.overplot_cr_pixels
        if self.overplot_cr_pixels:
            self.overplot_cr_button.config(text="CR overlay: ON ")
        else:
            self.overplot_cr_button.config(text="CR overlay: OFF")
        self.update_cr_overlay()

    def update_cr_overlay(self):
        """Update the overlay of cosmic ray pixels on the image."""
        if self.overplot_cr_pixels:
            # Remove previous CR pixel overlay (if any)
            if hasattr(self, "scatter_cr"):
                self.scatter_cr.remove()
                del self.scatter_cr
            # Overlay CR pixels in red
            if np.any(self.mask_crfound):
                y_indices, x_indices = np.where(self.mask_crfound)
                self.scatter_cr = self.ax.scatter(x_indices + 1, y_indices + 1, s=1, c="red", marker="o")
        else:
            # Remove CR pixel overlay
            if hasattr(self, "scatter_cr"):
                self.scatter_cr.remove()
                del self.scatter_cr
        self.canvas.draw_idle()

    def apply_lacosmic(self):
        """Apply the selected cleaning method to the detected cosmic rays."""
        if np.any(self.mask_crfound):
            # recalculate labels and number of features
            structure = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
            self.cr_labels, self.num_features = ndimage.label(self.mask_crfound, structure=structure)
            # Define parameters for L.A.Cosmic from default dictionary
            editor_window = tk.Toplevel(self.root)
            center_on_parent(child=editor_window, parent=self.root)
            editor = InterpolationEditor(
                root=editor_window,
                last_dilation=self.lacosmic_params["dilation"]["value"],
                last_npoints=self.last_npoints,
                last_degree=self.last_degree,
                last_maskfill_size=self.last_maskfill_size,
                last_maskfill_operator=self.last_maskfill_operator,
                last_maskfill_smooth=self.last_maskfill_smooth,
                last_maskfill_verbose=self.last_maskfill_verbose,
                auxdata=self.auxdata,
                cleandata_lacosmic=self.cleandata_lacosmic,
                xmin=self.last_xmin,
                xmax=self.last_xmax,
                ymin=self.last_ymin,
                ymax=self.last_ymax,
                imgshape=self.data.shape,
            )
            # Make it modal (blocks interaction with main window)
            editor_window.transient(self.root)
            editor_window.grab_set()
            # Wait for the editor window to close
            self.root.wait_window(editor_window)
            # Get the result after window closes
            cleaning_method = editor.cleaning_method
            if cleaning_method is None:
                print("Interpolation method selection cancelled. No cleaning applied!")
                return
            # Update last employed parameters
            self.last_npoints = editor.npoints
            self.last_degree = editor.degree
            self.last_maskfill_size = editor.maskfill_size
            self.last_maskfill_operator = editor.maskfill_operator
            self.last_maskfill_smooth = editor.maskfill_smooth
            self.last_maskfill_verbose = editor.maskfill_verbose
            cleaning_region = SliceRegion2D(
                f"[{editor.xmin}:{editor.xmax},{editor.ymin}:{editor.ymax}]", mode="fits"
            ).python
            print(
                "Applying cleaning method to region "
                f"x=[{editor.xmin},{editor.xmax}], y=[{editor.ymin},{editor.ymax}]"
            )
            mask_crfound_region = np.zeros_like(self.mask_crfound, dtype=bool)
            mask_crfound_region[cleaning_region] = self.mask_crfound[cleaning_region]
            data_has_been_modified = False
            if np.any(mask_crfound_region):
                if cleaning_method == "lacosmic":
                    # Replace detected CR pixels with L.A.Cosmic values
                    self.data[mask_crfound_region] = self.cleandata_lacosmic[mask_crfound_region]
                    # update mask_fixed to include the newly fixed pixels
                    self.mask_fixed[mask_crfound_region] = True
                    # upate mask_crfound by eliminating the cleaned pixels
                    self.mask_crfound[mask_crfound_region] = False
                    data_has_been_modified = True
                elif cleaning_method == "maskfill":
                    # Replace detected CR pixels with local median values
                    print(
                        f"Maskfill parameters: size={self.last_maskfill_size}, "
                        f"operator={self.last_maskfill_operator}, "
                        f"smooth={self.last_maskfill_smooth}, "
                        f"verbose={self.last_maskfill_verbose}"
                    )
                    smoothed_output, _ = maskfill(
                        input_image=self.data,
                        mask=mask_crfound_region,
                        size=self.last_maskfill_size,
                        operator=self.last_maskfill_operator,
                        smooth=self.last_maskfill_smooth,
                        verbose=self.last_maskfill_verbose,
                    )
                    self.data[mask_crfound_region] = smoothed_output[mask_crfound_region]
                    # update mask_fixed to include the newly fixed pixels
                    self.mask_fixed[mask_crfound_region] = True
                    # upate mask_crfound by eliminating the cleaned pixels
                    self.mask_crfound[mask_crfound_region] = False
                    data_has_been_modified = True
                elif cleaning_method == "auxdata":
                    if self.auxdata is None:
                        print("No auxiliary data available. Cleaning skipped!")
                        return
                    # Replace detected CR pixels with auxiliary data values
                    self.data[mask_crfound_region] = self.auxdata[mask_crfound_region]
                    # update mask_fixed to include the newly fixed pixels
                    self.mask_fixed[mask_crfound_region] = True
                    # upate mask_crfound by eliminating the cleaned pixels
                    self.mask_crfound[mask_crfound_region] = False
                    data_has_been_modified = True
                else:
                    # Determine features to process within the selected region
                    features_in_region = np.unique(self.cr_labels[mask_crfound_region])
                    with ModalProgressBar(
                        parent=self.root, iterable=range(1, self.num_features + 1), desc="Cleaning cosmic rays"
                    ) as pbar:
                        for i in pbar:
                            if i in features_in_region:
                                tmp_mask_fixed = np.zeros_like(self.data, dtype=bool)
                                if cleaning_method == "x":
                                    interpolation_performed, _, _ = interpolation_x(
                                        data=self.data,
                                        mask_fixed=tmp_mask_fixed,
                                        cr_labels=self.cr_labels,
                                        cr_index=i,
                                        npoints=editor.npoints,
                                        degree=editor.degree,
                                    )
                                elif cleaning_method == "y":
                                    interpolation_performed, _, _ = interpolation_y(
                                        data=self.data,
                                        mask_fixed=tmp_mask_fixed,
                                        cr_labels=self.cr_labels,
                                        cr_index=i,
                                        npoints=editor.npoints,
                                        degree=editor.degree,
                                    )
                                elif cleaning_method == "a-plane":
                                    interpolation_performed, _, _ = interpolation_a(
                                        data=self.data,
                                        mask_fixed=tmp_mask_fixed,
                                        cr_labels=self.cr_labels,
                                        cr_index=i,
                                        npoints=editor.npoints,
                                        method="surface",
                                    )
                                elif cleaning_method == "a-median":
                                    interpolation_performed, _, _ = interpolation_a(
                                        data=self.data,
                                        mask_fixed=tmp_mask_fixed,
                                        cr_labels=self.cr_labels,
                                        cr_index=i,
                                        npoints=editor.npoints,
                                        method="median",
                                    )
                                elif cleaning_method == "a-mean":
                                    interpolation_performed, _, _ = interpolation_a(
                                        data=self.data,
                                        mask_fixed=tmp_mask_fixed,
                                        cr_labels=self.cr_labels,
                                        cr_index=i,
                                        npoints=editor.npoints,
                                        method="mean",
                                    )
                                else:
                                    raise ValueError(f"Unknown cleaning method: {cleaning_method}")
                                if interpolation_performed:
                                    # update mask_fixed to include the newly fixed pixels
                                    self.mask_fixed[tmp_mask_fixed] = True
                                    # upate mask_crfound by eliminating the cleaned pixels
                                    self.mask_crfound[tmp_mask_fixed] = False
                                    # mark that data has been modified
                                    data_has_been_modified = True
            # If any pixels were cleaned, print message
            if data_has_been_modified:
                print("Cosmic ray cleaning applied.")
            else:
                print("No cosmic ray pixels cleaned.")
            # recalculate labels and number of features
            structure = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
            self.cr_labels, self.num_features = ndimage.label(self.mask_crfound, structure=structure)
            num_cr_remaining = np.sum(self.mask_crfound)
            sdum = str(num_cr_remaining)
            print(f"Remaining number of cosmic ray pixels...................: {sdum}")
            print(f"Remaining number of cosmic ray features (grouped pixels): {self.num_features:>{len(sdum)}}")
            if num_cr_remaining == 0:
                self.use_cursor = False
                self.use_cursor_button.config(text="[c]ursor: OFF")
            # redraw image to show the changes
            self.image.set_data(self.data)
            self.canvas.draw_idle()
            if data_has_been_modified:
                self.save_button.config(state=tk.NORMAL)
            if self.num_features == 0:
                self.review_detected_cr_button.config(state=tk.DISABLED)
                self.replace_detected_cr_button.config(state=tk.DISABLED)
            self.update_cr_overlay()

    def review_detected_cr(self, first_cr_index=1, single_cr=False, ixpix=None, iypix=None):
        """Open a window to examine and possibly clean detected cosmic rays."""
        self.working_in_review_window = True
        review_window = tk.Toplevel(self.root)
        center_on_parent(child=review_window, parent=self.root)
        if ixpix is not None and iypix is not None:
            # select single pixel based on provided coordinates
            tmp_cr_labels = np.zeros_like(self.data, dtype=int)
            tmp_cr_labels[iypix - 1, ixpix - 1] = 1
            review = ReviewCosmicRay(
                root=review_window,
                root_width=self.width,
                root_height=self.height,
                data=self.data,
                auxdata=self.auxdata,
                cleandata_lacosmic=self.cleandata_lacosmic,
                cr_labels=tmp_cr_labels,
                num_features=1,
                first_cr_index=1,
                single_cr=True,
                last_dilation=self.lacosmic_params["dilation"]["value"],
                last_npoints=self.last_npoints,
                last_degree=self.last_degree,
            )
        else:
            review = ReviewCosmicRay(
                root=review_window,
                root_width=self.width,
                root_height=self.height,
                data=self.data,
                auxdata=self.auxdata,
                cleandata_lacosmic=self.cleandata_lacosmic,
                cr_labels=self.cr_labels,
                num_features=self.num_features,
                first_cr_index=first_cr_index,
                single_cr=single_cr,
                last_dilation=self.lacosmic_params["dilation"]["value"],
                last_npoints=self.last_npoints,
                last_degree=self.last_degree,
            )
        # Make it modal (blocks interaction with main window)
        review_window.transient(self.root)
        review_window.grab_set()
        self.root.wait_window(review_window)
        self.working_in_review_window = False
        # Get the result after window closes
        if review.num_cr_cleaned > 0:
            self.last_npoints = review.npoints
            self.last_degree = review.degree
            print(f"Number of cosmic rays identified and cleaned: {review.num_cr_cleaned}")
            # update mask_fixed to include the newly fixed pixels
            self.mask_fixed[review.mask_fixed] = True
            # upate mask_crfound by eliminating the cleaned pixels
            self.mask_crfound[review.mask_fixed] = False
            # recalculate labels and number of features
            structure = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
            self.cr_labels, self.num_features = ndimage.label(self.mask_crfound, structure=structure)
            num_remaining = np.sum(self.mask_crfound)
            sdum = str(num_remaining)
            print(f"Remaining number of cosmic ray pixels...................: {sdum}")
            print(f"Remaining number of cosmic ray features (grouped pixels): {self.num_features:>{len(sdum)}}")
            if num_remaining == 0:
                self.use_cursor = False
                self.use_cursor_button.config(text="[c]ursor: OFF")
            # redraw image to show the changes
            self.image.set_data(self.data)
            self.canvas.draw_idle()
        if review.num_cr_cleaned > 0:
            self.save_button.config(state=tk.NORMAL)
        if self.num_features == 0:
            self.review_detected_cr_button.config(state=tk.DISABLED)
            self.replace_detected_cr_button.config(state=tk.DISABLED)
        self.update_cr_overlay()

    def stop_app(self):
        """Stop the application, prompting to save if there are unsaved changes."""
        proceed_with_stop = True
        if self.save_button["state"] == tk.NORMAL:
            print("Warning: There are unsaved changes!")
            proceed_with_stop = messagebox.askyesno(
                "Unsaved Changes", "You have unsaved changes.\nDo you really want to quit?", default=messagebox.NO
            )
        if proceed_with_stop:
            self.root.quit()
            self.root.destroy()

    def on_key(self, event):
        """Handle key press events."""
        if event.key == "c":
            self.set_cursor_onoff()
        elif event.key == "a":
            self.toggle_aspect()
        elif event.key == "t" and self.auxdata is not None:
            self.toggle_auxdata()
        elif event.key == ",":
            self.set_minmax()
        elif event.key == "/":
            self.set_zscale()
        elif event.key == "o":
            self.toolbar.zoom()
        elif event.key == "h":
            self.toolbar.home()
        elif event.key == "p":
            self.toolbar.pan()
        elif event.key == "s":
            self.toolbar.save_figure()
        elif event.key == "?":
            # Display list of keyboard shortcuts
            print("[bold blue]Keyboard Shortcuts:[/bold blue]")
            print("[red]  c [/red]: Toggle cursor selection mode on/off")
            print("[red]  t [/red]: Toggle between main data and auxiliary data")
            print("[red]  a [/red]: Toggle image aspect ratio equal/auto")
            print("[red]  , [/red]: Set vmin and vmax to minmax")
            print("[red]  / [/red]: Set vmin and vmax using zscale")
            print("[red]  h [/red]: Go to home view \\[toolbar]")
            print("[red]  o [/red]: Activate zoom mode \\[toolbar]")
            print("[red]  p [/red]: Activate pan mode \\[toolbar]")
            print("[red]  s [/red]: Save the current figure \\[toolbar]")
            print("[red]  q [/red]: (ignored) prevent closing the window")
        elif event.key == "q":
            pass  # Ignore the "q" key to prevent closing the window

    def on_click(self, event):
        """Handle mouse click events on the image."""
        # ignore clicks if we are working in the review window
        if self.working_in_review_window:
            print("Currently working in review window; click ignored.")
            return

        # check the toolbar is not active
        toolbar = self.fig.canvas.toolbar
        if toolbar.mode != "":
            print(f"Toolbar mode '{toolbar.mode}' active; click ignored.")
            return

        # proceed only if cursor selection mode is on
        if not self.use_cursor:
            return

        # ignore clicks outside the expected axes
        # (note that the color bar is a different axes)
        if event.inaxes == self.ax:
            x, y = event.xdata, event.ydata
            ix = int(x + 0.5)
            iy = int(y + 0.5)
            print(f"Clicked at image coordinates: ({ix}, {iy})")
            label_at_click = 0
            if self.mask_crfound is None:
                print("No cosmic ray pixels detected (mask_crfound is None)")
            elif not np.any(self.mask_crfound):
                print("No remaining cosmic ray pixels in mask_crfound")
            else:
                label_at_click = self.cr_labels[iy - 1, ix - 1]
                if label_at_click == 0:
                    (closest_x, closest_y), min_distance = find_closest_true(self.mask_crfound, ix - 1, iy - 1)
                    if closest_x is None and closest_y is None:
                        print("No remaining cosmic ray pixels")
                    elif min_distance > MAX_PIXEL_DISTANCE_TO_CR * 1.4142135:
                        print("No nearby cosmic ray pixels found in searching square")
                    else:
                        label_at_click = self.cr_labels[closest_y, closest_x]
                        print(f"Clicked pixel is part of cosmic ray number {label_at_click}.")
            if label_at_click == 0:
                # Find pixel with maximum value within a square region around the click
                semiwidth = MAX_PIXEL_DISTANCE_TO_CR
                jmin = (ix - 1) - semiwidth if (ix - 1) - semiwidth >= 0 else 0
                jmax = (ix - 1) + semiwidth if (ix - 1) + semiwidth < self.data.shape[1] else self.data.shape[1] - 1
                imin = (iy - 1) - semiwidth if (iy - 1) - semiwidth >= 0 else 0
                imax = (iy - 1) + semiwidth if (iy - 1) + semiwidth < self.data.shape[0] else self.data.shape[0] - 1
                ijmax = np.unravel_index(
                    np.argmax(self.data[imin : imax + 1, jmin : jmax + 1]),
                    self.data[imin : imax + 1, jmin : jmax + 1].shape,
                )
                ixpix = ijmax[1] + jmin + 1
                iypix = ijmax[0] + imin + 1
            else:
                ixpix = None
                iypix = None
            self.review_detected_cr(label_at_click, single_cr=True, ixpix=ixpix, iypix=iypix)
