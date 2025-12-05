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
import sys

from astropy.io import fits
from ccdproc import cosmicray_lacosmic
import matplotlib.pyplot as plt
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy import ndimage
import numpy as np
import os
from rich import print

from .centerchildparent import center_on_parent
from .definitions import lacosmic_default_dict
from .definitions import DEFAULT_NPOINTS_INTERP
from .definitions import DEFAULT_DEGREE_INTERP
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
from .parametereditor import ParameterEditor
from .reviewcosmicray import ReviewCosmicRay
from .modalprogressbar import ModalProgressBar

from ..imshow import imshow
from ..sliceregion import SliceRegion2D
from ..zscale import zscale

import matplotlib

matplotlib.use("TkAgg")


class CosmicRayCleanerApp(ImageDisplay):
    """Main application class for cosmic ray cleaning."""

    def __init__(
        self,
        root,
        input_fits,
        extension=0,
        auxfile=None,
        extension_auxfile=0,
        fontfamily=DEFAULT_FONT_FAMILY,
        fontsize=DEFAULT_FONT_SIZE,
        width=DEFAULT_TK_WINDOW_SIZE_X,
        height=DEFAULT_TK_WINDOW_SIZE_Y,
    ):
        """
        Initialize the application.

        Parameters
        ----------
        root : tk.Tk
            The main Tkinter window.
        input_fits : str
            Path to the FITS file to be cleaned.
        extension : int, optional
            FITS extension to use (default is 0).
        auxfile : str, optional
            Path to an auxiliary FITS file (default is None).
        extension_auxfile : int, optional
            FITS extension for auxiliary file (default is 0).
        fontfamily : str, optional
            Font family for the GUI (default is "Helvetica").
        fontsize : int, optional
            Font size for the GUI (default is 14).

        Methods
        -------
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
        examine_detected_cr()
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
        last_npoints : int
            Last used number of points for interpolation.
        last_degree : int
            Last used degree for interpolation.
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
        # self.root.geometry("800x800+50+0")  # This does not work in Fedora
        self.width = width
        self.height = height
        self.root.minsize(self.width, self.height)
        self.root.update_idletasks()
        self.root.title("Cosmic Ray Cleaner")
        self.fontfamily = fontfamily
        self.fontsize = fontsize
        self.default_font = tkfont.nametofont("TkDefaultFont")
        self.default_font.configure(
            family=fontfamily, size=fontsize, weight="normal", slant="roman", underline=0, overstrike=0
        )
        self.lacosmic_params = lacosmic_default_dict.copy()
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
        self.last_npoints = DEFAULT_NPOINTS_INTERP
        self.last_degree = DEFAULT_DEGREE_INTERP
        self.create_widgets()
        self.cleandata_lacosmic = None
        self.cr_labels = None
        self.num_features = 0
        self.working_in_review_window = False

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
        try:
            with fits.open(self.input_fits, mode="readonly") as hdul:
                self.data = hdul[self.extension].data
                if "CRMASK" in hdul:
                    self.mask_fixed = hdul["CRMASK"].data.astype(bool)
                else:
                    self.mask_fixed = np.zeros(self.data.shape, dtype=bool)
        except Exception as e:
            print(f"Error loading FITS file: {e}")
        self.mask_crfound = np.zeros(self.data.shape, dtype=bool)
        naxis2, naxis1 = self.data.shape
        self.region = SliceRegion2D(f"[1:{naxis1}, 1:{naxis2}]", mode="fits").python
        # Read auxiliary file if provided
        if self.auxfile is not None:
            try:
                with fits.open(self.auxfile, mode="readonly") as hdul_aux:
                    self.auxdata = hdul_aux[self.extension_auxfile].data
                    if self.auxdata.shape != self.data.shape:
                        print(f"data shape...: {self.data.shape}")
                        print(f"auxdata shape: {self.auxdata.shape}")
                        raise ValueError("Auxiliary file has different shape.")
            except Exception as e:
                sys.exit(f"Error loading auxiliary FITS file: {e}")

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
        # Row 1 of buttons
        self.button_frame1 = tk.Frame(self.root)
        self.button_frame1.pack(pady=5)
        self.run_lacosmic_button = tk.Button(self.button_frame1, text="Run L.A.Cosmic", command=self.run_lacosmic)
        self.run_lacosmic_button.pack(side=tk.LEFT, padx=5)
        self.apply_lacosmic_button = tk.Button(
            self.button_frame1, text="Replace detected CRs", command=self.apply_lacosmic
        )
        self.apply_lacosmic_button.pack(side=tk.LEFT, padx=5)
        self.apply_lacosmic_button.config(state=tk.DISABLED)  # Initially disabled
        self.examine_detected_cr_button = tk.Button(
            self.button_frame1, text="Examine detected CRs", command=lambda: self.examine_detected_cr(1)
        )
        self.examine_detected_cr_button.pack(side=tk.LEFT, padx=5)
        self.examine_detected_cr_button.config(state=tk.DISABLED)  # Initially disabled
        self.cursor_selection_mode = False
        self.run_cursor_button = tk.Button(self.button_frame1, text="[c]ursor: OFF", command=self.set_cursor_onoff)
        self.run_cursor_button.pack(side=tk.LEFT, padx=5)

        # Row 2 of buttons
        self.button_frame2 = tk.Frame(self.root)
        self.button_frame2.pack(pady=5)
        self.toggle_auxdata_button = tk.Button(self.button_frame2, text="[t]oggle data", command=self.toggle_auxdata)
        self.toggle_auxdata_button.pack(side=tk.LEFT, padx=5)
        if self.auxdata is None:
            self.toggle_auxdata_button.config(state=tk.DISABLED)
        else:
            self.toggle_auxdata_button.config(state=tk.NORMAL)
        self.image_aspect = "equal"
        self.toggle_aspect_button = tk.Button(
            self.button_frame2, text=f"[a]spect: {self.image_aspect}", command=self.toggle_aspect
        )
        self.toggle_aspect_button.pack(side=tk.LEFT, padx=5)
        self.save_button = tk.Button(self.button_frame2, text="Save cleaned FITS", command=self.save_fits_file)
        self.save_button.pack(side=tk.LEFT, padx=5)
        self.save_button.config(state=tk.DISABLED)  # Initially disabled
        self.stop_button = tk.Button(self.button_frame2, text="Stop program", command=self.stop_app)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Row 3 of buttons
        self.button_frame3 = tk.Frame(self.root)
        self.button_frame3.pack(pady=5)
        vmin, vmax = zscale(self.data)
        self.vmin_button = tk.Button(self.button_frame3, text=f"vmin: {vmin:.2f}", command=self.set_vmin)
        self.vmin_button.pack(side=tk.LEFT, padx=5)
        self.vmax_button = tk.Button(self.button_frame3, text=f"vmax: {vmax:.2f}", command=self.set_vmax)
        self.vmax_button.pack(side=tk.LEFT, padx=5)
        self.set_minmax_button = tk.Button(self.button_frame3, text="minmax [,]", command=self.set_minmax)
        self.set_minmax_button.pack(side=tk.LEFT, padx=5)
        self.set_zscale_button = tk.Button(self.button_frame3, text="zscale [/]", command=self.set_zscale)
        self.set_zscale_button.pack(side=tk.LEFT, padx=5)
        if self.overplot_cr_pixels:
            self.overplot_cr_button = tk.Button(
                self.button_frame3,
                text="CR overlay: ON ",
                command=self.toggle_cr_overlay,
            )
        else:
            self.overplot_cr_button = tk.Button(
                self.button_frame3,
                text="CR overlay: OFF",
                command=self.toggle_cr_overlay,
            )
        self.overplot_cr_button.pack(side=tk.LEFT, padx=5)

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
        if not self.cursor_selection_mode:
            self.cursor_selection_mode = True
            self.run_cursor_button.config(text="[c]ursor: ON ")
        else:
            self.cursor_selection_mode = False
            self.run_cursor_button.config(text="[c]ursor: OFF")

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
            usefulregion = SliceRegion2D(
                f"[{self.last_xmin}:{self.last_xmax},{self.last_ymin}:{self.last_ymax}]", mode="fits"
            ).python
            usefulmask = np.zeros_like(self.data)
            usefulmask[usefulregion] = 1.0
            # Update parameter dictionary with new values
            self.lacosmic_params = updated_params
            print("Parameters updated:")
            for key, info in self.lacosmic_params.items():
                print(f"  {key}: {info['value']}")
            if self.lacosmic_params["nruns"]["value"] not in [1, 2]:
                raise ValueError("nruns must be 1 or 2")
            # Execute L.A.Cosmic with updated parameters
            print("[bold green]Executing L.A.Cosmic (run 1)...[/bold green]")
            borderpadd = updated_params["borderpadd"]["value"]
            data_reflection_padded = np.pad(self.data, pad_width=borderpadd, mode="reflect")
            cleandata_lacosmic, mask_crfound = cosmicray_lacosmic(
                ccd=data_reflection_padded,
                gain=self.lacosmic_params["run1_gain"]["value"],
                readnoise=self.lacosmic_params["run1_readnoise"]["value"],
                sigclip=self.lacosmic_params["run1_sigclip"]["value"],
                sigfrac=self.lacosmic_params["run1_sigfrac"]["value"],
                objlim=self.lacosmic_params["run1_objlim"]["value"],
                niter=self.lacosmic_params["run1_niter"]["value"],
                verbose=self.lacosmic_params["run1_verbose"]["value"],
            )
            cleandata_lacosmic = cleandata_lacosmic[borderpadd:-borderpadd, borderpadd:-borderpadd]
            mask_crfound = mask_crfound[borderpadd:-borderpadd, borderpadd:-borderpadd]
            # Apply usefulmask to consider only selected region
            cleandata_lacosmic *= usefulmask
            mask_crfound = mask_crfound & (usefulmask.astype(bool))
            # Second execution if nruns == 2
            if self.lacosmic_params["nruns"]["value"] == 2:
                print("[bold green]Executing L.A.Cosmic (run 2)...[/bold green]")
                cleandata_lacosmic2, mask_crfound2 = cosmicray_lacosmic(
                    ccd=data_reflection_padded,
                    gain=self.lacosmic_params["run2_gain"]["value"],
                    readnoise=self.lacosmic_params["run2_readnoise"]["value"],
                    sigclip=self.lacosmic_params["run2_sigclip"]["value"],
                    sigfrac=self.lacosmic_params["run2_sigfrac"]["value"],
                    objlim=self.lacosmic_params["run2_objlim"]["value"],
                    niter=self.lacosmic_params["run2_niter"]["value"],
                    verbose=self.lacosmic_params["run2_verbose"]["value"],
                )
                cleandata_lacosmic2 = cleandata_lacosmic2[borderpadd:-borderpadd, borderpadd:-borderpadd]
                mask_crfound2 = mask_crfound2[borderpadd:-borderpadd, borderpadd:-borderpadd]
                # Apply usefulmask to consider only selected region
                cleandata_lacosmic2 *= usefulmask
                mask_crfound2 = mask_crfound2 & (usefulmask.astype(bool))
                # Combine results from both runs
                if np.any(mask_crfound):
                    print(f"Number of cosmic ray pixels (run1).......: {np.sum(mask_crfound)}")
                    print(f"Number of cosmic ray pixels (run2).......: {np.sum(mask_crfound2)}")
                    # find features in second run
                    structure = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
                    cr_labels2, num_features2 = ndimage.label(mask_crfound2, structure=structure)
                    # generate mask of ones at CR pixels found in first run
                    mask_peaks = np.zeros(mask_crfound.shape, dtype=float)
                    mask_peaks[mask_crfound] = 1.0
                    # preserve only those CR pixels in second run that are in the first run
                    cr_labels2_preserved = mask_peaks * cr_labels2
                    # generate new mask with preserved CR pixels from second run
                    mask_crfound = np.zeros_like(mask_crfound, dtype=bool)
                    for icr in np.unique(cr_labels2_preserved):
                        if icr > 0:
                            mask_crfound[cr_labels2 == icr] = True
                    print(f"Number of cosmic ray pixels (run1 & run2): {np.sum(mask_crfound)}")
                # Use the cleandata from the second run
                cleandata_lacosmic = cleandata_lacosmic2
            # Select the image region to process
            self.cleandata_lacosmic = self.data.copy()
            self.cleandata_lacosmic[usefulregion] = cleandata_lacosmic[usefulregion]
            self.mask_crfound = np.zeros_like(self.data, dtype=bool)
            self.mask_crfound[usefulregion] = mask_crfound[usefulregion]
            # Process the mask: dilation and labeling
            if np.any(self.mask_crfound):
                num_cr_pixels_before_dilation = np.sum(self.mask_crfound)
                dilation = self.lacosmic_params["dilation"]["value"]
                if dilation > 0:
                    # Dilate the mask by the specified number of pixels
                    self.mask_crfound = dilatemask(
                        mask=self.mask_crfound, iterations=self.lacosmic_params["dilation"]["value"], connectivity=1
                    )
                    num_cr_pixels_after_dilation = np.sum(self.mask_crfound)
                    sdum = str(num_cr_pixels_after_dilation)
                else:
                    sdum = str(num_cr_pixels_before_dilation)
                print(
                    "Number of cosmic ray pixels detected by L.A.Cosmic: "
                    f"{num_cr_pixels_before_dilation:>{len(sdum)}}"
                )
                if dilation > 0:
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
                self.apply_lacosmic_button.config(state=tk.NORMAL)
                self.examine_detected_cr_button.config(state=tk.NORMAL)
                self.update_cr_overlay()
                self.cursor_selection_mode = True
                self.run_cursor_button.config(text="[c]ursor: ON ")
            else:
                print("No cosmic ray pixels detected by L.A.Cosmic.")
                self.cr_labels = None
                self.num_features = 0
                self.apply_lacosmic_button.config(state=tk.DISABLED)
                self.examine_detected_cr_button.config(state=tk.DISABLED)
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
            sdum = str(np.sum(self.mask_crfound))
            print(f"Number of cosmic ray pixels detected by L.A.Cosmic...........: {sdum}")
            print(f"Number of cosmic rays (grouped pixels) detected by L.A.Cosmic: {self.num_features:>{len(sdum)}}")
            # Define parameters for L.A.Cosmic from default dictionary
            editor_window = tk.Toplevel(self.root)
            center_on_parent(child=editor_window, parent=self.root)
            editor = InterpolationEditor(
                root=editor_window,
                last_dilation=self.lacosmic_params["dilation"]["value"],
                last_npoints=self.last_npoints,
                last_degree=self.last_degree,
                auxdata=self.auxdata,
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
            self.last_npoints = editor.npoints
            self.last_degree = editor.degree
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
            sdum = str(np.sum(self.mask_crfound))
            print(f"Remaining number of cosmic ray pixels...................: {sdum}")
            print(f"Remaining number of cosmic ray features (grouped pixels): {self.num_features:>{len(sdum)}}")
            # redraw image to show the changes
            self.image.set_data(self.data)
            self.canvas.draw_idle()
            if data_has_been_modified:
                self.save_button.config(state=tk.NORMAL)
            if self.num_features == 0:
                self.examine_detected_cr_button.config(state=tk.DISABLED)
                self.apply_lacosmic_button.config(state=tk.DISABLED)
            self.update_cr_overlay()

    def examine_detected_cr(self, first_cr_index=1, single_cr=False, ixpix=None, iypix=None):
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
            sdum = str(np.sum(self.mask_crfound))
            print(f"Remaining number of cosmic ray pixels...................: {sdum}")
            print(f"Remaining number of cosmic ray features (grouped pixels): {self.num_features:>{len(sdum)}}")
            # redraw image to show the changes
            self.image.set_data(self.data)
            self.canvas.draw_idle()
        if review.num_cr_cleaned > 0:
            self.save_button.config(state=tk.NORMAL)
        if self.num_features == 0:
            self.examine_detected_cr_button.config(state=tk.DISABLED)
            self.apply_lacosmic_button.config(state=tk.DISABLED)
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
        if not self.cursor_selection_mode:
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
            self.examine_detected_cr(label_at_click, single_cr=True, ixpix=ixpix, iypix=iypix)
