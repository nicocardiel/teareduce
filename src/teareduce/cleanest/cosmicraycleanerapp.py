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

from astropy.io import fits
from ccdproc import cosmicray_lacosmic
import matplotlib.pyplot as plt
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import os
from rich import print

from .imagedisplay import ImageDisplay
from .reviewcosmicray import ReviewCosmicRay

from ..imshow import imshow
from ..sliceregion import SliceRegion2D
from ..zscale import zscale

import matplotlib
matplotlib.use("TkAgg")


class CosmicRayCleanerApp(ImageDisplay):
    """Main application class for cosmic ray cleaning."""

    def __init__(self, root, input_fits, extension=0, output_fits=None):
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
        output_fits : str, optional
            Path to save the cleaned FITS file (default is None, which prompts
            for a save location).
        """
        self.root = root
        self.root.title("Cosmic Ray Cleaner")
        self.root.geometry("800x700+50+0")
        self.input_fits = input_fits
        self.extension = extension
        self.output_fits = output_fits
        self.load_fits_file()
        self.create_widgets()

    def load_fits_file(self):
        try:
            with fits.open(self.input_fits, mode='readonly') as hdul:
                self.data = hdul[self.extension].data
                if 'CRMASK' in hdul:
                    self.mask_fixed = hdul['CRMASK'].data.astype(bool)
                else:
                    self.mask_fixed = np.zeros(self.data.shape, dtype=bool)
        except Exception as e:
            print(f"Error loading FITS file: {e}")
        naxis2, naxis1 = self.data.shape
        self.region = SliceRegion2D(f'[1:{naxis1}, 1:{naxis2}]', mode='fits').python

    def save_fits_file(self):
        if self.output_fits is None:
            base, ext = os.path.splitext(self.input_fits)
            suggested_name = f"{base}_cleaned"
        else:
            suggested_name, _ = os.path.splitext(self.output_fits)
        self.output_fits = filedialog.asksaveasfilename(
            initialdir=os.getcwd(),
            title="Save cleaned FITS file",
            defaultextension=".fits",
            filetypes=[("FITS files", "*.fits"), ("All files", "*.*")],
            initialfile=suggested_name
        )
        try:
            with fits.open(self.input_fits, mode='readonly') as hdul:
                hdul[self.extension].data = self.data
                if 'CRMASK' in hdul:
                    hdul['CRMASK'].data = self.mask_fixed.astype(np.uint8)
                else:
                    crmask_hdu = fits.ImageHDU(self.mask_fixed.astype(np.uint8), name='CRMASK')
                    hdul.append(crmask_hdu)
                hdul.writeto(self.output_fits, overwrite=True)
            print(f"Cleaned data saved to {self.output_fits}")
        except Exception as e:
            print(f"Error saving FITS file: {e}")

    def create_widgets(self):
        # Row 1 of buttons
        self.button_frame1 = tk.Frame(self.root)
        self.button_frame1.grid(row=0, column=0, pady=5)
        self.run_lacosmic_button = tk.Button(self.button_frame1, text="Run L.A.Cosmic", command=self.run_lacosmic)
        self.run_lacosmic_button.pack(side=tk.LEFT, padx=5)
        self.save_button = tk.Button(self.button_frame1, text="Save cleaned FITS", command=self.save_fits_file)
        self.save_button.pack(side=tk.LEFT, padx=5)
        self.save_button.config(state=tk.DISABLED)  # Initially disabled
        self.stop_button = tk.Button(self.button_frame1, text="Stop program", command=self.stop_app)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Row 2 of buttons
        self.button_frame2 = tk.Frame(self.root)
        self.button_frame2.grid(row=1, column=0, pady=5)
        vmin, vmax = zscale(self.data)
        self.vmin_button = tk.Button(self.button_frame2, text=f"vmin: {vmin:.2f}", command=self.set_vmin)
        self.vmin_button.pack(side=tk.LEFT, padx=5)
        self.vmax_button = tk.Button(self.button_frame2, text=f"vmax: {vmax:.2f}", command=self.set_vmax)
        self.vmax_button.pack(side=tk.LEFT, padx=5)
        self.set_minmax_button = tk.Button(self.button_frame2, text="minmax [,]", command=self.set_minmax)
        self.set_minmax_button.pack(side=tk.LEFT, padx=5)
        self.set_zscale_button = tk.Button(self.button_frame2, text="zscale [/]", command=self.set_zscale)
        self.set_zscale_button.pack(side=tk.LEFT, padx=5)

        # Main frame for figure and toolbar
        self.main_frame = tk.Frame(self.root)
        self.main_frame.grid(row=2, column=0, sticky="nsew")
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        xlabel = 'X pixel (from 1 to NAXIS1)'
        ylabel = 'Y pixel (from 1 to NAXIS2)'
        extent = [0.5, self.data.shape[1] + 0.5, 0.5, self.data.shape[0] + 0.5]
        self.image, _, _ = imshow(self.fig, self.ax, self.data, vmin=vmin, vmax=vmax,
                                  xlabel=xlabel, ylabel=ylabel,
                                  title=os.path.basename(self.input_fits),
                                  extent=extent)
        # Note: tight_layout should be called before defining the canvas
        self.fig.tight_layout()

        # Create canvas and toolbar
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        # The next two instructions prevent a segmentation fault when pressing "q"
        self.canvas.mpl_disconnect(self.canvas.mpl_connect("key_press_event", key_press_handler))
        self.canvas.mpl_connect("key_press_event", self.on_key)
        self.canvas.mpl_connect("button_press_event", self.on_click)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.grid(row=0, column=0, sticky="nsew")

        # Matplotlib toolbar
        self.toolbar_frame = tk.Frame(self.main_frame)
        self.toolbar_frame.grid(row=1, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()

    def run_lacosmic(self):
        self.run_lacosmic_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.DISABLED)
        # Parameters for L.A.Cosmic can be adjusted as needed
        cleandata_lacosmic, mask_crfound = cosmicray_lacosmic(
            self.data,
            sigclip=4.5,
            sigfrac=0.3,
            objlim=5.0,
            verbose=True
        )
        review = ReviewCosmicRay(
            root=self.root,
            data=self.data,
            cleandata_lacosmic=cleandata_lacosmic,
            mask_fixed=self.mask_fixed,
            mask_crfound=mask_crfound
        )
        if review.num_cr_cleaned > 0:
            print(f"Number of cosmic rays identified and cleaned: {review.num_cr_cleaned}")
            # redraw image to show the changes
            self.image.set_data(self.data)
            self.canvas.draw()
            self.save_button.config(state=tk.NORMAL)
        self.run_lacosmic_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.NORMAL)

    def stop_app(self):
        self.root.quit()
        self.root.destroy()

    def on_key(self, event):
        if event.key == 'q':
            pass  # Ignore the "q" key to prevent closing the window
        elif event.key == ',':
            self.set_minmax()
        elif event.key == '/':
            self.set_zscale()
        else:
            print(f"Key pressed: {event.key}")

    def on_click(self, event):
        # check the toolbar is not active
        toolbar = self.fig.canvas.toolbar
        if toolbar.mode != "":
            print(f"Toolbar mode '{toolbar.mode}' active; click ignored.")
            return

        # ignore clicks outside the expected axes
        # (note that the color bar is a different axes)
        if event.inaxes == self.ax:
            x, y = event.xdata, event.ydata
            print(f"Clicked at image coordinates: ({x:.2f}, {y:.2f})")
            mask_crfound = np.zeros(self.data.shape, dtype=bool)
            ix = int(x + 0.5)
            iy = int(y + 0.5)
            mask_crfound[iy-1, ix-1] = True
            review = ReviewCosmicRay(
                root=self.root,
                data=self.data,
                cleandata_lacosmic=None,
                mask_fixed=self.mask_fixed,
                mask_crfound=mask_crfound
            )
            if review.num_cr_cleaned > 0:
                print(f"Number of cosmic rays identified and cleaned: {review.num_cr_cleaned}")
                # redraw image to show the changes
                self.image.set_data(self.data)
                self.canvas.draw()
                self.save_button.config(state=tk.NORMAL)
