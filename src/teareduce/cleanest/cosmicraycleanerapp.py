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
from tkinter import simpledialog

from astropy.io import fits
from ccdproc import cosmicray_lacosmic
import matplotlib.pyplot as plt
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import os

from .reviewcosmicray import ReviewCosmicRay

from ..imshow import imshow
from ..zscale import zscale

import matplotlib
matplotlib.use("TkAgg")


class CosmicRayCleanerApp():
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
        # Row 1
        self.button_frame1 = tk.Frame(self.root)
        self.button_frame1.grid(row=0, column=0, pady=5)
        self.run_lacosmic_button = tk.Button(self.button_frame1, text="Run L.A.Cosmic", command=self.run_lacosmic)
        self.run_lacosmic_button.pack(side=tk.LEFT, padx=5)
        self.save_button = tk.Button(self.button_frame1, text="Save cleaned FITS", command=self.save_fits_file)
        self.save_button.pack(side=tk.LEFT, padx=5)

        # Row 2
        self.button_frame2 = tk.Frame(self.root)
        self.button_frame2.grid(row=1, column=0, pady=5)
        vmin, vmax = zscale(self.data)
        self.vmin_button = tk.Button(self.button_frame2, text=f"vmin: {vmin:.2f}", command=self.set_vmin)
        self.vmin_button.pack(side=tk.LEFT, padx=5)
        self.vmax_button = tk.Button(self.button_frame2, text=f"vmax: {vmax:.2f}", command=self.set_vmax)
        self.vmax_button.pack(side=tk.LEFT, padx=5)
        self.stop_button = tk.Button(self.button_frame2, text="Stop program", command=self.stop_app)
        self.stop_button.pack(side=tk.LEFT, padx=5)

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
                                  xlabel=xlabel, ylabel=ylabel, extent=extent)
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

    def set_vmin(self):
        old_vmin = self.get_vmin()
        new_vmin = simpledialog.askfloat("Set vmin", "Enter new vmin:", initialvalue=old_vmin)
        if new_vmin is None:
            return
        self.vmin_button.config(text=f"vmin: {new_vmin:.2f}")
        self.image.set_clim(vmin=new_vmin)
        self.canvas.draw()

    def set_vmax(self):
        old_vmax = self.get_vmax()
        new_vmax = simpledialog.askfloat("Set vmax", "Enter new vmax:", initialvalue=old_vmax)
        if new_vmax is None:
            return
        self.vmax_button.config(text=f"vmax: {new_vmax:.2f}")
        self.image.set_clim(vmax=new_vmax)
        self.canvas.draw()

    def get_vmin(self):
        return float(self.vmin_button.cget("text").split(":")[1])

    def get_vmax(self):
        return float(self.vmax_button.cget("text").split(":")[1])

    def run_lacosmic(self):
        self.run_lacosmic_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.DISABLED)
        # Parameters for L.A.Cosmic can be adjusted as needed
        _, mask_crfound = cosmicray_lacosmic(self.data, sigclip=4.5, sigfrac=0.3, objlim=5.0, verbose=True)
        ReviewCosmicRay(
            root=self.root,
            data=self.data,
            mask_fixed=self.mask_fixed,
            mask_crfound=mask_crfound
        )
        print("L.A.Cosmic cleaning applied.")
        self.run_lacosmic_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.NORMAL)

    def stop_app(self):
        self.root.quit()
        self.root.destroy()

    def on_key(self, event):
        if event.key == 'q':
            pass  # Ignore the "q" key to prevent closing the window
        else:
            print(f"Key pressed: {event.key}")

    def on_click(self, event):
        if event.inaxes:
            x, y = event.xdata, event.ydata
            print(f"Clicked at image coordinates: ({x:.2f}, {y:.2f})")
