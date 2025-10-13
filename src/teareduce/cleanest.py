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
from tkinter import simpledialog

from astropy.io import fits
from ccdproc import cosmicray_lacosmic
import matplotlib.pyplot as plt
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import os
from scipy import ndimage

from .imshow import imshow
from .sliceregion import SliceRegion2D
from .zscale import zscale

import matplotlib
matplotlib.use("TkAgg")


# Disable de "q" shortcut in matplotlib to avoid conflicts with tkinter
def on_key(event):
    if event.key == 'q':
        pass  # Ignore the "q" key to prevent closing the window
    else:
        print(f"Key pressed: {event.key}")


class CosmicRayCleanerApp():
    """Main application class for cosmic ray cleaning."""

    def __init__(self, root, fits_file_path, extension=0):
        """
        Initialize the application.

        Parameters
        ----------
        root : tk.Tk
            The main Tkinter window.
        fits_file_path : str
            Path to the FITS file to be cleaned.
        extension : int, optional
            FITS extension to use (default is 0).
        """
        self.root = root
        self.root.title("Cosmic Ray Cleaner")
        self.root.geometry("800x700+50+0")
        self.fits_file_path = fits_file_path
        self.extension = extension
        self.load_fits_file()
        self.create_widgets()

    def load_fits_file(self):
        try:
            with fits.open(self.fits_file_path) as hdul:
                self.data = hdul[self.extension].data
        except Exception as e:
            print(f"Error loading FITS file: {e}")

    def create_widgets(self):
        # Row 1
        self.button_frame1 = tk.Frame(self.root)
        self.button_frame1.grid(row=0, column=0, pady=5)
        self.run_lacosmic_button = tk.Button(self.button_frame1, text="Run L.A.Cosmic", command=self.run_lacosmic)
        self.run_lacosmic_button.pack(side=tk.LEFT, padx=5)

        # Row 2
        self.button_frame2 = tk.Frame(self.root)
        self.button_frame2.grid(row=1, column=0, pady=5)
        vmin, vmax = zscale(self.data)
        self.vmin_button = tk.Button(self.button_frame2, text=f"vmin: {vmin:.2f}", command=self.set_vmin)
        self.vmin_button.pack(side=tk.LEFT, padx=5)
        self.vmax_button = tk.Button(self.button_frame2, text=f"vmax: {vmax:.2f}", command=self.set_vmax)
        self.vmax_button.pack(side=tk.LEFT, padx=5)
        self.exit_button = tk.Button(self.button_frame2, text="Exit", command=self.exit_app)
        self.exit_button.pack(side=tk.LEFT, padx=5)

        # Main frame for figure and toolbar
        self.main_frame = tk.Frame(self.root)
        self.main_frame.grid(row=2, column=0, sticky="nsew")
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.image, _, _ = imshow(self.fig, self.ax, self.data, vmin=vmin, vmax=vmax, cmap="viridis")
        # Note: tight_layout should be called before defining the canvas
        self.fig.tight_layout()

        # Create canvas and toolbar
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        # The next two instructions prevent a segmentation fault when pressing "q"
        self.canvas.mpl_disconnect(self.canvas.mpl_connect("key_press_event", key_press_handler))
        self.canvas.mpl_connect("key_press_event", on_key)
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
        # Parameters for L.A.Cosmic can be adjusted as needed
        cleaned_data, mask = cosmicray_lacosmic(self.data, sigclip=4.5, sigfrac=0.3, objlim=5.0, verbose=True)
        labels_cr, num_cr = ndimage.label(mask)
        print(f"Number of cosmic ray pixels detected: {np.sum(mask)}")
        print(f"Number of cosmic rays detected: {num_cr}")
        for i in range(1, num_cr + 1):
            y, x = np.where(labels_cr == i)
            print(f"Cosmic ray {i}: Number of pixels = {len(x)}, Centroid = ({np.mean(x):.2f}, {np.mean(y):.2f})")
            i0 = int(np.mean(y) + 0.5)
            j0 = int(np.mean(x) + 0.5)
            jmin = j0 - 15 if j0 - 15 >= 0 else 0
            jmax = j0 + 15 if j0 + 15 < self.data.shape[1] else self.data.shape[1] - 1
            imin = i0 - 15 if i0 - 15 >= 0 else 0
            imax = i0 + 15 if i0 + 15 < self.data.shape[0] else self.data.shape[0] - 1
            region = SliceRegion2D(f'[{jmin+1}:{jmax+1}, {imin+1}:{imax+1}]', mode='fits').python
            self.image.set_data(self.data[region])
            self.image.set_extent([jmin + 1, jmax + 1, imin + 1, imax + 1])
            self.ax.set_title(f"Zoom on cosmic ray {i}")
            self.fig.tight_layout()
            self.canvas.draw()
            input("Press Enter to continue...")
        print("L.A.Cosmic cleaning applied.")

    def exit_app(self):
        self.root.quit()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Interactive cosmic ray cleaner for FITS images.")
    parser.add_argument("fits_file", help="Path to the FITS file to be cleaned.")
    parser.add_argument("--extension", type=int, default=0,
                        help="FITS extension to use (default: 0).")
    args = parser.parse_args()

    if not os.path.isfile(args.fits_file):
        print(f"Error: File '{args.fits_file}' does not exist.")
        return

    # Initialize Tkinter root
    root = tk.Tk()

    # Create and run the application
    app = CosmicRayCleanerApp(root, args.fits_file, args.extension)

    # Execute
    root.mainloop()


if __name__ == "__main__":
    main()
