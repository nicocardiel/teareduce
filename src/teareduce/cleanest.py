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


class ReviewCosmicRay():
    """Class to review cosmic ray masked pixels."""

    def __init__(self, root, data, la_clean_data, mask):
        self.root = root
        self.data = data
        self.data_original = data.copy()
        self.la_clean_data = la_clean_data
        self.mask = mask
        self.first_plot = True
        self.degree = 1    # Degree of polynomial for interpolation
        self.npoints = 2   # Number of points at each side of the CR pixel for interpolation
        # Label connected components in the mask; note that by default,
        # structure is a cross [0,1,0;1,1,1;0,1,0], but we want to consider
        # diagonal connections too, so we define a 3x3 square.
        structure = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        self.cr_labels, self.num_features = ndimage.label(self.mask, structure=structure)
        # Make a copy of the original labels to allow pixel re-marking
        self.cr_labels_original = self.cr_labels.copy()
        print(f"Number of cosmic ray pixels detected: {np.sum(self.mask)}")
        print(f"Number of cosmic rays detected: {self.num_features}")
        if self.num_features == 0:
            print('No CR hits found!')
        else:
            self.cr_index = 1
            self.create_widgets()

    def create_widgets(self):
        self.review_window = tk.Toplevel(self.root)
        self.review_window.title("Review Cosmic Rays")
        self.review_window.geometry("800x700+850+0")

        self.button_frame1 = tk.Frame(self.review_window)
        self.button_frame1.pack(pady=5)
        self.remove_crosses_button = tk.Button(self.button_frame1, text="remove all X's", command=self.remove_crosses)
        self.remove_crosses_button.pack(side=tk.LEFT, padx=5)
        self.restore_cr_button = tk.Button(self.button_frame1, text="[r]estore CR", command=self.restore_cr)
        self.restore_cr_button.pack(side=tk.LEFT, padx=5)
        self.restore_cr_button.config(state=tk.DISABLED)
        self.next_button = tk.Button(self.button_frame1, text="[c]ontinue", command=self.continue_cr)
        self.next_button.pack(side=tk.LEFT, padx=5)

        self.button_frame2 = tk.Frame(self.review_window)
        self.button_frame2.pack(pady=5)
        self.ndeg_label = tk.Button(self.button_frame2, text=f"deg={self.degree}, n={self.npoints}",
                                    command=self.set_ndeg)
        self.ndeg_label.pack(side=tk.LEFT, padx=5)
        self.x_interp_button = tk.Button(self.button_frame2, text="[x] interp.", command=self.x_interp)
        self.x_interp_button.pack(side=tk.LEFT, padx=5)
        self.y_interp_button = tk.Button(self.button_frame2, text="[y] interp.", command=self.y_interp)
        self.y_interp_button.pack(side=tk.LEFT, padx=5)
        self.s_interp_button = tk.Button(self.button_frame2, text="[s] interp.", command=self.s_interp)
        self.s_interp_button.pack(side=tk.LEFT, padx=5)

        self.button_frame3 = tk.Frame(self.review_window)
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
        self.exit_button = tk.Button(self.button_frame3, text="[e]xit review", command=self.exit_review)
        self.exit_button.pack(side=tk.LEFT, padx=5)

        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.review_window)
        # The next two instructions prevent a segmentation fault when pressing "q"
        self.canvas.mpl_disconnect(self.canvas.mpl_connect("key_press_event", key_press_handler))
        self.canvas.mpl_connect("key_press_event", self.on_key)
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Matplotlib toolbar
        self.toolbar_frame = tk.Frame(self.review_window)
        self.toolbar_frame.pack(fill=tk.X, expand=False, pady=5)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()

        self.update_display()

        self.root.wait_window(self.review_window)

    def update_display(self):
        ycr_list, xcr_list = np.where(self.cr_labels == self.cr_index)
        ycr_list_original, xcr_list_original = np.where(self.cr_labels_original == self.cr_index)
        if self.first_plot:
            print(f"Cosmic ray {self.cr_index}: "
                  f"Number of pixels = {len(xcr_list)}, "
                  f"Centroid = ({np.mean(xcr_list):.2f}, {np.mean(ycr_list):.2f})")
        # Use original positions to define the region to display in order
        # to avoid image shifts when some pixels are unmarked or new ones are marked
        i0 = int(np.mean(ycr_list_original) + 0.5)
        j0 = int(np.mean(xcr_list_original) + 0.5)
        jmin = j0 - 15 if j0 - 15 >= 0 else 0
        jmax = j0 + 15 if j0 + 15 < self.data.shape[1] else self.data.shape[1] - 1
        imin = i0 - 15 if i0 - 15 >= 0 else 0
        imax = i0 + 15 if i0 + 15 < self.data.shape[0] else self.data.shape[0] - 1
        self.region = SliceRegion2D(f'[{jmin+1}:{jmax+1}, {imin+1}:{imax+1}]', mode='fits').python
        self.ax.clear()
        vmin = self.get_vmin()
        vmax = self.get_vmax()
        xlabel = 'X pixel (from 1 to NAXIS1)'
        ylabel = 'Y pixel (from 1 to NAXIS2)'
        self.image_review, _, _ = imshow(self.fig, self.ax, self.data[self.region], colorbar=False,
                                         xlabel=xlabel, ylabel=ylabel,
                                         vmin=vmin, vmax=vmax)
        self.image_review.set_extent([jmin + 0.5, jmax + 1.5, imin + 0.5, imax + 1.5])
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        for xcr, ycr in zip(xcr_list, ycr_list):
            xcr += 1  # from index to pixel
            ycr += 1  # from index to pixel
            self.ax.plot([xcr - 0.5, xcr + 0.5], [ycr + 0.5, ycr - 0.5], 'r-')
            self.ax.plot([xcr - 0.5, xcr + 0.5], [ycr - 0.5, ycr + 0.5], 'r-')
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_title(f"Cosmic ray #{self.cr_index}/{self.num_features}")
        if self.first_plot:
            self.first_plot = False
            self.fig.tight_layout()
        self.canvas.draw()

    def set_vmin(self):
        old_vmin = self.get_vmin()
        new_vmin = simpledialog.askfloat("Set vmin", "Enter new vmin:", initialvalue=old_vmin)
        if new_vmin is None:
            return
        self.vmin_button.config(text=f"vmin: {new_vmin:.2f}")
        self.image_review.set_clim(vmin=new_vmin)
        self.canvas.draw()

    def set_vmax(self):
        old_vmax = self.get_vmax()
        new_vmax = simpledialog.askfloat("Set vmax", "Enter new vmax:", initialvalue=old_vmax)
        if new_vmax is None:
            return
        self.vmax_button.config(text=f"vmax: {new_vmax:.2f}")
        self.image_review.set_clim(vmax=new_vmax)
        self.canvas.draw()

    def get_vmin(self):
        return float(self.vmin_button.cget("text").split(":")[1])

    def get_vmax(self):
        return float(self.vmax_button.cget("text").split(":")[1])

    def set_minmax(self):
        vmin_new = np.min(self.data[self.region])
        vmax_new = np.max(self.data[self.region])
        self.vmin_button.config(text=f"vmin: {vmin_new:.2f}")
        self.vmax_button.config(text=f"vmax: {vmax_new:.2f}")
        self.image_review.set_clim(vmin=vmin_new)
        self.image_review.set_clim(vmax=vmax_new)
        self.canvas.draw()

    def set_zscale(self):
        vmin_new, vmax_new = zscale(self.data[self.region])
        self.vmin_button.config(text=f"vmin: {vmin_new:.2f}")
        self.vmax_button.config(text=f"vmax: {vmax_new:.2f}")
        self.image_review.set_clim(vmin=vmin_new)
        self.image_review.set_clim(vmax=vmax_new)
        self.canvas.draw()

    def set_ndeg(self):
        new_degree = simpledialog.askinteger("Set degree", "Enter new degree (min=0):",
                                             initialvalue=self.degree, minvalue=0)
        if new_degree is None:
            return
        new_npoints = simpledialog.askinteger("Set n", f"Enter new n (min={2*new_degree}):",
                                              initialvalue=self.npoints, minvalue=2*new_degree)
        if new_npoints is None:
            return
        self.degree = new_degree
        self.npoints = new_npoints
        self.ndeg_label.config(text=f"deg={self.degree}, n={self.npoints}")

    def x_interp(self):
        print(f"X-interpolation of cosmic ray {self.cr_index}")
        ycr_list, xcr_list = np.where(self.cr_labels == self.cr_index)
        ycr_min = np.min(ycr_list)
        ycr_max = np.max(ycr_list)
        xfit_all = []
        yfit_all = []
        for ycr in range(ycr_min, ycr_max + 1):
            xmarked = xcr_list[np.where(ycr_list == ycr)]
            jmin = np.min(xmarked)
            jmax = np.max(xmarked)
            # mark intermediate pixels too
            for ix in range(jmin, jmax + 1):
                self.cr_labels[ycr, ix] = self.cr_index
            xmarked = xcr_list[np.where(ycr_list == ycr)]
            xfit = []
            zfit = []
            for i in range(jmin - self.npoints, jmin):
                if 0 <= i < self.data.shape[1]:
                    xfit.append(i)
                    xfit_all.append(i)
                    yfit_all.append(ycr)
                    zfit.append(self.data[ycr, i])
            for i in range(jmax + 1, jmax + 1 + self.npoints):
                if 0 <= i < self.data.shape[1]:
                    xfit.append(i)
                    xfit_all.append(i)
                    yfit_all.append(ycr)
                    zfit.append(self.data[ycr, i])
            if len(xfit) > self.degree:
                p = np.polyfit(xfit, zfit, self.degree)
                for i in range(jmin, jmax + 1):
                    if 0 <= i < self.data.shape[1]:
                        self.data[ycr, i] = np.polyval(p, i)
            else:
                print(f"Not enough points to fit at y={ycr+1}")
        self.restore_cr_button.config(state=tk.NORMAL)
        self.remove_crosses_button.config(state=tk.DISABLED)
        self.x_interp_button.config(state=tk.DISABLED)
        self.y_interp_button.config(state=tk.DISABLED)
        self.s_interp_button.config(state=tk.DISABLED)
        self.update_display()
        if len(xfit_all) > 0:
            self.ax.plot(np.array(xfit_all) + 1, np.array(yfit_all) + 1, 'mo', markersize=4)  # +1: from index to pixel
            self.canvas.draw()

    def y_interp(self):
        print(f"Y-interpolation of cosmic ray {self.cr_index}")
        ycr_list, xcr_list = np.where(self.cr_labels == self.cr_index)
        xcr_min = np.min(xcr_list)
        xcr_max = np.max(xcr_list)
        xfit_all = []
        yfit_all = []
        for xcr in range(xcr_min, xcr_max + 1):
            ymarked = ycr_list[np.where(xcr_list == xcr)]
            imin = np.min(ymarked)
            imax = np.max(ymarked)
            # mark intermediate pixels too
            for iy in range(imin, imax + 1):
                self.cr_labels[iy, xcr] = self.cr_index
            ymarked = ycr_list[np.where(xcr_list == xcr)]
            yfit = []
            zfit = []
            for i in range(imin - self.npoints, imin):
                if 0 <= i < self.data.shape[0]:
                    yfit.append(i)
                    yfit_all.append(i)
                    xfit_all.append(xcr)
                    zfit.append(self.data[i, xcr])
            for i in range(imax + 1, imax + 1 + self.npoints):
                if 0 <= i < self.data.shape[0]:
                    yfit.append(i)
                    yfit_all.append(i)
                    xfit_all.append(xcr)
                    zfit.append(self.data[i, xcr])
            if len(yfit) > self.degree:
                p = np.polyfit(yfit, zfit, self.degree)
                for i in range(imin, imax + 1):
                    if 0 <= i < self.data.shape[1]:
                        self.data[i, xcr] = np.polyval(p, i)
            else:
                print(f"Not enough points to fit at x={xcr+1}")
        self.restore_cr_button.config(state=tk.NORMAL)
        self.remove_crosses_button.config(state=tk.DISABLED)
        self.x_interp_button.config(state=tk.DISABLED)
        self.y_interp_button.config(state=tk.DISABLED)
        self.s_interp_button.config(state=tk.DISABLED)
        self.update_display()
        if len(xfit_all) > 0:
            self.ax.plot(np.array(xfit_all) + 1, np.array(yfit_all) + 1, 'mo', markersize=4)  # +1: from index to pixel
            self.canvas.draw()

    def s_interp(self):
        print(f"S-interpolation of cosmic ray {self.cr_index}")
        ycr_list, xcr_list = np.where(self.cr_labels == self.cr_index)
        self.update_display()

    def remove_crosses(self):
        ycr_list, xcr_list = np.where(self.cr_labels == self.cr_index)
        for iy, ix in zip(ycr_list, xcr_list):
            self.cr_labels[iy, ix] = 0
        print(f"Removed all pixels of cosmic ray {self.cr_index}")
        self.remove_crosses_button.config(state=tk.DISABLED)
        self.x_interp_button.config(state=tk.DISABLED)
        self.y_interp_button.config(state=tk.DISABLED)
        self.s_interp_button.config(state=tk.DISABLED)
        self.update_display()

    def restore_cr(self):
        ycr_list, xcr_list = np.where(self.cr_labels == self.cr_index)
        for iy, ix in zip(ycr_list, xcr_list):
            self.data[iy, ix] = self.data_original[iy, ix]
            self.x_interp_button.config(state=tk.NORMAL)
            self.y_interp_button.config(state=tk.NORMAL)
            self.s_interp_button.config(state=tk.NORMAL)
        print(f"Restored all pixels of cosmic ray {self.cr_index}")
        self.remove_crosses_button.config(state=tk.NORMAL)
        self.restore_cr_button.config(state=tk.DISABLED)
        self.update_display()

    def continue_cr(self):
        self.cr_index += 1
        if self.cr_index > self.num_features:
            self.cr_index = 1
        self.first_plot = True
        self.restore_cr_button.config(state=tk.DISABLED)
        self.x_interp_button.config(state=tk.NORMAL)
        self.y_interp_button.config(state=tk.NORMAL)
        self.s_interp_button.config(state=tk.NORMAL)
        self.update_display()

    def exit_review(self):
        self.review_window.destroy()

    def on_key(self, event):
        if event.key == 'q':
            pass  # Ignore the "q" key to prevent closing the window
        elif event.key == 'r':
            if self.restore_cr_button.cget("state") != "disabled":
                self.restore_cr()
        elif event.key == 'x':
            if self.x_interp_button.cget("state") != "disabled":
                self.x_interp()
        elif event.key == 'y':
            if self.y_interp_button.cget("state") != "disabled":
                self.y_interp()
        elif event.key == 's':
            if self.s_interp_button.cget("state") != "disabled":
                self.s_interp()
        elif event.key == 'right' or event.key == 'c':
            self.continue_cr()
        elif event.key == ',':
            self.set_minmax()
        elif event.key == '/':
            self.set_zscale()
        elif event.key == 'e':
            self.exit_review()
        else:
            print(f"Key pressed: {event.key}")

    def on_click(self, event):
        if event.inaxes:
            x, y = event.xdata, event.ydata
            print(f"Clicked at image coordinates: ({x:.2f}, {y:.2f})")
            ix = int(x+0.5) - 1  # from pixel to index
            iy = int(y+0.5) - 1  # from pixel to index
            if int(self.cr_labels[iy, ix]) == self.cr_index:
                self.cr_labels[iy, ix] = 0
                print(f"Pixel ({ix+1}, {iy+1}) unmarked as cosmic ray.")
            else:
                self.cr_labels[iy, ix] = self.cr_index
                print(f"Pixel ({ix+1}, {iy+1}) marked as cosmic ray.")
            xcr_list, ycr_list = np.where(self.cr_labels == self.cr_index)
            if len(xcr_list) == 0:
                self.x_interp_button.config(state=tk.DISABLED)
                self.y_interp_button.config(state=tk.DISABLED)
                self.s_interp_button.config(state=tk.DISABLED)
                self.remove_crosses_button.config(state=tk.DISABLED)
            else:
                self.x_interp_button.config(state=tk.NORMAL)
                self.y_interp_button.config(state=tk.NORMAL)
                self.s_interp_button.config(state=tk.NORMAL)
                self.remove_crosses_button.config(state=tk.NORMAL)
            # Update the display to reflect the change
            self.update_display()


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
        la_cleaned_data, mask = cosmicray_lacosmic(self.data, sigclip=4.5, sigfrac=0.3, objlim=5.0, verbose=True)
        ReviewCosmicRay(self.root, self.data, la_cleaned_data, mask)
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
    CosmicRayCleanerApp(root, args.fits_file, args.extension)

    # Execute
    root.mainloop()


if __name__ == "__main__":
    main()
