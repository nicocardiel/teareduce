#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Define the ReviewCosmicRay class."""

import tkinter as tk
from tkinter import simpledialog

import matplotlib.pyplot as plt
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
from rich import print

from .imagedisplay import ImageDisplay

from ..imshow import imshow
from ..sliceregion import SliceRegion2D
from ..zscale import zscale

import matplotlib
matplotlib.use("TkAgg")


class ReviewCosmicRay(ImageDisplay):
    """Class to review suspected cosmic ray pixels."""

    def __init__(self, root, data, cleandata_lacosmic, cr_labels, num_features,
                 first_cr_index=1, single_cr=False):
        """Initialize the review window.

        Parameters
        ----------
        root : tk.Tk
            The main Tkinter window.
        data : 2D numpy array
            The original image data.
        cleandata_lacosmic: 2D numpy array or None
            The cleaned image data from L.A.Cosmic.
        cr_labels : 2D numpy array
            Labels of connected cosmic ray pixel groups.
        num_features : int
            Number of connected cosmic ray pixel groups.
        first_cr_index : int, optional
            The index of the first cosmic ray to review (default is 1).
        single_cr : bool, optional
            Whether to review a single cosmic ray (default is False).
            If True, the review window will close after reviewing the
            selected first cosmic ray.
        """
        self.root = root
        self.data = data
        self.cleandata_lacosmic = cleandata_lacosmic
        self.data_original = data.copy()
        self.cr_labels = cr_labels
        self.num_features = num_features
        self.num_cr_cleaned = 0
        self.mask_fixed = np.zeros(self.data.shape, dtype=bool)  # Mask of pixels fixed during review
        self.first_plot = True
        self.degree = 1    # Degree of polynomial for interpolation
        self.npoints = 2   # Number of points at each side of the CR pixel for interpolation
        # Make a copy of the original labels to allow pixel re-marking
        self.cr_labels_original = self.cr_labels.copy()
        print(f"Number of cosmic ray pixels detected..: {np.sum(self.cr_labels > 0)}")
        print(f"Number of cosmic rays (grouped pixels): {self.num_features}")
        if self.num_features == 0:
            print('No CR hits found!')
        else:
            self.cr_index = first_cr_index
            self.single_cr = single_cr
            self.create_widgets()

    def create_widgets(self):
        self.review_window = tk.Toplevel(self.root)
        self.review_window.title("Review Cosmic Rays")
        self.review_window.geometry("800x700+100+100")

        # Row 1 of buttons
        self.button_frame1 = tk.Frame(self.review_window)
        self.button_frame1.pack(pady=5)
        self.ndeg_label = tk.Button(self.button_frame1, text=f"deg={self.degree}, n={self.npoints}",
                                    command=self.set_ndeg)
        self.ndeg_label.pack(side=tk.LEFT, padx=5)
        self.remove_crosses_button = tk.Button(self.button_frame1, text="remove all x's", command=self.remove_crosses)
        self.remove_crosses_button.pack(side=tk.LEFT, padx=5)
        self.restore_cr_button = tk.Button(self.button_frame1, text="[r]estore CR", command=self.restore_cr)
        self.restore_cr_button.pack(side=tk.LEFT, padx=5)
        self.restore_cr_button.config(state=tk.DISABLED)
        self.next_button = tk.Button(self.button_frame1, text="[c]ontinue", command=self.continue_cr)
        self.next_button.pack(side=tk.LEFT, padx=5)
        self.exit_button = tk.Button(self.button_frame1, text="[e]xit review", command=self.exit_review)
        self.exit_button.pack(side=tk.LEFT, padx=5)

        # Row 2 of buttons
        self.button_frame2 = tk.Frame(self.review_window)
        self.button_frame2.pack(pady=5)
        self.interp_x_button = tk.Button(self.button_frame2, text="[X] interp.", command=self.interp_x)
        self.interp_x_button.pack(side=tk.LEFT, padx=5)
        self.interp_y_button = tk.Button(self.button_frame2, text="[Y] interp.", command=self.interp_y)
        self.interp_y_button.pack(side=tk.LEFT, padx=5)
        # it is important to use lambda here to pass the method argument correctly
        # (avoiding the execution of the function at button creation time, which would happen
        # if we didn't use lambda; in that case, the function would be called immediately and
        # its return value (None) would be assigned to the command parameter; furthermore,
        # the function is trying to deactivate the buttons before they are created, which
        # would lead to an error; in addition, since I have two buttons calling the same function
        # with different arguments, using lambda allows to differentiate them)
        self.interp_s_button = tk.Button(self.button_frame2, text="[s]urface interp.",
                                         command=lambda: self.interp_a('surface'))
        self.interp_s_button.pack(side=tk.LEFT, padx=5)
        self.interp_m_button = tk.Button(self.button_frame2, text="[m]edian",
                                         command=lambda: self.interp_a('median'))
        self.interp_m_button.pack(side=tk.LEFT, padx=5)
        self.interp_l_button = tk.Button(self.button_frame2, text="[l]acosmic", command=self.use_lacosmic)
        self.interp_l_button.pack(side=tk.LEFT, padx=5)
        if self.cleandata_lacosmic is None:
            self.interp_l_button.config(state=tk.DISABLED)

        # Row 3 of buttons
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

        # Figure
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
        self.image, _, _ = imshow(self.fig, self.ax, self.data[self.region], colorbar=False,
                                  xlabel=xlabel, ylabel=ylabel,
                                  vmin=vmin, vmax=vmax)
        self.image.set_extent([jmin + 0.5, jmax + 1.5, imin + 0.5, imax + 1.5])
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

    def set_buttons_after_cleaning_cr(self):
        self.restore_cr_button.config(state=tk.NORMAL)
        self.remove_crosses_button.config(state=tk.DISABLED)
        self.interp_x_button.config(state=tk.DISABLED)
        self.interp_y_button.config(state=tk.DISABLED)
        self.interp_s_button.config(state=tk.DISABLED)
        self.interp_m_button.config(state=tk.DISABLED)
        self.interp_l_button.config(state=tk.DISABLED)

    def interp_x(self):
        print(f"X-interpolation of cosmic ray {self.cr_index}")
        ycr_list, xcr_list = np.where(self.cr_labels == self.cr_index)
        ycr_min = np.min(ycr_list)
        ycr_max = np.max(ycr_list)
        xfit_all = []
        yfit_all = []
        interpolation_performed = False
        for ycr in range(ycr_min, ycr_max + 1):
            xmarked = xcr_list[np.where(ycr_list == ycr)]
            if len(xmarked) > 0:
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
                            self.mask_fixed[ycr, i] = True
                    interpolation_performed = True
                else:
                    print(f"Not enough points to fit at y={ycr+1}")
                    self.update_display()
                    return
        if interpolation_performed:
            self.num_cr_cleaned += 1
        self.set_buttons_after_cleaning_cr()
        self.update_display()
        if len(xfit_all) > 0:
            self.ax.plot(np.array(xfit_all) + 1, np.array(yfit_all) + 1, 'mo', markersize=4)  # +1: from index to pixel
            self.canvas.draw()

    def interp_y(self):
        print(f"Y-interpolation of cosmic ray {self.cr_index}")
        ycr_list, xcr_list = np.where(self.cr_labels == self.cr_index)
        xcr_min = np.min(xcr_list)
        xcr_max = np.max(xcr_list)
        xfit_all = []
        yfit_all = []
        interpolation_performed = False
        for xcr in range(xcr_min, xcr_max + 1):
            ymarked = ycr_list[np.where(xcr_list == xcr)]
            if len(ymarked) > 0:
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
                            self.mask_fixed[i, xcr] = True
                    interpolation_performed = True
                else:
                    print(f"Not enough points to fit at x={xcr+1}")
                    self.update_display()
                    return
        if interpolation_performed:
            self.num_cr_cleaned += 1
        self.set_buttons_after_cleaning_cr()
        self.update_display()
        if len(xfit_all) > 0:
            self.ax.plot(np.array(xfit_all) + 1, np.array(yfit_all) + 1, 'mo', markersize=4)  # +1: from index to pixel
            self.canvas.draw()

    def interp_a(self, method):
        print(f"{method} interpolation of cosmic ray {self.cr_index}")
        ycr_list, xcr_list = np.where(self.cr_labels == self.cr_index)
        ycr_min = np.min(ycr_list)
        ycr_max = np.max(ycr_list)
        xfit_all = []
        yfit_all = []
        zfit_all = []
        # First do horizontal lines
        for ycr in range(ycr_min, ycr_max + 1):
            xmarked = xcr_list[np.where(ycr_list == ycr)]
            if len(xmarked) > 0:
                jmin = np.min(xmarked)
                jmax = np.max(xmarked)
                # mark intermediate pixels too
                for ix in range(jmin, jmax + 1):
                    self.cr_labels[ycr, ix] = self.cr_index
                xmarked = xcr_list[np.where(ycr_list == ycr)]
                for i in range(jmin - self.npoints, jmin):
                    if 0 <= i < self.data.shape[1]:
                        xfit_all.append(i)
                        yfit_all.append(ycr)
                        zfit_all.append(self.data[ycr, i])
                for i in range(jmax + 1, jmax + 1 + self.npoints):
                    if 0 <= i < self.data.shape[1]:
                        xfit_all.append(i)
                        yfit_all.append(ycr)
                        zfit_all.append(self.data[ycr, i])
                    xcr_min = np.min(xcr_list)
        # Now do vertical lines
        xcr_max = np.max(xcr_list)
        for xcr in range(xcr_min, xcr_max + 1):
            ymarked = ycr_list[np.where(xcr_list == xcr)]
            if len(ymarked) > 0:
                imin = np.min(ymarked)
                imax = np.max(ymarked)
                # mark intermediate pixels too
                for iy in range(imin, imax + 1):
                    self.cr_labels[iy, xcr] = self.cr_index
                ymarked = ycr_list[np.where(xcr_list == xcr)]
                for i in range(imin - self.npoints, imin):
                    if 0 <= i < self.data.shape[0]:
                        yfit_all.append(i)
                        xfit_all.append(xcr)
                        zfit_all.append(self.data[i, xcr])
                for i in range(imax + 1, imax + 1 + self.npoints):
                    if 0 <= i < self.data.shape[0]:
                        yfit_all.append(i)
                        xfit_all.append(xcr)
                        zfit_all.append(self.data[i, xcr])
        if method == 'surface':
            if len(xfit_all) > 3:
                # Construct the design matrix for a 2D polynomial fit to a plane,
                # where each row corresponds to a point (x, y, z) and the model
                # is z = C[0]*x + C[1]*y + C[2]
                A = np.c_[xfit_all, yfit_all, np.ones(len(xfit_all))]
                # Least squares polynomial fit
                C, _, _, _ = np.linalg.lstsq(A, zfit_all, rcond=None)
                # recompute all CR pixels to take into account "holes" between marked pixels
                ycr_list, xcr_list = np.where(self.cr_labels == self.cr_index)
                for iy, ix in zip(ycr_list, xcr_list):
                    self.data[iy, ix] = C[0] * ix + C[1] * iy + C[2]
                    self.mask_fixed[iy, ix] = True
                self.num_cr_cleaned += 1
            else:
                print("Not enough points to fit a plane")
                self.update_display()
                return
        elif method == 'median':
            # Compute median of all surrounding points
            if len(zfit_all) > 0:
                zmed = np.median(zfit_all)
                print(f"Replacing by median value: {zmed:.2f}")
                # recompute all CR pixels to take into account "holes" between marked pixels
                ycr_list, xcr_list = np.where(self.cr_labels == self.cr_index)
                for iy, ix in zip(ycr_list, xcr_list):
                    self.data[iy, ix] = zmed
                    self.mask_fixed[iy, ix] = True
                self.num_cr_cleaned += 1
        else:
            print(f"Unknown interpolation method: {method}")
            return
        self.set_buttons_after_cleaning_cr()
        self.update_display()
        if len(xfit_all) > 0:
            self.ax.plot(np.array(xfit_all) + 1, np.array(yfit_all) + 1, 'mo', markersize=4)  # +1: from index to pixel
            self.canvas.draw()

    def use_lacosmic(self):
        print(f"L.A.Cosmic interpolation of cosmic ray {self.cr_index}")
        ycr_list, xcr_list = np.where(self.cr_labels == self.cr_index)
        for iy, ix in zip(ycr_list, xcr_list):
            self.data[iy, ix] = self.cleandata_lacosmic[iy, ix]
            self.mask_fixed[iy, ix] = True
        self.num_cr_cleaned += 1
        self.set_buttons_after_cleaning_cr()
        self.update_display()

    def remove_crosses(self):
        ycr_list, xcr_list = np.where(self.cr_labels == self.cr_index)
        for iy, ix in zip(ycr_list, xcr_list):
            self.cr_labels[iy, ix] = 0
        print(f"Removed all pixels of cosmic ray {self.cr_index}")
        self.remove_crosses_button.config(state=tk.DISABLED)
        self.interp_x_button.config(state=tk.DISABLED)
        self.interp_y_button.config(state=tk.DISABLED)
        self.interp_s_button.config(state=tk.DISABLED)
        self.interp_m_button.config(state=tk.DISABLED)
        self.interp_l_button.config(state=tk.DISABLED)
        self.update_display()

    def restore_cr(self):
        ycr_list, xcr_list = np.where(self.cr_labels == self.cr_index)
        for iy, ix in zip(ycr_list, xcr_list):
            self.data[iy, ix] = self.data_original[iy, ix]
            self.mask_fixed[iy, ix] = False
            self.interp_x_button.config(state=tk.NORMAL)
            self.interp_y_button.config(state=tk.NORMAL)
            self.interp_s_button.config(state=tk.NORMAL)
            self.interp_m_button.config(state=tk.NORMAL)
            if self.cleandata_lacosmic is not None:
                self.interp_l_button.config(state=tk.NORMAL)
        print(f"Restored all pixels of cosmic ray {self.cr_index}")
        self.num_cr_cleaned -= 1
        self.remove_crosses_button.config(state=tk.NORMAL)
        self.restore_cr_button.config(state=tk.DISABLED)
        self.update_display()

    def continue_cr(self):
        if self.single_cr:
            self.exit_review()
        self.cr_index += 1
        if self.cr_index > self.num_features:
            self.exit_review()
            return  # important: do not remove (to avoid errors)
        self.first_plot = True
        self.restore_cr_button.config(state=tk.DISABLED)
        self.interp_x_button.config(state=tk.NORMAL)
        self.interp_y_button.config(state=tk.NORMAL)
        self.interp_s_button.config(state=tk.NORMAL)
        self.interp_m_button.config(state=tk.NORMAL)
        if self.cleandata_lacosmic is not None:
            self.interp_l_button.config(state=tk.NORMAL)
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
            if self.interp_x_button.cget("state") != "disabled":
                self.interp_x()
        elif event.key == 'y':
            if self.interp_y_button.cget("state") != "disabled":
                self.interp_y()
        elif event.key == 's':
            if self.interp_s_button.cget("state") != "disabled":
                self.interp_a('surface')
        elif event.key == 'm':
            if self.interp_m_button.cget("state") != "disabled":
                self.interp_a('median')
        elif event.key == 'l':
            if self.interp_l_button.cget("state") != "disabled":
                self.use_lacosmic()
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
            ix = int(x+0.5) - 1  # from pixel to index
            iy = int(y+0.5) - 1  # from pixel to index
            if int(self.cr_labels[iy, ix]) == self.cr_index:
                self.cr_labels[iy, ix] = 0
                print(f"Pixel ({ix+1}, {iy+1}) unmarked as cosmic ray.")
            else:
                self.cr_labels[iy, ix] = self.cr_index
                print(f"Pixel ({ix+1}, {iy+1}), with signal {self.data[iy, ix]}, marked as cosmic ray.")
            xcr_list, ycr_list = np.where(self.cr_labels == self.cr_index)
            if len(xcr_list) == 0:
                self.interp_x_button.config(state=tk.DISABLED)
                self.interp_y_button.config(state=tk.DISABLED)
                self.interp_s_button.config(state=tk.DISABLED)
                self.interp_m_button.config(state=tk.DISABLED)
                self.interp_l_button.config(state=tk.DISABLED)
                self.remove_crosses_button.config(state=tk.DISABLED)
            else:
                self.interp_x_button.config(state=tk.NORMAL)
                self.interp_y_button.config(state=tk.NORMAL)
                self.interp_s_button.config(state=tk.NORMAL)
                self.interp_m_button.config(state=tk.NORMAL)
                if self.cleandata_lacosmic is not None:
                    self.interp_l_button.config(state=tk.NORMAL)
                self.remove_crosses_button.config(state=tk.NORMAL)
            # Update the display to reflect the change
            self.update_display()
