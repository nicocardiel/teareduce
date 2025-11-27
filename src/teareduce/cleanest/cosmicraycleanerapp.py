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
from tkinter import messagebox
import sys

from astropy.io import fits
from ccdproc import cosmicray_lacosmic
import matplotlib.pyplot as plt
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import os
from rich import print
from scipy import ndimage

from .definitions import lacosmic_default_dict
from .definitions import MAX_PIXEL_DISTANCE_TO_CR
from .find_closest_true import find_closest_true
from .interpolation_a import interpolation_a
from .interpolation_x import interpolation_x
from .interpolation_y import interpolation_y
from .interpolationeditor import InterpolationEditor
from .imagedisplay import ImageDisplay
from .parametereditor import ParameterEditor
from .reviewcosmicray import ReviewCosmicRay

from ..imshow import imshow
from ..sliceregion import SliceRegion2D
from ..zscale import zscale

import matplotlib
matplotlib.use("TkAgg")


class CosmicRayCleanerApp(ImageDisplay):
    """Main application class for cosmic ray cleaning."""

    def __init__(self, root, input_fits, extension=0, auxfile=None, extension_auxfile=0):
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
        """
        self.root = root
        self.root.title("Cosmic Ray Cleaner")
        self.root.geometry("800x700+50+0")
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
        self.create_widgets()
        self.cleandata_lacosmic = None
        self.cr_labels = None
        self.num_features = 0
        self.working_in_review_window = False

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
        self.mask_crfound = np.zeros(self.data.shape, dtype=bool)
        naxis2, naxis1 = self.data.shape
        self.region = SliceRegion2D(f'[1:{naxis1}, 1:{naxis2}]', mode='fits').python
        # Read auxiliary file if provided
        if self.auxfile is not None:
            try:
                with fits.open(self.auxfile, mode='readonly') as hdul_aux:
                    self.auxdata = hdul_aux[self.extension_auxfile].data
                    if self.auxdata.shape != self.data.shape:
                        print(f"data shape...: {self.data.shape}")
                        print(f"auxdata shape: {self.auxdata.shape}")
                        raise ValueError("Auxiliary file has different shape.")
            except Exception as e:
                sys.exit(f"Error loading auxiliary FITS file: {e}")

    def save_fits_file(self):
        base, ext = os.path.splitext(self.input_fits)
        suggested_name = f"{base}_cleaned"
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
            self.ax.set_title(os.path.basename(self.output_fits))
            self.canvas.draw()
            self.input_fits = os.path.basename(self.output_fits)
            self.save_button.config(state=tk.DISABLED)
        except Exception as e:
            print(f"Error saving FITS file: {e}")

    def create_widgets(self):
        # Row 1 of buttons
        self.button_frame1 = tk.Frame(self.root)
        self.button_frame1.pack(pady=5)
        self.run_lacosmic_button = tk.Button(self.button_frame1, text="Run L.A.Cosmic", command=self.run_lacosmic)
        self.run_lacosmic_button.pack(side=tk.LEFT, padx=5)
        if self.overplot_cr_pixels:
            self.overplot_cr_button = tk.Button(self.button_frame1, text="CR overlay: On",
                                                command=self.toggle_cr_overlay)
        else:
            self.overplot_cr_button = tk.Button(self.button_frame1, text="CR overlay: Off",
                                                command=self.toggle_cr_overlay)
        self.overplot_cr_button.pack(side=tk.LEFT, padx=5)
        self.apply_lacosmic_button = tk.Button(self.button_frame1, text="Replace all detected CRs",
                                               command=self.apply_lacosmic)
        self.apply_lacosmic_button.pack(side=tk.LEFT, padx=5)
        self.apply_lacosmic_button.config(state=tk.DISABLED)  # Initially disabled
        self.examine_detected_cr_button = tk.Button(self.button_frame1, text="Examine detected CRs",
                                                    command=lambda: self.examine_detected_cr(1))
        self.examine_detected_cr_button.pack(side=tk.LEFT, padx=5)
        self.examine_detected_cr_button.config(state=tk.DISABLED)  # Initially disabled

        # Row 2 of buttons
        self.button_frame2 = tk.Frame(self.root)
        self.button_frame2.pack(pady=5)
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

        # Figure
        self.fig, self.ax = plt.subplots(figsize=(7, 5.5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        # The next two instructions prevent a segmentation fault when pressing "q"
        self.canvas.mpl_disconnect(self.canvas.mpl_connect("key_press_event", key_press_handler))
        self.canvas.mpl_connect("key_press_event", self.on_key)
        self.canvas.mpl_connect("button_press_event", self.on_click)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Matplotlib toolbar
        self.toolbar_frame = tk.Frame(self.root)
        self.toolbar_frame.pack(fill=tk.X, expand=False, pady=5)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()

        # update the image display
        xlabel = 'X pixel (from 1 to NAXIS1)'
        ylabel = 'Y pixel (from 1 to NAXIS2)'
        extent = [0.5, self.data.shape[1] + 0.5, 0.5, self.data.shape[0] + 0.5]
        self.image, _, _ = imshow(self.fig, self.ax, self.data, vmin=vmin, vmax=vmax,
                                  title=os.path.basename(self.input_fits),
                                  xlabel=xlabel, ylabel=ylabel,
                                  extent=extent)
        self.fig.tight_layout()

    def run_lacosmic(self):
        self.run_lacosmic_button.config(state=tk.DISABLED)
        # Define parameters for L.A.Cosmic from default dictionary
        editor_window = tk.Toplevel(self.root)
        editor = ParameterEditor(
            root=editor_window,
            param_dict=self.lacosmic_params,
            window_title='Cosmic Ray Mask Generation Parameters',
            xmin=self.last_xmin,
            xmax=self.last_xmax,
            ymin=self.last_ymin,
            ymax=self.last_ymax,
            imgshape=self.data.shape
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
            self.last_xmin = updated_params['xmin']['value']
            self.last_xmax = updated_params['xmax']['value']
            self.last_ymin = updated_params['ymin']['value']
            self.last_ymax = updated_params['ymax']['value']
            # Update parameter dictionary with new values
            self.lacosmic_params = updated_params
            print("Parameters updated:")
            for key, info in self.lacosmic_params.items():
                print(f"  {key}: {info['value']}")
            # Execute L.A.Cosmic with updated parameters
            cleandata_lacosmic, mask_crfound = cosmicray_lacosmic(
                self.data,
                gain=self.lacosmic_params['gain']['value'],
                readnoise=self.lacosmic_params['readnoise']['value'],
                sigclip=self.lacosmic_params['sigclip']['value'],
                sigfrac=self.lacosmic_params['sigfrac']['value'],
                objlim=self.lacosmic_params['objlim']['value'],
                niter=self.lacosmic_params['niter']['value'],
                verbose=self.lacosmic_params['verbose']['value']
            )
            # Select the image region to process
            fits_region = f"[{updated_params['xmin']['value']}:{updated_params['xmax']['value']}"
            fits_region += f",{updated_params['ymin']['value']}:{updated_params['ymax']['value']}]"
            region = SliceRegion2D(fits_region, mode="fits").python
            self.cleandata_lacosmic = self.data.copy()
            self.cleandata_lacosmic[region] = cleandata_lacosmic[region]
            self.mask_crfound = np.zeros_like(self.data, dtype=bool)
            self.mask_crfound[region] = mask_crfound[region]
            # Process the mask: dilation and labeling
            if np.any(self.mask_crfound):
                num_cr_pixels_before_dilation = np.sum(self.mask_crfound)
                dilation = self.lacosmic_params['dilation']['value']
                if dilation > 0:
                    # Dilate the mask by the specified number of pixels
                    structure = ndimage.generate_binary_structure(2, 2)  # 8-connectivity
                    self.mask_crfound = ndimage.binary_dilation(
                        self.mask_crfound,
                        structure=structure,
                        iterations=self.lacosmic_params['dilation']['value']
                    )
                    num_cr_pixels_after_dilation = np.sum(self.mask_crfound)
                    sdum = str(num_cr_pixels_after_dilation)
                else:
                    sdum = str(num_cr_pixels_before_dilation)
                print("Number of cosmic ray pixels detected by L.A.Cosmic: "
                      f"{num_cr_pixels_before_dilation:{len(sdum)}}")
                if dilation > 0:
                    print(f"Number of cosmic ray pixels after dilation........: "
                          f"{num_cr_pixels_after_dilation:{len(sdum)}}")
                # Label connected components in the mask; note that by default,
                # structure is a cross [0,1,0;1,1,1;0,1,0], but we want to consider
                # diagonal connections too, so we define a 3x3 square.
                structure = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
                self.cr_labels, self.num_features = ndimage.label(self.mask_crfound, structure=structure)
                print(f"Number of cosmic rays features (grouped pixels)...: {self.num_features:{len(sdum)}}")
                self.apply_lacosmic_button.config(state=tk.NORMAL)
                self.examine_detected_cr_button.config(state=tk.NORMAL)
                self.update_cr_overlay()
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
        self.overplot_cr_pixels = not self.overplot_cr_pixels
        if self.overplot_cr_pixels:
            self.overplot_cr_button.config(text="CR overlay: On")
        else:
            self.overplot_cr_button.config(text="CR overlay: Off")
        self.update_cr_overlay()

    def update_cr_overlay(self):
        if self.overplot_cr_pixels:
            # Remove previous CR pixel overlay (if any)
            if hasattr(self, 'scatter_cr'):
                self.scatter_cr.remove()
                del self.scatter_cr
            # Overlay CR pixels in red
            if np.any(self.mask_crfound):
                y_indices, x_indices = np.where(self.mask_crfound)
                self.scatter_cr = self.ax.scatter(x_indices + 1, y_indices + 1, s=1, c='red', marker='o')
        else:
            # Remove CR pixel overlay
            if hasattr(self, 'scatter_cr'):
                self.scatter_cr.remove()
                del self.scatter_cr
        self.canvas.draw()

    def apply_lacosmic(self):
        if np.any(self.mask_crfound):
            # recalculate labels and number of features
            structure = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
            self.cr_labels, self.num_features = ndimage.label(self.mask_crfound, structure=structure)
            print(f"Number of cosmic ray pixels detected by L.A.Cosmic...........: {np.sum(self.mask_crfound)}")
            print(f"Number of cosmic rays (grouped pixels) detected by L.A.Cosmic: {self.num_features}")
            # Define parameters for L.A.Cosmic from default dictionary
            editor_window = tk.Toplevel(self.root)
            editor = InterpolationEditor(
                root=editor_window,
                last_dilation=self.lacosmic_params['dilation']['value'],
                auxdata=self.auxdata
            )
            # Make it modal (blocks interaction with main window)
            editor_window.transient(self.root)
            editor_window.grab_set()
            # Wait for the editor window to close
            self.root.wait_window(editor_window)
            # Get the result after window closes
            cleaning_method = editor.cleaning_method
            num_cr_cleaned = 0
            if cleaning_method is None:
                print("Interpolation method selection cancelled. No cleaning applied!")
                return
            if cleaning_method == 'lacosmic':
                # Replace all detected CR pixels with L.A.Cosmic values
                self.data[self.mask_crfound] = self.cleandata_lacosmic[self.mask_crfound]
                # update mask_fixed to include the newly fixed pixels
                self.mask_fixed[self.mask_crfound] = True
                # upate mask_crfound by eliminating the cleaned pixels
                self.mask_crfound[self.mask_crfound] = False
                num_cr_cleaned = self.num_features
            elif cleaning_method == 'auxdata':
                if self.auxdata is None:
                    print("No auxiliary data available. Cleaning skipped!")
                    return
                # Replace all detected CR pixels with auxiliary data values
                self.data[self.mask_crfound] = self.auxdata[self.mask_crfound]
                # update mask_fixed to include the newly fixed pixels
                self.mask_fixed[self.mask_crfound] = True
                # upate mask_crfound by eliminating the cleaned pixels
                self.mask_crfound[self.mask_crfound] = False
                num_cr_cleaned = self.num_features
            else:
                for i in range(1, self.num_features + 1):
                    tmp_mask_fixed = np.zeros_like(self.data, dtype=bool)
                    if cleaning_method == 'x':
                        interpolation_performed, _, _ = interpolation_x(
                            data=self.data,
                            mask_fixed=tmp_mask_fixed,
                            cr_labels=self.cr_labels,
                            cr_index=i,
                            npoints=editor.npoints,
                            degree=editor.degree
                        )
                    elif cleaning_method == 'y':
                        interpolation_performed, _, _ = interpolation_y(
                            data=self.data,
                            mask_fixed=tmp_mask_fixed,
                            cr_labels=self.cr_labels,
                            cr_index=i,
                            npoints=editor.npoints,
                            degree=editor.degree
                        )
                    elif cleaning_method == 'a-plane':
                        interpolation_performed, _, _ = interpolation_a(
                            data=self.data,
                            mask_fixed=tmp_mask_fixed,
                            cr_labels=self.cr_labels,
                            cr_index=i,
                            npoints=editor.npoints,
                            method='surface'
                        )
                    elif cleaning_method == 'a-median':
                        interpolation_performed, _, _ = interpolation_a(
                            data=self.data,
                            mask_fixed=tmp_mask_fixed,
                            cr_labels=self.cr_labels,
                            cr_index=i,
                            npoints=editor.npoints,
                            method='median'
                        )
                    else:
                        raise ValueError(f"Unknown cleaning method: {cleaning_method}")
                    if interpolation_performed:
                        num_cr_cleaned += 1
                        # update mask_fixed to include the newly fixed pixels
                        self.mask_fixed[tmp_mask_fixed] = True
                        # upate mask_crfound by eliminating the cleaned pixels
                        self.mask_crfound[tmp_mask_fixed] = False
            print(f"Number of cosmic rays identified and cleaned: {num_cr_cleaned}")
            # recalculate labels and number of features
            structure = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
            self.cr_labels, self.num_features = ndimage.label(self.mask_crfound, structure=structure)
            print(f"Remaining number of cosmic ray pixels...........: {np.sum(self.mask_crfound)}")
            print(f"Remaining number of cosmic rays (grouped pixels): {self.num_features}")
            # redraw image to show the changes
            self.image.set_data(self.data)
            self.canvas.draw()
            if num_cr_cleaned > 0:
                self.save_button.config(state=tk.NORMAL)
            if self.num_features == 0:
                self.examine_detected_cr_button.config(state=tk.DISABLED)
                self.apply_lacosmic_button.config(state=tk.DISABLED)
            self.update_cr_overlay()

    def examine_detected_cr(self, first_cr_index=1, single_cr=False, ixpix=None, iypix=None):
        self.working_in_review_window = True
        review_window = tk.Toplevel(self.root)
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
                last_dilation=self.lacosmic_params['dilation']['value']
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
                last_dilation=self.lacosmic_params['dilation']['value']
            )
        # Make it modal (blocks interaction with main window)
        review_window.transient(self.root)
        review_window.grab_set()
        self.root.wait_window(review_window)
        self.working_in_review_window = False
        # Get the result after window closes
        if review.num_cr_cleaned > 0:
            print(f"Number of cosmic rays identified and cleaned: {review.num_cr_cleaned}")
            # update mask_fixed to include the newly fixed pixels
            self.mask_fixed[review.mask_fixed] = True
            # upate mask_crfound by eliminating the cleaned pixels
            self.mask_crfound[review.mask_fixed] = False
            # recalculate labels and number of features
            structure = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
            self.cr_labels, self.num_features = ndimage.label(self.mask_crfound, structure=structure)
            print(f"Remaining number of cosmic ray pixels...........: {np.sum(self.mask_crfound)}")
            print(f"Remaining number of cosmic rays (grouped pixels): {self.num_features}")
            # redraw image to show the changes
            self.image.set_data(self.data)
            self.canvas.draw()
        if review.num_cr_cleaned > 0:
            self.save_button.config(state=tk.NORMAL)
        if self.num_features == 0:
            self.examine_detected_cr_button.config(state=tk.DISABLED)
            self.apply_lacosmic_button.config(state=tk.DISABLED)
        self.update_cr_overlay()

    def stop_app(self):
        proceed_with_stop = True
        if self.save_button['state'] == tk.NORMAL:
            print("Warning: There are unsaved changes!")
            proceed_with_stop = messagebox.askyesno(
                "Unsaved Changes",
                "You have unsaved changes.\nDo you really want to quit?",
                default=messagebox.NO
            )
        if proceed_with_stop:
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
        # ignore clicks if we are working in the review window
        if self.working_in_review_window:
            print("Currently working in review window; click ignored.")
            return

        # check the toolbar is not active
        toolbar = self.fig.canvas.toolbar
        if toolbar.mode != "":
            print(f"Toolbar mode '{toolbar.mode}' active; click ignored.")
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
                    np.argmax(self.data[imin:imax+1, jmin:jmax+1]),
                    self.data[imin:imax+1, jmin:jmax+1].shape
                )
                ixpix = ijmax[1] + jmin + 1
                iypix = ijmax[0] + imin + 1
            else:
                ixpix = None
                iypix = None
            self.examine_detected_cr(label_at_click, single_cr=True, ixpix=ixpix, iypix=iypix)
