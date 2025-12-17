#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Base class for image display with min/max and zscale controls."""

from tkinter import simpledialog
import numpy as np
from rich import print

from ..sliceregion import SliceRegion2D
from ..zscale import zscale


# The functionality defined here is used in multiple classes
class ImageDisplay:
    """Class to handle image display with min/max and zscale controls.

    Methods
    -------
    set_vmin()
        Prompt user to set a new minimum display value (vmin).
    set_vmax()
        Prompt user to set a new maximum display value (vmax).
    get_vmin()
        Get the current minimum display value (vmin).
    get_vmax()
        Get the current maximum display value (vmax).
    get_displayed_region()
        Get the currently displayed region of the image.
    set_minmax()
        Set vmin and vmax based on the currently displayed region.
    set_zscale()
        Set vmin and vmax using zscale on the currently displayed region.

    Attributes
    ----------
    vmin_button : tkinter.Button
        Button to display and set the minimum display value (vmin).
    vmax_button : tkinter.Button
        Button to display and set the maximum display value (vmax).
    image : matplotlib.image.AxesImage
        The main image being displayed.
    image_aux : matplotlib.image.AxesImage, optional
        An auxiliary image being displayed (if any).
    canvas : matplotlib.backends.backend_tkagg.FigureCanvasTkAgg
        The canvas on which the image is drawn.

    Notes
    -----
    This class is intented to be used as a parent class for different
    classes that display images and need functionality to adjust the
    display limits (vmin and vmax) interactively.

    This class assumes that the image data is stored in `self.data` and that
    the displayed region can be determined from either the axes limits or a
    predefined region attribute.
    """

    def set_vmin(self):
        """Prompt user to set a new minimum display value (vmin)."""
        old_vmin = self.get_vmin()
        old_vmax = self.get_vmax()
        new_vmin = simpledialog.askfloat("Set vmin", "Enter new vmin:", initialvalue=old_vmin)
        if new_vmin is None:
            return
        if new_vmin >= old_vmax:
            print("Error: vmin must be less than vmax.")
            return
        self.vmin_button.config(text=f"vmin: {new_vmin:.2f}")
        self.image.set_clim(vmin=new_vmin)
        if hasattr(self, "image_aux"):
            self.image_aux.set_clim(vmin=new_vmin)
        self.canvas.draw_idle()

    def set_vmax(self):
        """Prompt user to set a new maximum display value (vmax)."""
        old_vmin = self.get_vmin()
        old_vmax = self.get_vmax()
        new_vmax = simpledialog.askfloat("Set vmax", "Enter new vmax:", initialvalue=old_vmax)
        if new_vmax is None:
            return
        if new_vmax <= old_vmin:
            print("Error: vmax must be greater than vmin.")
            return
        self.vmax_button.config(text=f"vmax: {new_vmax:.2f}")
        self.image.set_clim(vmax=new_vmax)
        if hasattr(self, "image_aux"):
            self.image_aux.set_clim(vmax=new_vmax)
        self.canvas.draw_idle()

    def get_vmin(self):
        """Get the current minimum display value (vmin)."""
        return float(self.vmin_button.cget("text").split(":")[1])

    def get_vmax(self):
        """Get the current maximum display value (vmax)."""
        return float(self.vmax_button.cget("text").split(":")[1])

    def get_displayed_region(self):
        """Get the currently displayed region of the image."""
        if hasattr(self, "ax"):
            xmin, xmax = self.ax.get_xlim()
            xmin = int(xmin + 0.5)
            if xmin < 1:
                xmin = 1
            xmax = int(xmax + 0.5)
            if xmax > self.data.shape[1]:
                xmax = self.data.shape[1]
            ymin, ymax = self.ax.get_ylim()
            ymin = int(ymin + 0.5)
            if ymin < 1:
                ymin = 1
            ymax = int(ymax + 0.5)
            if ymax > self.data.shape[0]:
                ymax = self.data.shape[0]
            print(f"Setting min/max using axis limits: x=({xmin:.2f}, {xmax:.2f}), y=({ymin:.2f}, {ymax:.2f})")
            region = self.region = SliceRegion2D(f"[{xmin}:{xmax}, {ymin}:{ymax}]", mode="fits").python
        elif hasattr(self, "region"):
            region = self.region
        else:
            raise AttributeError("No axis or region defined for set_minmax.")
        return region

    def set_minmax(self):
        """Set vmin and vmax based on the currently displayed region."""
        region = self.get_displayed_region()
        vmin_new = np.min(self.data[region])
        vmax_new = np.max(self.data[region])
        self.vmin_button.config(text=f"vmin: {vmin_new:.2f}")
        self.vmax_button.config(text=f"vmax: {vmax_new:.2f}")
        self.image.set_clim(vmin=vmin_new)
        self.image.set_clim(vmax=vmax_new)
        if hasattr(self, "image_aux"):
            self.image_aux.set_clim(vmin=vmin_new)
            self.image_aux.set_clim(vmax=vmax_new)
        self.canvas.draw_idle()

    def set_zscale(self):
        """Set vmin and vmax using zscale on the currently displayed region."""
        region = self.get_displayed_region()
        vmin_new, vmax_new = zscale(self.data[region])
        self.vmin_button.config(text=f"vmin: {vmin_new:.2f}")
        self.vmax_button.config(text=f"vmax: {vmax_new:.2f}")
        self.image.set_clim(vmin=vmin_new)
        self.image.set_clim(vmax=vmax_new)
        if hasattr(self, "image_aux"):
            self.image_aux.set_clim(vmin=vmin_new)
            self.image_aux.set_clim(vmax=vmax_new)
        self.canvas.draw_idle()
