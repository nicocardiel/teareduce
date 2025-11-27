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

from ..sliceregion import SliceRegion2D
from ..zscale import zscale


# The functionality defined here is used in multiple classes
class ImageDisplay:
    def set_vmin(self):
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
        if hasattr(self, 'auxdata') and self.auxdata is not None:
            self.image_aux.set_clim(vmin=new_vmin)
        self.canvas.draw_idle()

    def set_vmax(self):
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
        if hasattr(self, 'auxdata') and self.auxdata is not None:
            self.image_aux.set_clim(vmax=new_vmax)
        self.canvas.draw_idle()

    def get_vmin(self):
        return float(self.vmin_button.cget("text").split(":")[1])

    def get_vmax(self):
        return float(self.vmax_button.cget("text").split(":")[1])

    def get_displayed_region(self):
        if hasattr(self, 'ax'):
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
            region = self.region = SliceRegion2D(
                f'[{xmin}:{xmax}, {ymin}:{ymax}]', mode='fits'
            ).python
        elif hasattr(self, 'region'):
            region = self.region
        else:
            raise AttributeError("No axis or region defined for set_minmax.")
        return region

    def set_minmax(self):
        region = self.get_displayed_region()
        vmin_new = np.min(self.data[region])
        vmax_new = np.max(self.data[region])
        self.vmin_button.config(text=f"vmin: {vmin_new:.2f}")
        self.vmax_button.config(text=f"vmax: {vmax_new:.2f}")
        self.image.set_clim(vmin=vmin_new)
        self.image.set_clim(vmax=vmax_new)
        if hasattr(self, 'auxdata') and self.auxdata is not None:
            self.image_aux.set_clim(vmin=vmin_new)
            self.image_aux.set_clim(vmax=vmax_new)
        self.canvas.draw_idle()

    def set_zscale(self):
        region = self.get_displayed_region()
        vmin_new, vmax_new = zscale(self.data[region])
        self.vmin_button.config(text=f"vmin: {vmin_new:.2f}")
        self.vmax_button.config(text=f"vmax: {vmax_new:.2f}")
        self.image.set_clim(vmin=vmin_new)
        self.image.set_clim(vmax=vmax_new)
        if hasattr(self, 'auxdata') and self.auxdata is not None:
            self.image_aux.set_clim(vmin=vmin_new)
            self.image_aux.set_clim(vmax=vmax_new)
        self.canvas.draw_idle()
