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

from ..zscale import zscale


# The functionality defined here is used in multiple classes
class ImageDisplay:
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

    def set_minmax(self):
        vmin_new = np.min(self.data[self.region])
        vmax_new = np.max(self.data[self.region])
        self.vmin_button.config(text=f"vmin: {vmin_new:.2f}")
        self.vmax_button.config(text=f"vmax: {vmax_new:.2f}")
        self.image.set_clim(vmin=vmin_new)
        self.image.set_clim(vmax=vmax_new)
        self.canvas.draw()

    def set_zscale(self):
        vmin_new, vmax_new = zscale(self.data[self.region])
        self.vmin_button.config(text=f"vmin: {vmin_new:.2f}")
        self.vmax_button.config(text=f"vmax: {vmax_new:.2f}")
        self.image.set_clim(vmin=vmin_new)
        self.image.set_clim(vmax=vmax_new)
        self.canvas.draw()
