#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Set min/max values for image display."""

import numpy as np


# This function is defined here because its functionality is used in
# multiple classes
def set_minmax(obj):
    vmin_new = np.min(obj.data[obj.region])
    vmax_new = np.max(obj.data[obj.region])
    obj.vmin_button.config(text=f"vmin: {vmin_new:.2f}")
    obj.vmax_button.config(text=f"vmax: {vmax_new:.2f}")
    obj.image.set_clim(vmin=vmin_new)
    obj.image.set_clim(vmax=vmax_new)
    obj.canvas.draw()
