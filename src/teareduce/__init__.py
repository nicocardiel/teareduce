#
# Copyright 2023-2024 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

from .avoid_astropy_warnings import avoid_astropy_warnings
from .cosmicrays import cr2images, apply_cr2images_ccddata, crmedian
from .ctext import ctext
from .draw_rectangle import draw_rectangle
from .imshow import imshow
from .sliceregion import SliceRegion1D, SliceRegion2D
from .statsummary import ifc_statsummary, statsummary
from .robust_std import robust_std
from .version import version
from .zscale import zscale

__version__ = version
