#
# Copyright 2023-2025 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Module providing functions for cleaning astronomical images.
Includes functions for interpolating over bad pixels,
applying the L.A.Cosmic algorithm to detect cosmic ray pixels.
"""

from .interpolate import interpolate
from .lacosmicpad import lacosmicpad
from .mergemasks import merge_peak_tail_masks

__all__ = ["interpolate", "lacosmicpad", "merge_peak_tail_masks"]
