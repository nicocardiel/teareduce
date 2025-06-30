# -*- coding: utf-8 -*-
#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
"""Tests for the SliceRegion class."""

import numpy as np

from ..sliceregion import SliceRegion1D, SliceRegion2D, SliceRegion3D


def test_slice_region_creation():
    """Test the creation of a SliceRegion."""

    region1d = SliceRegion1D(np.s_[1:10], mode='python')
    assert region1d.fits == slice(2, 10, None)
    assert region1d.python == slice(1, 10, None)
    assert region1d.fits_section == '[2:10]'

    region2d = SliceRegion2D(np.s_[1:10, 2:20], mode='python')
    assert region2d.fits == (slice(3, 20, None), slice(2, 10, None))
    assert region2d.python == (slice(1, 10, None), slice(2, 20, None))
    assert region2d.fits_section == '[3:20,2:10]'

    region3d = SliceRegion3D(np.s_[1:10, 2:20, 3:30], mode='python')
    assert region3d.fits == (slice(4, 30, None), slice(3, 20, None), slice(2, 10, None))
    assert region3d.python == (slice(1, 10, None), slice(2, 20, None), slice(3, 30, None))
    assert region3d.fits_section == '[4:30,3:20,2:10]'

def test_slice_values():
    """Test the values of the slices in different modes."""

    array1d = np.arange(10)

    region1d = SliceRegion1D(np.s_[1:3], mode='python')
    assert np.all(array1d[region1d.python] == np.array([1, 2]))

    array2d = np.arange(12).reshape(3, 4)
    region2d = SliceRegion2D(np.s_[1:3, 2:3], mode='python')
    assert np.all(array2d[region2d.python] == np.array([[6], [10]]))

    array3d = np.arange(24).reshape(3, 4, 2)
    region3d = SliceRegion3D(np.s_[1:3, 2:4, 1:2], mode='python')
    assert np.all(array3d[region3d.python] == np.array([[[13], [15]], [[21], [23]]]))
