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


def test_wrong_number_of_dimensions():
    """Test the creation of a SliceRegion with wrong slices."""

    try:
        SliceRegion1D(np.s_[1:3, 2:4], mode='python')
        assert False, "Expected ValueError for 1D slice with 2D input"
    except ValueError:
        pass

    try:
        SliceRegion2D(np.s_[1:3], mode='python')
        assert False, "Expected ValueError for 2D slice with 1D input"
    except ValueError:
        pass

    try:
        SliceRegion3D(np.s_[1:3, 2:4], mode='python')
        assert False, "Expected ValueError for 3D slice with 2D input"
    except ValueError:
        pass


def test_wrong_limits_order():
    """Test the creation of a SliceRegion with wrong limits."""

    try:
        SliceRegion1D(np.s_[10:5], mode='python')
        assert False, "Expected ValueError for 1D slice with start > stop"
    except ValueError:
        pass

    try:
        SliceRegion2D(np.s_[1:3, 5:2], mode='python')
        assert False, "Expected ValueError for 2D slice with start > stop in second dimension"
    except ValueError:
        pass

    try:
        SliceRegion3D(np.s_[1:3, 2:4, 6:1], mode='python')
        assert False, "Expected ValueError for 3D slice with start > stop in third dimension"
    except ValueError:
        pass


def test_limits_out_of_range():
    """Test the creation of a SliceRegion with limits out of range."""

    try:
        SliceRegion1D(np.s_[-1:5], mode='python', naxis1=10)
        assert False, "Expected ValueError for 1D slice with negative start"
    except ValueError:
        pass

    try:
        SliceRegion2D(np.s_[1:3, 2:25], mode='python', naxis1=10, naxis2=20)
        assert False, "Expected ValueError for 2D slice with stop > naxis in second dimension"
    except ValueError:
        pass

    try:
        SliceRegion3D(np.s_[1:3, 2:4, 3:35], mode='python', naxis1=10, naxis2=20, naxis3=30)
        assert False, "Expected ValueError for 3D slice with stop > naxis in third dimension"
    except ValueError:
        pass
