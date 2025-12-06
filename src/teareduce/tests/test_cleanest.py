# -*- coding: utf-8 -*-
#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
"""Tests for the cleanest module."""

import numpy as np


from ..cleanest.interpolate import interpolate


def test_cleanest_no_cr():
    """Test cleanest function with no cosmic rays."""
    data = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]], dtype=float)
    mask_crfound = np.zeros_like(data, dtype=bool)

    for interp_method in ['x', 'y', 's', 'd', 'm']:
        cleaned_data, mask_fixed = interpolate(data, mask_crfound, interp_method=interp_method, npoints=2, degree=1)

        assert np.array_equal(cleaned_data, data), "Data should remain unchanged when no cosmic rays are present."
        assert np.array_equal(mask_fixed, mask_crfound), "Mask should remain unchanged when no cosmic rays are present."


def test_cleanest_interpolation_x():
    """Test cleanest function with interpolation in X direction."""
    data = np.array([[1, 1, 2, 1, 1],
                     [2, 2, 3, 2, 2],
                     [3, 3, 4, 3, 3]], dtype=float)
    mask_crfound = np.array([[False, False, True, False, False],
                             [False, False, True, False, False],
                             [False, False, True, False, False]], dtype=bool)

    cleaned_data, mask_fixed = interpolate(data, mask_crfound,
                                        interp_method='x', npoints=2, degree=1)

    expected_data = np.array([[1, 1, 1, 1, 1],
                              [2, 2, 2, 2, 2],
                              [3, 3, 3, 3, 3]], dtype=float)

    assert np.allclose(cleaned_data, expected_data), "Interpolation in X direction failed."
    assert np.array_equal(mask_fixed, mask_crfound), "Mask should remain unchanged after interpolation."


def test_cleanest_interpolation_y():
    """Test cleanest function with interpolation in Y direction."""
    data = np.array([[1, 2, 3],
                     [1, 2, 3],
                     [2, 3, 4],
                     [1, 2, 3],
                     [1, 2, 3]], dtype=float)
    mask_crfound = np.array([[False, False, False],
                             [False, False, False],
                             [True, True, True],
                             [False, False, False],
                             [False, False, False]], dtype=bool)

    cleaned_data, mask_fixed = interpolate(data, mask_crfound,
                                           interp_method='y', npoints=2, degree=1)

    expected_data = np.array([[1, 2, 3],
                              [1, 2, 3],
                              [1, 2, 3],
                              [1, 2, 3],
                              [1, 2, 3]], dtype=float)
    assert np.allclose(cleaned_data, expected_data), "Interpolation in Y direction failed."
    assert np.array_equal(mask_fixed, mask_crfound), "Mask should remain unchanged after interpolation."


def test_cleanest_interpolation_surface():
    """Test cleanest function with surface interpolation."""
    data = np.array([[1, 2, 3],
                     [4, 100, 6],
                     [7, 8, 9]], dtype=float)
    mask_crfound = np.array([[False, False, False],
                             [False, True, False],
                             [False, False, False]], dtype=bool)

    cleaned_data, mask_fixed = interpolate(data, mask_crfound,
                                           interp_method='s', npoints=1)

    expected_data = np.array([[1, 2, 3],
                              [4, 5, 6],
                              [7, 8, 9]], dtype=float)
    assert np.allclose(cleaned_data, expected_data), "Surface interpolation failed."
    assert np.array_equal(mask_fixed, mask_crfound), "Mask should remain unchanged after interpolation."


def test_cleanest_interpolation_median():
    """Test cleanest function with median border pixel interpolation."""
    data = np.array([[1, 2, 3],
                     [4, 100, 6],
                     [7, 8, 9]], dtype=float)
    mask_crfound = np.array([[False, False, False],
                             [False, True, False],
                             [False, False, False]], dtype=bool)

    cleaned_data, mask_fixed = interpolate(data, mask_crfound,
                                           interp_method='d', npoints=1)

    expected_data = np.array([[1, 2, 3],
                              [4, 5, 6],
                              [7, 8, 9]], dtype=float)
    assert np.allclose(cleaned_data, expected_data), "Median border pixel interpolation failed."
    assert np.array_equal(mask_fixed, mask_crfound), "Mask should remain unchanged after interpolation."


def test_cleanest_interpolation_mean():
    """Test cleanest function with mean border pixel interpolation."""
    data = np.array([[1, 2, 3],
                     [4, 100, 6],
                     [7, 8, 9]], dtype=float)
    mask_crfound = np.array([[False, False, False],
                             [False, True, False],
                             [False, False, False]], dtype=bool)

    cleaned_data, mask_fixed = interpolate(data, mask_crfound,
                                           interp_method='m', npoints=1)

    expected_data = np.array([[1, 2, 3],
                              [4, 5, 6],
                              [7, 8, 9]], dtype=float)
    assert np.allclose(cleaned_data, expected_data), "Mean border pixel interpolation failed."
    assert np.array_equal(mask_fixed, mask_crfound), "Mask should remain unchanged after interpolation."
