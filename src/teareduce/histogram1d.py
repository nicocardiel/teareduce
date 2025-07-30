#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Auxiliary function to display 1D histograms computed with numpy"""

import numpy as np


def plot_hist_step(ax, bins, h, color='C0', alpha=1.0, fill_color=None, fill_alpha=0.4):
    """Plot histogram already computed.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    bins : np.ndarray
        Array of bin edges.
    h : np.ndarray
        Array of histogram values.
    color : str, optional
        Color of the histogram line, by default 'C0'.
    alpha : float, optional
        Transparency of the histogram line, by default 1.0.
    fill_color : str, optional
        Color to fill the histogram area, by default None (no fill).
    fill_alpha : float, optional
        Transparency of the filled area, by default 0.4.
    """
    # bin centers
    xdum = (bins[:-1] + bins[1:]) / 2
    ax.step(xdum, h, where='mid')
    # draw vertical lines at the edges
    ax.plot([bins[0], bins[0], xdum[0]], [0, h[0], h[0]], alpha=alpha, color=f'{color}', linestyle='-')
    ax.plot([xdum[-1], bins[-1], bins[-1]], [h[-1], h[-1], 0], alpha=alpha, color=f'{color}', linestyle='-')
    # fill area under the histogram
    if fill_color is not None:
        ax.fill_between(np.concatenate((np.array([bins[0]]), xdum, np.array([bins[-1]]))),
                        np.concatenate((np.array([h[0]]), h, np.array([h[-1]]))),
                        step='mid', alpha=fill_alpha, color=f'{fill_color}')


def hist_step(ax, data, bins, color='C0', alpha=1.0, fill_color=None, fill_alpha=0.4):
    """Compute and plot histogram of data.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    data : np.ndarray
        Data to compute the histogram from.
    bins : int or np.ndarray
        Number of bins or array of bin edges.
    color : str, optional
        Color of the histogram line, by default 'C0'.
    alpha : float, optional
        Transparency of the histogram line, by default 1.0.
    fill_color : str, optional
        Color to fill the histogram area, by default None (no fill).
    fill_alpha : float, optional
        Transparency of the filled area, by default 0.4.

    Returns
    -------
    h : np.ndarray
        Histogram values.
    edges : np.ndarray
        Bin edges of the histogram.
    """

    if isinstance(bins, int):
        bins = np.linspace(np.min(data), np.max(data), bins + 1)
    elif isinstance(bins, np.ndarray):
        pass
    else:
        raise ValueError(f'Unexpected {bins=}')
    h, edges = np.histogram(data, bins=bins)
    plot_hist_step(ax, bins, h, color=color, alpha=alpha, fill_color=fill_color, fill_alpha=fill_alpha)

    return h, edges
