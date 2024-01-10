# -*- coding: utf-8 -*-
#
# Copyright 2015-2024 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial


def polfit_residuals(
        x, y, deg, reject=None,
        color='b', size=75,
        xlim=None, ylim=None,
        xlabel=None, ylabel=None, title=None,
        show=True,
        debugplot=False):
    """Polynomial fit with display of residuals.

    Parameters
    ----------
    x : numpy array
        X coordinates of the data being fitted (floats).
    y : numpy array
        Y coordinates of the data being fitted (floats).
    deg : int
        Degree of the fitting polynomial.
    reject : None or 1d numpy array (bool)
        If not None, it must be a boolean array indicating whether a
        particular point is rejected or not (i.e., the rejected points
        are flagged as True in this array). Rejected points are
        displayed but not used in the fit.
    color : single character or 1d numpy array of characters
        Color for all the symbols (single character) or for each
        individual symbol (array of color names with the same length as
        'x' or 'y'). If 'color' is a single character, the rejected
        points are displayed in red color, whereas when 'color' is an
        array of color names, rejected points are displayed with the
        color provided in this array.
    size : int
        Marker size for all the symbols (single character) or for each
        individual symbol (array of integers with the same length as
        'x' or 'y').
    xlim : tuple (floats)
        Plot limits in the X axis.
    ylim : tuple (floats)
        Plot limits in the Y axis.
    xlabel : string
        Character string for label in X axis.
    ylabel : string
        Character string for label in y axis.
    title : string
        Character string for graph title.
    show : bool
        If True, the function shows the displayed image. Otherwise
        plt.show() is expected to be executed outside.
    debugplot : bool
        If True, display intermediate plots and results.

    Returns
    -------
    poly : instance of Polynomial (numpy)
        Result from the polynomial fit using numpy Polynomial. Only
        points not flagged as rejected are employed in the fit.
    yres : numpy array
        Residuals from polynomial fit. Note that the residuals are
        computed for all the points, including the rejected ones. In
        this way the dimension of this array is the same as the
        dimensions of the input 'x' and 'y' arrays.

    """

    # protections
    if type(x) is not np.ndarray:
        raise ValueError("x=" + str(x) + " must be a numpy.ndarray")
    elif x.ndim != 1:
        raise ValueError("x.ndim=" + str(x.ndim) + " must be 1")
    if type(y) is not np.ndarray:
        raise ValueError("y=" + str(y) + " must be a numpy.ndarray")
    elif y.ndim != 1:
        raise ValueError("y.ndim=" + str(y.ndim) + " must be 1")
    npoints = x.size
    if npoints != y.size:
        raise ValueError("x.size != y.size")
    if reject is not None:
        if npoints != reject.size:
            raise ValueError("x.size != reject.size")
    if type(deg) not in [int, np.int64]:
        raise ValueError("deg=" + str(deg) +
                         " is not a valid integer")

    # select points for fit
    if reject is None:
        xfitted = np.copy(x)
        yfitted = np.copy(y)
        xrejected = None
        yrejected = None
        nfitted = npoints
        nrejected = 0
    else:
        xfitted = x[np.logical_not(reject)]
        yfitted = y[np.logical_not(reject)]
        xrejected = x[reject]
        yrejected = y[reject]
        # update number of points for fit
        nfitted = xfitted.size
        nrejected = sum(reject)

    if deg > nfitted - 1:
        raise ValueError("Insufficient nfitted=" + str(nfitted) +
                         " for deg=" + str(deg))

    # fit with requested degree (and raw polynomials)
    if deg == 0 and len(xfitted) == 1:  # constant (avoid fitting error)
        poly = Polynomial(yfitted[0])
    else:
        poly = Polynomial.fit(x=xfitted, y=yfitted, deg=deg)
    poly = Polynomial.cast(poly)

    # compute residuals
    yres = y - poly(x)  # of all the points
    yres_fitted = yfitted - poly(xfitted)  # points employed in the fit
    yres_rejected = None
    if nrejected > 0:
        yres_rejected = yrejected - poly(xrejected)  # points rejected

    if debugplot:
        print(">>> Polynomial fit:\n", poly)
        # define colors, markers and sizes for symbols
        if np.array(color).size == 1:
            mycolor = np.array([color] * npoints)
            if reject is not None:
                mycolor[reject] = 'r'
        elif np.array(color).size == npoints:
            mycolor = np.copy(np.array(color))
        elif np.array(color).shape[0] == npoints:  # assume rgb color
            mycolor = np.copy(np.array(color))
        else:
            raise ValueError("color=" + str(color) +
                             " doesn't have the expected dimension")
        if np.array(size).size == 1:
            mysize = np.repeat([size], npoints)
        elif np.array(size).size == npoints:
            mysize = np.copy(np.array(size))
        else:
            raise ValueError("size=" + str(size) +
                             " doesn't have the expected dimension")

        if reject is None:
            cfitted = np.copy(mycolor)
            crejected = None
            sfitted = np.copy(mysize)
            srejected = None
        else:
            cfitted = mycolor[np.logical_not(reject)]
            crejected = mycolor[reject]
            sfitted = mysize[np.logical_not(reject)]
            srejected = mysize[reject]

        fig = plt.figure()

        # residuals
        ax2 = fig.add_subplot(2, 1, 2)
        if xlabel is None:
            ax2.set_xlabel('x')
        else:
            ax2.set_xlabel(xlabel)
        ax2.set_ylabel('residuals')
        if xlim is None:
            xmin = min(x)
            xmax = max(x)
            dx = xmax - xmin
            if dx > 0:
                xmin -= dx/20
                xmax += dx/20
            else:
                xmin -= 0.5
                xmax += 0.5
        else:
            xmin, xmax = xlim
        ax2.set_xlim(xmin, xmax)
        ymin = min(yres_fitted)
        ymax = max(yres_fitted)
        dy = ymax - ymin
        if dy > 0:
            ymin -= dy/20
            ymax += dy/20
        else:
            ymin -= 0.5
            ymax += 0.5
        ax2.set_ylim(ymin, ymax)
        ax2.axhline(y=0.0, color="black", linestyle="dashed")
        ax2.scatter(xfitted, yres_fitted, color=cfitted,
                    marker='o',
                    edgecolor='k', s=sfitted)
        if nrejected > 0:
            ax2.scatter(xrejected, yres_rejected,
                        marker='x', s=srejected,
                        color=crejected)

        # original data and polynomial fit
        ax = fig.add_subplot(2, 1, 1, sharex=ax2)
        if ylabel is None:
            ax.set_ylabel('y')
        else:
            ax.set_ylabel(ylabel)
        ax.set_xlim(xmin, xmax)
        if ylim is None:
            ymin = min(y)
            ymax = max(y)
            dy = ymax - ymin
            if dy > 0:
                ymin -= dy/20
                ymax += dy/20
            else:
                ymin -= 0.5
                ymax += 0.5
        else:
            ymin, ymax = ylim
        ax.set_ylim(ymin, ymax)
        ax.scatter(xfitted, yfitted,
                   color=cfitted, marker='o', edgecolor='k',
                   s=sfitted, label="fitted data")
        xpol = np.linspace(start=xmin, stop=xmax, num=1000)
        ypol = poly(xpol)
        ax.plot(xpol, ypol, 'c-', label="fit")
        if nrejected > 0:
            ax.scatter(xrejected, yrejected,
                       marker='x', s=srejected, color=crejected,
                       label="rejected")

        # put a legend
        ax.legend(numpoints=1)

        # graph title
        if title is not None:
            plt.title(title)

        if show:
            plt.tight_layout()
            plt.show()

    # return result
    return poly, yres


def polfit_residuals_with_sigma_rejection(
        x, y, deg, times_sigma_reject,
        color='b', size=75,
        xlim=None, ylim=None,
        xlabel=None, ylabel=None, title=None,
        debugplot=False):
    """Polynomial fit with iterative rejection of points.

    This function makes use of function polfit_residuals for display
    purposes.

    Parameters
    ----------
    x : numpy array
        X coordinates of the data being fitted (floats).
    y : numpy array
        Y coordinates of the data being fitted (floats).
    deg : int
        Degree of the fitting polynomial.
    times_sigma_reject : float or None
        Number of times the standard deviation to reject points
        iteratively. If None, the fit does not reject any point.
    color : single character or 1d numpy array of characters
        Color for all the symbols (single character) or for each
        individual symbol (array of color names with the same length as
        'x' or 'y'). If 'color' is a single character, the rejected
        points are displayed in red color, whereas when 'color' is an
        array of color names, rejected points are displayed with the
        color provided in this array.
    size : int
        Marker size for all the symbols (single character) or for each
        individual symbol (array of integers with the same length as
        'x' or 'y').
    xlim : tuple (floats)
        Plot limits in the X axis.
    ylim : tuple (floats)
        Plot limits in the Y axis.
    xlabel : string
        Character string for label in X axis.
    ylabel : string
        Character string for label in y axis.
    title : string
        Character string for graph title.
    debugplot : bool
        If True, display intermediate plots and results.

    Returns
    -------
    poly : instance of Polynomial (numpy)
        Result from the polynomial fit using numpy Polynomial. Only
        points not flagged as rejected are employed in the fit.
    yres : numpy array
        Residuals from polynomial fit. Note that the residuals are
        computed for all the points, including the rejected ones. In
        this way the dimension of this array is the same as the
        dimensions of the input 'x' and 'y' arrays.
    reject : numpy array
        Boolean array indicating rejected points.

    """

    # protections
    if type(x) is not np.ndarray:
        raise ValueError("x=" + str(x) + " must be a numpy.ndarray")
    elif x.ndim != 1:
        raise ValueError("x.ndim=" + str(x.ndim) + " must be 1")
    if type(y) is not np.ndarray:
        raise ValueError("y=" + str(y) + " must be a numpy.ndarray")
    elif y.ndim != 1:
        raise ValueError("y.ndim=" + str(y.ndim) + " must be 1")
    npoints = x.size
    if npoints != y.size:
        raise ValueError("x.size != y.size")
    if type(deg) not in [int, np.int64]:
        raise ValueError("deg=" + str(deg) +
                         " is not a valid integer")
    if deg >= npoints:
        raise ValueError("Polynomial degree=" + str(deg) +
                         " can't be fitted with npoints=" + str(npoints))

    # initialize boolean rejection array
    reject = np.zeros(npoints, dtype=bool)

    # if there is no room to remove any point, compute a fit without
    # rejection
    if deg == npoints - 1:
        poly, yres = polfit_residuals(x=x, y=y, deg=deg, reject=None,
                                      color=color, size=size,
                                      xlim=xlim, ylim=ylim,
                                      xlabel=xlabel, ylabel=ylabel,
                                      title=title,
                                      debugplot=debugplot)
        return poly, yres, reject

    # main loop to reject points iteratively
    loop_to_reject_points = True
    poly = None
    yres = None
    while loop_to_reject_points:
        poly, yres = polfit_residuals(x=x, y=y, deg=deg, reject=reject)
        # check that there is room to remove a point with the current
        # polynomial degree
        npoints_effective = npoints - np.sum(reject)
        if deg < npoints_effective - 1:
            # determine robuts standard deviation, excluding points
            # already rejected
            # --- method 1 ---
            # yres_fitted = yres[np.logical_not(reject)]
            # q25, q75 = np.percentile(yres_fitted, q=[25.0, 75.0])
            # rms = 0.7413 * (q75 - q25)
            # --- method 2 ---
            yres_fitted = np.abs(yres[np.logical_not(reject)])
            rms = np.median(yres_fitted)
            if debugplot:
                print("--> robust rms:", rms)
            # reject fitted point exceeding the threshold with the
            # largest deviation (note: with this method only one point
            # is removed in each iteration of the loop; this allows the
            # recomputation of the polynomial fit which, sometimes,
            # transforms deviant points into good ones)
            index_to_remove = []
            for i in range(npoints):
                if not reject[i]:
                    if np.abs(yres[i]) > times_sigma_reject * rms:
                        index_to_remove.append(i)
                        if debugplot:
                            print('--> suspicious point #', i + 1)
            if len(index_to_remove) == 0:
                if debugplot:
                    print('==> no need to remove any point')
                loop_to_reject_points = False
            else:
                imax = np.argmax(np.abs(yres[index_to_remove]))
                reject[index_to_remove[imax]] = True
                if debugplot:
                    print('==> removing point #', index_to_remove[imax] + 1)
        else:
            loop_to_reject_points = False

    # plot final fit in case it has not been already shown
    if debugplot:
        print(' ')

    # return result
    return poly, yres, reject
